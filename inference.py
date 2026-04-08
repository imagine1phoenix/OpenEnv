"""Inference script for OpenEnv email triage with strict stdout event format."""

import argparse
import json
import os
import re
import time
from typing import Any

from openai import OpenAI

from environment import EmailTriageEnv
from models import EmailObservation, TriageAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "openenv-email-triage"
MAX_STEPS = 30
TEMPERATURE = 0.2
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.5
LOG_SCORE_EPSILON = 1e-2
DEFAULT_RUNTIME_BUDGET_SECONDS = int(os.getenv("INFERENCE_RUNTIME_BUDGET_SECONDS", "1140"))
DEFAULT_REQUEST_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_REQUEST_TIMEOUT_SECONDS", "12"))

SYSTEM_PROMPT = (
    "You are an email triage assistant. For each email, prioritize risk/time impact, "
    "categorize with one label (urgent|normal|spam|archive), route to the best team, "
    "and summarize the key evidence. Return one JSON object with keys label, summary, route_to."
)

FALLBACK_ACTION = {
    "label": "normal",
    "summary": "Unable to parse response",
    "route_to": "general",
}

TASK_MAP = {
    "1": "task_easy",
    "2": "task_medium",
    "3": "task_hard",
    "4": "task_production",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for task and optional model override."""
    parser = argparse.ArgumentParser(description="Run OpenEnv email triage inference.")
    parser.add_argument(
        "--task",
        default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Task selection: 1, 2, 3, 4, or all.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override. Falls back to MODEL_NAME environment variable.",
    )
    parser.add_argument(
        "--split",
        default=os.getenv("OPENENV_EVAL_SPLIT", "public"),
        choices=["public", "private_eval"],
        help="Scenario split to evaluate.",
    )
    parser.add_argument(
        "--episodes-per-task",
        default=1,
        type=int,
        help="Number of deterministic scenarios to evaluate per task.",
    )
    parser.add_argument(
        "--runtime-budget-seconds",
        default=DEFAULT_RUNTIME_BUDGET_SECONDS,
        type=int,
        help="Global wall-clock budget for the full script run.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        type=float,
        help="Timeout per LLM request.",
    )
    parser.add_argument(
        "--production-profile",
        default="standard",
        choices=["light", "standard", "heavy"],
        help="Runtime workload profile used for task 4 episodes.",
    )
    parser.add_argument(
        "--business-hours-mode",
        action="store_true",
        help="If set, task 4 timestamps focus on business-hours windows.",
    )
    parser.add_argument(
        "--escalation-mode",
        default="normal",
        choices=["low", "normal", "high"],
        help="Escalation strictness for task 4 follow-up generation.",
    )
    return parser.parse_args()


def validate_runtime_config(model_name: str | None) -> str:
    """Validate required runtime settings and return effective model name."""
    if not API_KEY:
        raise ValueError("Missing HF_TOKEN or API_KEY environment variable.")

    effective_model = model_name or MODEL_NAME
    return effective_model


def log_start(task_name: str, benchmark_name: str, model_name: str) -> None:
    """Emit mandatory START line."""
    print(
        f"[START] task={task_name} env={benchmark_name} model={model_name}",
        flush=True,
    )


def _format_open_score(value: float) -> str:
    """Format scores in strict-open range while preserving .2f log contract."""
    clamped = max(LOG_SCORE_EPSILON, min(1.0 - LOG_SCORE_EPSILON, float(value)))
    return f"{clamped:.2f}"


def _strict_task_score(raw_score: float) -> float:
    """Return task score in strict-open interval for evaluator compatibility."""
    return max(LOG_SCORE_EPSILON, min(1.0 - LOG_SCORE_EPSILON, float(raw_score)))


def log_step(step: int, action_str: str, reward: float, done: bool, error: str | None) -> None:
    """Emit mandatory STEP line."""
    error_value = error if error else "null"
    done_value = str(done).lower()
    print(
        f"[STEP] step={step} action={action_str} reward={_format_open_score(reward)} "
        f"done={done_value} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float], task_score: float) -> None:
    """Emit mandatory END line."""
    rewards_str = ",".join(_format_open_score(reward) for reward in rewards)
    strict_task_score = _strict_task_score(task_score)
    print(
        f"[END] task_score={_format_open_score(strict_task_score)} "
        f"success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(observation: EmailObservation, history: list[str]) -> str:
    """Build model prompt from current observation and recent history."""
    recent_history = "\n".join(history[-5:]) if history else "None"
    return (
        f"email_id: {observation.email_id}\n"
        f"subject: {observation.subject}\n"
        f"sender: {observation.sender}\n"
        f"timestamp: {observation.timestamp}\n"
        f"body: {observation.body}\n"
        f"thread_history: {observation.thread_history}\n"
        f"task_id: {observation.task_id}\n"
        f"step_number: {observation.step_number}\n"
        f"total_emails: {observation.total_emails}\n\n"
        f"recent_history:\n{recent_history}\n\n"
        "Return exactly one JSON object with label, summary, route_to."
    )


def strip_action_prefixes(response_text: str) -> str:
    """Remove common formatting wrappers before parsing model output."""
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = re.sub(r"^(next\s+action|action)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def parse_text_action(cleaned_text: str) -> dict[str, str]:
    """Parse action from free-form text with deterministic regex fallback."""
    result: dict[str, str] = {}

    label_match = re.search(
        r"(?:\"label\"|label)\s*[:=]\s*\"?(urgent|normal|spam|archive)\"?",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    if label_match:
        result["label"] = label_match.group(1).lower()

    route_match = re.search(
        r"(?:\"route_to\"|route_to|route)\s*[:=]\s*\"?([a-zA-Z0-9_\-/ ]+)\"?",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    if route_match:
        result["route_to"] = route_match.group(1).strip().lower()

    summary_match = re.search(
        r"(?:\"summary\"|summary)\s*[:=]\s*\"?([^\"\n]+)\"?",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    if summary_match:
        result["summary"] = summary_match.group(1).strip()

    return result


def parse_action_response(response_text: str) -> TriageAction:
    """Parse model response into a valid TriageAction with fallback behavior."""
    cleaned_text = strip_action_prefixes(response_text)
    parsed_payload: dict[str, Any] = {}

    json_start = cleaned_text.find("{")
    json_end = cleaned_text.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        candidate = cleaned_text[json_start : json_end + 1]
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                parsed_payload = loaded
        except json.JSONDecodeError:
            parsed_payload = {}

    if not parsed_payload:
        parsed_payload = parse_text_action(cleaned_text)

    fallback_copy = dict(FALLBACK_ACTION)
    fallback_copy.update(parsed_payload)

    try:
        return TriageAction.model_validate(fallback_copy)
    except Exception:
        return TriageAction.model_validate(FALLBACK_ACTION)


def action_to_log_string(action: TriageAction) -> str:
    """Return single-line action string for required STEP logging."""
    return json.dumps(action.model_dump(), separators=(",", ":"), ensure_ascii=True)


def run_episode(
    client: OpenAI,
    model_name: str,
    task_id: str,
    scenario_index: int,
    eval_split: str,
    deadline: float,
    request_timeout_seconds: float,
    runtime_options: dict[str, Any] | None = None,
) -> None:
    """Run one episode and emit strict START/STEP/END lines."""
    rewards: list[float] = []
    steps_taken = 0
    success = False
    final_task_score = LOG_SCORE_EPSILON
    env: EmailTriageEnv | None = None

    log_start(task_name=task_id, benchmark_name=BENCHMARK, model_name=model_name)

    try:
        env = EmailTriageEnv(
            task_id=task_id,
            scenario_index=scenario_index,
            split=eval_split,
            runtime_options=runtime_options,
        )
        reset_result = env.reset()
        observation = reset_result.observation
        history: list[str] = []

        for step in range(1, MAX_STEPS + 1):
            if time.monotonic() >= deadline:
                break

            prompt = build_user_prompt(observation, history)

            response_text = ""
            try:
                remaining = max(1.0, deadline - time.monotonic())
                timeout_seconds = max(
                    1.0,
                    min(float(request_timeout_seconds), float(remaining)),
                )
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                    timeout=timeout_seconds,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception:
                response_text = ""

            action = parse_action_response(response_text)
            step_result = env.step(action)

            reward = _strict_task_score(float(step_result.reward))
            done = bool(step_result.done)
            error_raw = step_result.info.get("validation_error")
            error = str(error_raw) if isinstance(error_raw, str) else None

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action_str=action_to_log_string(action),
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"step={step} action={action.label}/{action.route_to} reward={_format_open_score(reward)}"
            )
            observation = step_result.observation

            if done:
                break

        if not rewards:
            rewards.append(LOG_SCORE_EPSILON)

        final_task_score = _strict_task_score(sum(rewards) / len(rewards))
        success = final_task_score >= SUCCESS_SCORE_THRESHOLD
    except Exception:
        if not rewards:
            rewards.append(LOG_SCORE_EPSILON)
        final_task_score = _strict_task_score(sum(rewards) / len(rewards))
        success = False
    finally:
        if env is not None:
            close_method = getattr(env, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    pass

        log_end(
            success=success,
            steps=steps_taken,
            rewards=rewards,
            task_score=final_task_score,
        )


def main() -> None:
    """Entrypoint for running one or many tasks with strict stdout logs."""
    args = parse_args()
    deadline = time.monotonic() + max(args.runtime_budget_seconds, 1)
    request_timeout_seconds = max(float(args.request_timeout_seconds), 1.0)

    try:
        effective_model = validate_runtime_config(args.model)
    except ValueError as error:
        print(str(error), flush=True)
        raise SystemExit(1) from error

    _ = LOCAL_IMAGE_NAME

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    task_ids = [TASK_MAP[args.task]] if args.task in TASK_MAP else list(TASK_MAP.values())
    for task_id in task_ids:
        runtime_options = None
        if task_id == "task_production":
            runtime_options = {
                "production_profile": args.production_profile,
                "business_hours_mode": args.business_hours_mode,
                "escalation_mode": args.escalation_mode,
            }
        for scenario_index in range(max(args.episodes_per_task, 1)):
            run_episode(
                client=client,
                model_name=effective_model,
                task_id=task_id,
                scenario_index=scenario_index,
                eval_split=args.split,
                deadline=deadline,
                request_timeout_seconds=request_timeout_seconds,
                runtime_options=runtime_options,
            )


if __name__ == "__main__":
    main()
