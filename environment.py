"""Core OpenEnv email triage environment implementation."""

import os
from typing import cast

from pydantic import ValidationError

from graders import SCORE_EPSILON, grade_easy, grade_hard, grade_medium_step
from models import (
    EmailObservation,
    EnvironmentState,
    ResetResult,
    RewardResult,
    StepResult,
    TriageAction,
)
from tasks import get_task_definition


class EmailTriageEnv:
    """Deterministic email triage environment implementing reset, step, and state."""

    def __init__(
        self,
        task_id: str,
        scenario_index: int = 0,
        split: str | None = None,
        runtime_options: dict[str, object] | None = None,
    ) -> None:
        """Initialize environment with a selected task.

        Args:
            task_id: Task identifier such as task_easy, task_medium, or task_hard.
            scenario_index: Deterministic scenario index within the task pool.
            split: Scenario split, either public or private_eval.
            runtime_options: Optional deterministic runtime controls for task generation.
        """
        self.task_id = task_id
        self._episode_index = max(0, scenario_index)
        self.split = split or os.getenv("OPENENV_EVAL_SPLIT", "public")
        self.runtime_options = runtime_options or {}
        self._task_definition = get_task_definition(
            task_id,
            self._episode_index,
            self.split,
            self.runtime_options,
        )
        self._scenario_id = str(self._task_definition.get("scenario_id", "unknown"))
        self._emails = cast(list[dict[str, object]], self._task_definition.get("emails", []))
        self._ground_truth = cast(
            list[dict[str, object]], self._task_definition.get("ground_truth", [])
        )

        self._current_index = 0
        self._current_step = 0
        self._done = False
        self._max_steps = max(10, len(self._emails) + 5)
        self._action_history: list[TriageAction] = []
        self._reward_history: list[float] = []
        self._base_score_history: list[float] = []
        self._generated_followups = 0
        self._max_generated_followups = 4
        self._followup_quality_threshold = 0.7
        self._configure_runtime_controls()

    def reset(self) -> ResetResult:
        """Reset episode state and return the first observation.

        Returns:
            ResetResult containing first observation and metadata.
        """
        self._task_definition = get_task_definition(
            self.task_id,
            self._episode_index,
            self.split,
            self.runtime_options,
        )
        self._scenario_id = str(self._task_definition.get("scenario_id", "unknown"))
        self._emails = cast(list[dict[str, object]], self._task_definition.get("emails", []))
        self._ground_truth = cast(
            list[dict[str, object]], self._task_definition.get("ground_truth", [])
        )

        self._current_index = 0
        self._current_step = 0
        self._done = False
        self._max_steps = max(10, len(self._emails) + 5)
        self._action_history = []
        self._reward_history = []
        self._base_score_history = []
        self._generated_followups = 0
        self._configure_runtime_controls()
        self._episode_index += 1

        first_observation = self._build_observation(self._current_index)
        return ResetResult(
            observation=first_observation,
            info={
                "task_id": self.task_id,
                "scenario_id": self._scenario_id,
                "split": self.split,
                "step": self._current_step,
                "emails_total": len(self._emails),
                "task_description": str(self._task_definition.get("description", "")),
            },
        )

    def step(self, action: TriageAction) -> StepResult:
        """Apply an action and return StepResult.

        Args:
            action: Proposed triage action.

        Returns:
            StepResult with next observation, reward, done flag, and metadata.
        """
        if self._done:
            return StepResult(
                observation=self._terminal_observation(),
                reward=SCORE_EPSILON,
                done=True,
                info={
                    "task_id": self.task_id,
                    "scenario_id": self._scenario_id,
                    "split": self.split,
                    "step": self._current_step,
                    "already_done": True,
                },
            )

        try:
            validated_action = TriageAction.model_validate(action)
        except ValidationError as validation_error:
            self._current_step += 1
            self._reward_history.append(SCORE_EPSILON)
            self._done = self._current_step >= self._max_steps
            return StepResult(
                observation=self._build_observation(self._current_index),
                reward=SCORE_EPSILON,
                done=self._done,
                info={
                    "task_id": self.task_id,
                    "scenario_id": self._scenario_id,
                    "split": self.split,
                    "step": self._current_step,
                    "emails_total": len(self._emails),
                    "emails_processed": self._current_index,
                    "emails_remaining": max(len(self._emails) - self._current_index, 0),
                    "validation_error": str(validation_error),
                },
            )

        base_result = self._grade_current_step(validated_action)
        base_score = base_result.score
        previous_base_score = self._base_score_history[-1] if self._base_score_history else None
        progress_signal = self._compute_progress_signal(base_score, previous_base_score)

        truth_for_step = (
            self._ground_truth[min(self._current_index, len(self._ground_truth) - 1)]
            if self._ground_truth
            else {}
        )
        self._maybe_enqueue_follow_up(validated_action, truth_for_step, base_score)

        self._action_history.append(validated_action)
        self._base_score_history.append(base_score)
        self._current_step += 1

        penalties = self._compute_penalties(validated_action)
        trajectory_bonus = self._compute_trajectory_bonus()
        step_cost = self._compute_step_cost()
        final_reward = self._clip_reward(
            base_score + progress_signal + trajectory_bonus - penalties - step_cost
        )

        self._reward_history.append(final_reward)

        if self._current_index < len(self._emails):
            self._current_index += 1

        all_emails_processed = self._current_index >= len(self._emails)
        self._done = all_emails_processed or self._current_step >= self._max_steps

        next_observation = (
            self._terminal_observation()
            if self._done
            else self._build_observation(self._current_index)
        )

        info = {
            "task_id": self.task_id,
            "scenario_id": self._scenario_id,
            "split": self.split,
            "step": self._current_step,
            "emails_total": len(self._emails),
            "emails_processed": min(self._current_index, len(self._emails)),
            "emails_remaining": max(len(self._emails) - self._current_index, 0),
            "base_score": float(base_score),
            "progress_signal": float(progress_signal),
            "step_cost": float(step_cost),
            "penalties": float(penalties),
            "trajectory_bonus": float(trajectory_bonus),
            "grading_feedback": base_result.feedback,
        }
        for breakdown_key, breakdown_value in base_result.breakdown.items():
            if isinstance(breakdown_value, (int, float)):
                info[f"grade_{breakdown_key}"] = float(breakdown_value)

        return StepResult(
            observation=next_observation,
            reward=final_reward,
            done=self._done,
            info=info,
        )

    def _maybe_enqueue_follow_up(
        self,
        action: TriageAction,
        truth: dict[str, object],
        base_score: float,
    ) -> None:
        """Insert deterministic escalation follow-up emails for production mode."""
        if self.task_id != "task_production":
            return
        if self._generated_followups >= self._max_generated_followups:
            return
        if not self._emails:
            return

        expected_label = str(truth.get("label", ""))
        expected_route = str(truth.get("route_to", "general"))
        is_missed_critical = (
            expected_label == "urgent"
            and (action.label != "urgent" or expected_route not in action.route_to.lower())
        )
        if not is_missed_critical and base_score >= self._followup_quality_threshold:
            return

        source_email = self._emails[min(self._current_index, len(self._emails) - 1)]
        source_subject = str(source_email.get("subject", "Inbox incident"))
        source_timestamp = str(source_email.get("timestamp", "2026-04-03T00:00:00Z"))

        followup_email = {
            "email_id": f"followup-{self._scenario_id}-{self._generated_followups + 1}",
            "subject": f"Escalation follow-up: {source_subject}",
            "body": (
                "Automated escalation triggered because prior triage appears incomplete. "
                "Please route to the responsible team and provide a clear summary now."
            ),
            "sender": "incident-control@acme-enterprise.com",
            "timestamp": source_timestamp,
            "thread_history": [f"Previous message subject: {source_subject}"],
        }
        followup_truth = {
            "label": "urgent",
            "route_to": expected_route,
            "priority_weight": min(max(float(truth.get("priority_weight", 1.5)) + 0.2, 1.5), 2.0),
            "summary_keywords": ["escalation", "follow-up", expected_route],
        }

        insert_at = min(self._current_index + 1, len(self._emails))
        self._emails.insert(insert_at, followup_email)
        self._ground_truth.insert(insert_at, followup_truth)
        self._generated_followups += 1

    def _configure_runtime_controls(self) -> None:
        """Apply deterministic runtime control options for production simulator."""
        if self.task_id != "task_production":
            self._max_generated_followups = 4
            self._followup_quality_threshold = 0.7
            return

        escalation_mode = str(self.runtime_options.get("escalation_mode", "normal")).lower()
        escalation_map = {
            "low": (2, 0.55),
            "normal": (4, 0.7),
            "high": (8, 0.85),
        }
        max_followups, threshold = escalation_map.get(escalation_mode, escalation_map["normal"])
        self._max_generated_followups = max_followups
        self._followup_quality_threshold = threshold

    def state(self) -> EnvironmentState:
        """Return read-only snapshot of full internal state.

        Returns:
            EnvironmentState with progress and history.
        """
        return EnvironmentState(
            task_id=self.task_id,
            current_step=self._current_step,
            total_steps=self._max_steps,
            done=self._done,
            action_history=list(self._action_history),
            reward_history=list(self._reward_history),
        )

    def _build_observation(self, email_index: int) -> EmailObservation:
        """Build observation for the email at a given index.

        Args:
            email_index: Zero-based email index.

        Returns:
            EmailObservation for the selected email or terminal placeholder.
        """
        if not self._emails:
            return self._terminal_observation()

        safe_index = min(max(email_index, 0), len(self._emails) - 1)
        email_payload = self._emails[safe_index]

        return EmailObservation(
            email_id=str(email_payload.get("email_id", "")),
            subject=str(email_payload.get("subject", "")),
            body=str(email_payload.get("body", "")),
            sender=str(email_payload.get("sender", "")),
            timestamp=str(email_payload.get("timestamp", "")),
            thread_history=[str(item) for item in email_payload.get("thread_history", [])],
            task_id=self.task_id,
            step_number=self._current_step,
            total_emails=len(self._emails),
        )

    def _terminal_observation(self) -> EmailObservation:
        """Build terminal observation returned when episode is complete.

        Returns:
            Terminal EmailObservation payload.
        """
        return EmailObservation(
            email_id="terminal",
            subject="Episode complete",
            body="No further emails remain for this task.",
            sender="system",
            timestamp="",
            thread_history=[],
            task_id=self.task_id,
            step_number=self._current_step,
            total_emails=len(self._emails),
        )

    def _grade_current_step(self, action: TriageAction) -> RewardResult:
        """Select deterministic grader based on task and current progress.

        Args:
            action: Validated action for the current step.

        Returns:
            RewardResult from task-specific grader.
        """
        if not self._ground_truth:
            return RewardResult(
                score=SCORE_EPSILON,
                breakdown={"missing_ground_truth": 1.0 - SCORE_EPSILON},
                feedback="Missing ground truth for task.",
            )

        if self.task_id == "task_easy":
            truth = self._ground_truth[min(self._current_index, len(self._ground_truth) - 1)]
            return grade_easy(action, truth)

        if self.task_id == "task_medium":
            truth = self._ground_truth[min(self._current_index, len(self._ground_truth) - 1)]
            return grade_medium_step(action, truth)

        truth = self._ground_truth[min(self._current_index, len(self._ground_truth) - 1)]
        return grade_hard(action, truth)

    def _compute_penalties(self, action: TriageAction) -> float:
        """Compute deterministic penalties according to reward policy.

        Args:
            action: Validated action for the step.

        Returns:
            Total penalty value for current step.
        """
        penalty_total = 0.0

        summary_too_short = len(action.summary.strip()) < 10
        if action.label == "archive" and summary_too_short:
            penalty_total += 0.5

        if self._is_repeated_action_pattern(action):
            penalty_total += 0.3

        return penalty_total

    def _compute_progress_signal(
        self,
        base_score: float,
        previous_base_score: float | None,
    ) -> float:
        """Compute dense partial-progress reward independent of final completion.

        Args:
            base_score: Current-step base grade in [0.0, 1.0].
            previous_base_score: Previous step base grade when available.

        Returns:
            Small positive/negative signal reflecting progress and quality trend.
        """
        total_emails = max(len(self._emails), 1)
        progress_ratio = min(1.0, (self._current_index + 1) / total_emails)

        completion_signal = 0.05 * progress_ratio
        quality_signal = 0.05 * self._clip_reward(base_score)

        trend_signal = 0.0
        if previous_base_score is not None:
            delta = base_score - previous_base_score
            trend_signal = max(-0.02, min(0.03, delta * 0.1))

        return completion_signal + quality_signal + trend_signal

    def _compute_step_cost(self) -> float:
        """Return a gentle efficiency cost that grows with episode length."""
        normalized_step = self._current_step / max(self._max_steps, 1)
        return 0.005 + (0.01 * normalized_step)

    def _compute_trajectory_bonus(self) -> float:
        """Return trajectory bonus when episode completion quality is high.

        Returns:
            0.2 when mean base score is above threshold at completion, else 0.0.
        """
        if not self._base_score_history:
            return 0.0

        all_emails_done_after_step = self._current_index + 1 >= len(self._emails)
        if not all_emails_done_after_step:
            return 0.0

        mean_base = sum(self._base_score_history) / len(self._base_score_history)
        return 0.2 if mean_base > 0.8 else 0.0

    def _is_repeated_action_pattern(self, action: TriageAction) -> bool:
        """Detect whether same action appears three times consecutively.

        Args:
            action: Current action.

        Returns:
            True when repeated label and route occur three times in a row.
        """
        if len(self._action_history) < 2:
            return False

        previous_action = self._action_history[-1]
        older_action = self._action_history[-2]

        return (
            previous_action.label == older_action.label == action.label
            and previous_action.route_to.strip().lower()
            == older_action.route_to.strip().lower()
            == action.route_to.strip().lower()
        )

    def _clip_reward(self, reward_value: float) -> float:
        """Clip reward to the strict range (0.0, 1.0).

        Args:
            reward_value: Raw reward value.

        Returns:
            Clipped reward.
        """
        return max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, reward_value))
