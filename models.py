"""Data models for the OpenEnv email triage environment."""

from typing import Literal

from pydantic import BaseModel, field_validator

OPEN_INTERVAL_EPSILON = 1e-2


def _strict_open_unit_interval(raw_value: float) -> float:
    """Clamp numeric values to the strict open interval (0, 1)."""
    numeric_value = float(raw_value)
    if numeric_value <= 0.0:
        return OPEN_INTERVAL_EPSILON
    if numeric_value >= 1.0:
        return 1.0 - OPEN_INTERVAL_EPSILON
    return numeric_value


class EmailObservation(BaseModel):
    """Represents the email context visible to the agent at each step."""

    email_id: str
    subject: str
    body: str
    sender: str
    timestamp: str
    thread_history: list[str]
    task_id: str
    step_number: int
    total_emails: int


class TriageAction(BaseModel):
    """Represents the action chosen by the agent for an email."""

    label: Literal["urgent", "normal", "spam", "archive"]
    summary: str
    route_to: str


class RewardResult(BaseModel):
    """Represents deterministic grading output before reward shaping."""

    score: float
    breakdown: dict[str, float]
    feedback: str

    @field_validator("score")
    @classmethod
    def _validate_score(cls, value: float) -> float:
        return _strict_open_unit_interval(value)


class EnvironmentState(BaseModel):
    """Represents full internal environment state for debugging and evaluation."""

    task_id: str
    current_step: int
    total_steps: int
    done: bool
    action_history: list[TriageAction]
    reward_history: list[float]

    @field_validator("reward_history")
    @classmethod
    def _validate_reward_history(cls, values: list[float]) -> list[float]:
        return [_strict_open_unit_interval(value) for value in values]


class StepResult(BaseModel):
    """Represents the standardized output of environment step calls."""

    observation: EmailObservation
    reward: float
    done: bool
    info: dict[str, str | int | float | bool]

    @field_validator("reward")
    @classmethod
    def _validate_reward(cls, value: float) -> float:
        return _strict_open_unit_interval(value)


class ResetResult(BaseModel):
    """Represents the standardized output of environment reset calls."""

    observation: EmailObservation
    info: dict[str, str | int | float | bool]
