from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


SEVERITY_CATEGORIES: tuple[str, ...] = (
    "cardiac",
    "trauma",
    "respiratory",
    "obstetric",
    "neurological",
    "pediatric",
)

VALID_EVENT_FLAGS: tuple[str, ...] = (
    "monsoon_surge",
    "festival_traffic",
    "power_outage",
    "mass_casualty",
    "non_critical_backlog",
)

VALID_CITY_CONTEXTS: tuple[str, ...] = (
    "normal",
    "monsoon_season",
    "festival_day",
    "disaster_zone",
)

VALID_PRIORITY_LEVELS: tuple[str, ...] = (
    "low",
    "medium",
    "high",
    "critical",
)

VALID_SURGE_REDIRECTS: tuple[str, ...] = (
    "mutual_aid",
    "defer_non_critical",
    "request_backup",
    "activate_protocol",
)

VALID_FATIGUE_STYLES: tuple[str, ...] = (
    "rotate_operator",
    "take_micro_break",
    "request_supervisor",
)

VALID_ESCALATION_TARGETS: tuple[str, ...] = (
    "hospital",
    "police",
    "fire",
    "disaster_management",
)

VALID_HANDOFF_QUALITIES: tuple[str, ...] = (
    "poor",
    "standard",
    "thorough",
)

ACTION_TYPES: tuple[str, ...] = (
    "dispatch_ambulance",
    "triage_call",
    "handle_surge",
    "manage_fatigue",
    "escalate_incident",
    "defer_call",
    "request_mutual_aid",
    "close_shift",
)

TASK_NAMES: tuple[str, ...] = ("EASY", "MEDIUM", "HARD")


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp_reward(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


class Overload108Observation(BaseSchema):
    caller_severity_vector: dict[str, float] = Field(default_factory=dict)
    ambulances_available: int = Field(ge=0, le=20)
    ambulances_en_route: int = Field(ge=0, le=20)
    operator_fatigue: float = Field(ge=0.0, le=1.0)
    response_time_pressure: float = Field(ge=0.0, le=1.0)
    queue_length: int = Field(ge=0)
    incident_cascade_risk: float = Field(ge=0.0, le=1.0)
    current_call_id: str = ""
    event_flags: list[str] = Field(default_factory=list)
    city_context: Literal["normal", "monsoon_season", "festival_day", "disaster_zone"] = "normal"
    recent_dispatch_accuracy: float = Field(ge=0.0, le=1.0, default=0.5)
    streak: int = Field(ge=0, default=0)

    @field_validator("caller_severity_vector")
    @classmethod
    def validate_severity_vector(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for category in SEVERITY_CATEGORIES:
            if category not in value:
                raise ValueError(f"missing severity category: {category}")
            normalized[category] = _clamp_unit_interval(value[category])
        extra = sorted(set(value) - set(SEVERITY_CATEGORIES))
        if extra:
            raise ValueError(f"unexpected severity categories: {extra}")
        return normalized

    @field_validator("event_flags")
    @classmethod
    def validate_event_flags(cls, value: list[str]) -> list[str]:
        invalid = sorted(set(value) - set(VALID_EVENT_FLAGS))
        if invalid:
            raise ValueError(f"unexpected event flags: {invalid}")
        return list(dict.fromkeys(value))

    @field_validator("operator_fatigue", "response_time_pressure", "incident_cascade_risk", "recent_dispatch_accuracy")
    @classmethod
    def validate_unit_interval_fields(cls, value: float) -> float:
        return _clamp_unit_interval(value)


class Overload108FullState(BaseSchema):
    """Full internal state including hidden true_severity. Never exposed to agent."""
    caller_severity_vector: dict[str, float] = Field(default_factory=dict)
    true_severity: dict[str, float] = Field(default_factory=dict)
    ambulances_available: int = Field(ge=0, le=20)
    ambulances_en_route: int = Field(ge=0, le=20)
    operator_fatigue: float = Field(ge=0.0, le=1.0)
    response_time_pressure: float = Field(ge=0.0, le=1.0)
    queue_length: int = Field(ge=0)
    incident_cascade_risk: float = Field(ge=0.0, le=1.0)
    current_call_id: str = ""
    event_flags: list[str] = Field(default_factory=list)
    city_context: Literal["normal", "monsoon_season", "festival_day", "disaster_zone"] = "normal"
    recent_dispatch_accuracy: float = Field(ge=0.0, le=1.0, default=0.5)
    streak: int = Field(ge=0, default=0)

    def to_observation(self) -> Overload108Observation:
        return Overload108Observation(
            caller_severity_vector=dict(self.caller_severity_vector),
            ambulances_available=self.ambulances_available,
            ambulances_en_route=self.ambulances_en_route,
            operator_fatigue=self.operator_fatigue,
            response_time_pressure=self.response_time_pressure,
            queue_length=self.queue_length,
            incident_cascade_risk=self.incident_cascade_risk,
            current_call_id=self.current_call_id,
            event_flags=list(self.event_flags),
            city_context=self.city_context,
            recent_dispatch_accuracy=self.recent_dispatch_accuracy,
            streak=self.streak,
        )


class Overload108Action(BaseSchema):
    action_type: Literal[
        "dispatch_ambulance",
        "triage_call",
        "handle_surge",
        "manage_fatigue",
        "escalate_incident",
        "defer_call",
        "request_mutual_aid",
        "close_shift",
    ]
    params: dict[str, Any] = Field(default_factory=dict)


class Overload108TaskSpec(BaseSchema):
    task_name: Literal["EASY", "MEDIUM", "HARD"]
    max_steps: int = Field(ge=1)
    success_threshold: float = Field(ge=0.0, le=1.0)
    initial_state: Overload108FullState
    grader: str = ""
    description: str = ""

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        return value.strip()


class Overload108EnvironmentState(BaseSchema):
    task_name: Literal["EASY", "MEDIUM", "HARD"]
    step_count: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    success_threshold: float = Field(ge=0.0, le=1.0)
    observation: Overload108Observation
    full_state: Overload108FullState | None = None
    reward_history: list[float] = Field(default_factory=list)
    action_history: list[Overload108Action] = Field(default_factory=list)
    done: bool = False
    success: bool = False

    @model_validator(mode="after")
    def validate_reward_history(self) -> "Overload108EnvironmentState":
        self.reward_history = [_clamp_reward(r) for r in self.reward_history]
        return self


class Overload108StepResult(BaseSchema):
    observation: Overload108Observation
    reward: float = Field(ge=-1.0, le=1.0)
    done: bool
    success: bool
    info: dict[str, Any] = Field(default_factory=dict)


class Overload108ResetPayload(BaseSchema):
    task_name: Literal["EASY", "MEDIUM", "HARD"]


class Overload108ResetResponse(BaseSchema):
    state: Overload108EnvironmentState


class Overload108StepPayload(BaseSchema):
    action: Overload108Action


class Overload108StepResponse(BaseSchema):
    state: Overload108EnvironmentState
    result: Overload108StepResult


Action = Overload108Action
Observation = Overload108Observation
State = Overload108EnvironmentState


__all__ = [
    "ACTION_TYPES",
    "Action",
    "BaseSchema",
    "Observation",
    "Overload108Action",
    "Overload108EnvironmentState",
    "Overload108FullState",
    "Overload108Observation",
    "Overload108ResetPayload",
    "Overload108ResetResponse",
    "Overload108StepPayload",
    "Overload108StepResponse",
    "Overload108StepResult",
    "Overload108TaskSpec",
    "SEVERITY_CATEGORIES",
    "State",
    "TASK_NAMES",
    "VALID_CITY_CONTEXTS",
    "VALID_ESCALATION_TARGETS",
    "VALID_EVENT_FLAGS",
    "VALID_FATIGUE_STYLES",
    "VALID_HANDOFF_QUALITIES",
    "VALID_PRIORITY_LEVELS",
    "VALID_SURGE_REDIRECTS",
]
