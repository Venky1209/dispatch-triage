from __future__ import annotations

from typing import Any, Final

from .models import (
    ACTION_TYPES,
    Overload108FullState,
    Overload108TaskSpec,
    SEVERITY_CATEGORIES,
    TASK_NAMES,
)


TASK_MAX_STEPS: Final[dict[str, int]] = {
    "EASY": 8,
    "MEDIUM": 15,
    "HARD": 25,
}

TASK_SUCCESS_THRESHOLDS: Final[dict[str, float]] = {
    "EASY": 0.50,
    "MEDIUM": 0.62,
    "HARD": 0.72,
}

TASK_INITIAL_STATES: Final[dict[str, dict[str, object]]] = {
    "EASY": {
        "operator_fatigue": 0.2,
        "ambulances_available": 15,
        "ambulances_en_route": 0,
        "queue_length": 5,
        "incident_cascade_risk": 0.1,
        "response_time_pressure": 0.2,
        "event_flags": ["non_critical_backlog"],
        "city_context": "normal",
        "recent_dispatch_accuracy": 0.5,
        "streak": 0,
        "caller_severity_vector": {
            "cardiac": 0.3,
            "trauma": 0.2,
            "respiratory": 0.4,
            "obstetric": 0.1,
            "neurological": 0.2,
            "pediatric": 0.3,
        },
        "true_severity": {
            "cardiac": 0.45,
            "trauma": 0.25,
            "respiratory": 0.35,
            "obstetric": 0.15,
            "neurological": 0.30,
            "pediatric": 0.20,
        },
    },
    "MEDIUM": {
        "operator_fatigue": 0.5,
        "ambulances_available": 8,
        "ambulances_en_route": 3,
        "queue_length": 18,
        "incident_cascade_risk": 0.45,
        "response_time_pressure": 0.55,
        "event_flags": ["monsoon_surge", "power_outage"],
        "city_context": "monsoon_season",
        "recent_dispatch_accuracy": 0.4,
        "streak": 2,
        "caller_severity_vector": {
            "cardiac": 0.5,
            "trauma": 0.6,
            "respiratory": 0.5,
            "obstetric": 0.4,
            "neurological": 0.3,
            "pediatric": 0.5,
        },
        "true_severity": {
            "cardiac": 0.65,
            "trauma": 0.75,
            "respiratory": 0.40,
            "obstetric": 0.55,
            "neurological": 0.45,
            "pediatric": 0.60,
        },
    },
    "HARD": {
        "operator_fatigue": 0.7,
        "ambulances_available": 4,
        "ambulances_en_route": 5,
        "queue_length": 35,
        "incident_cascade_risk": 0.75,
        "response_time_pressure": 0.8,
        "event_flags": ["festival_traffic", "mass_casualty", "power_outage"],
        "city_context": "disaster_zone",
        "recent_dispatch_accuracy": 0.3,
        "streak": 1,
        "caller_severity_vector": {
            "cardiac": 0.8,
            "trauma": 0.9,
            "respiratory": 0.7,
            "obstetric": 0.6,
            "neurological": 0.8,
            "pediatric": 0.7,
        },
        "true_severity": {
            "cardiac": 0.90,
            "trauma": 0.95,
            "respiratory": 0.55,
            "obstetric": 0.75,
            "neurological": 0.70,
            "pediatric": 0.85,
        },
    },
}


def _highest_severity_category(severity: dict[str, float]) -> str:
    return max(severity, key=severity.get)  # type: ignore[arg-type]


def _build_call_id(task_name: str) -> str:
    return f"{task_name.lower()}-call-001"


def build_initial_full_state(task_name: str) -> Overload108FullState:
    if task_name not in TASK_INITIAL_STATES:
        raise KeyError(f"unknown task: {task_name}")

    s = TASK_INITIAL_STATES[task_name]
    return Overload108FullState(
        caller_severity_vector={cat: float(s["caller_severity_vector"][cat]) for cat in SEVERITY_CATEGORIES},
        true_severity={cat: float(s["true_severity"][cat]) for cat in SEVERITY_CATEGORIES},
        ambulances_available=int(s["ambulances_available"]),
        ambulances_en_route=int(s["ambulances_en_route"]),
        operator_fatigue=float(s["operator_fatigue"]),
        response_time_pressure=float(s["response_time_pressure"]),
        queue_length=int(s["queue_length"]),
        incident_cascade_risk=float(s["incident_cascade_risk"]),
        current_call_id=_build_call_id(task_name),
        event_flags=list(s["event_flags"]),
        city_context=str(s["city_context"]),
        recent_dispatch_accuracy=float(s["recent_dispatch_accuracy"]),
        streak=int(s["streak"]),
    )


def build_task_spec(task_name: str) -> Overload108TaskSpec:
    if task_name not in TASK_NAMES:
        raise KeyError(f"unknown task: {task_name}")

    return Overload108TaskSpec(
        task_name=task_name,  # type: ignore[arg-type]
        max_steps=TASK_MAX_STEPS[task_name],
        success_threshold=TASK_SUCCESS_THRESHOLDS[task_name],
        initial_state=build_initial_full_state(task_name),
        grader=f"grade_{task_name.lower()}",
        description=(
            f"{task_name} 108-Overload dispatch task with deterministic "
            f"transitions, hidden severity, and passive dynamics."
        ),
    )


TASK_SPECS: Final[dict[str, Overload108TaskSpec]] = {
    task_name: build_task_spec(task_name) for task_name in TASK_NAMES
}


def grade_easy_task(*args: Any, **kwargs: Any):
    from .graders import grade_easy
    return grade_easy(*args, **kwargs)


def grade_medium_task(*args: Any, **kwargs: Any):
    from .graders import grade_medium
    return grade_medium(*args, **kwargs)


def grade_hard_task(*args: Any, **kwargs: Any):
    from .graders import grade_hard
    return grade_hard(*args, **kwargs)


TASKS: list[dict[str, Any]] = [
    {
        "task_id": "EASY",
        "task_name": "EASY",
        "difficulty": "easy",
        "spec": TASK_SPECS["EASY"],
        "grader": grade_easy_task,
        "grader_name": "grade_easy",
    },
    {
        "task_id": "MEDIUM",
        "task_name": "MEDIUM",
        "difficulty": "medium",
        "spec": TASK_SPECS["MEDIUM"],
        "grader": grade_medium_task,
        "grader_name": "grade_medium",
    },
    {
        "task_id": "HARD",
        "task_name": "HARD",
        "difficulty": "hard",
        "spec": TASK_SPECS["HARD"],
        "grader": grade_hard_task,
        "grader_name": "grade_hard",
    },
]

TASKS_WITH_GRADERS: dict[str, dict[str, Any]] = {
    entry["task_name"]: entry for entry in TASKS
}


def list_task_names() -> tuple[str, ...]:
    return TASK_NAMES


def list_task_specs() -> tuple[Overload108TaskSpec, ...]:
    return tuple(TASK_SPECS[t] for t in TASK_NAMES)


def get_task_spec(task_name: str) -> Overload108TaskSpec:
    try:
        return TASK_SPECS[task_name]
    except KeyError as exc:
        raise KeyError(f"unknown task: {task_name}") from exc


def get_default_task_name() -> str:
    return TASK_NAMES[0]


def get_max_steps(task_name: str) -> int:
    return TASK_MAX_STEPS[task_name]


def get_success_threshold(task_name: str) -> float:
    return TASK_SUCCESS_THRESHOLDS[task_name]


__all__ = [
    "TASKS",
    "TASKS_WITH_GRADERS",
    "TASK_INITIAL_STATES",
    "TASK_MAX_STEPS",
    "TASK_SPECS",
    "TASK_SUCCESS_THRESHOLDS",
    "build_initial_full_state",
    "build_task_spec",
    "get_default_task_name",
    "get_max_steps",
    "get_success_threshold",
    "get_task_spec",
    "list_task_names",
    "list_task_specs",
]
