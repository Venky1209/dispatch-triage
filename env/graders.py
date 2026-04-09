from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .models import (
    ACTION_TYPES,
    Overload108Action,
    Overload108EnvironmentState,
    Overload108FullState,
    Overload108Observation,
    Overload108TaskSpec,
    SEVERITY_CATEGORIES,
    VALID_ESCALATION_TARGETS,
    VALID_SURGE_REDIRECTS,
)
from .tasks import get_task_spec


@dataclass(slots=True)
class GradingResult:
    task_name: str
    score: float
    breakdown: dict[str, float]


def _clamp_score(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _coerce_observation(value: Any) -> Overload108Observation:
    if isinstance(value, Overload108Observation):
        return value
    data = _as_mapping(value)
    if "observation" in data and not isinstance(data.get("caller_severity_vector"), dict):
        return _coerce_observation(data["observation"])
    return Overload108Observation.model_validate(data)


def _coerce_full_state(value: Any) -> Overload108FullState:
    if isinstance(value, Overload108FullState):
        return value
    data = _as_mapping(value)
    if "full_state" in data:
        return _coerce_full_state(data["full_state"])
    return Overload108FullState.model_validate(data)


def _coerce_action(value: Any) -> Overload108Action:
    if isinstance(value, Overload108Action):
        return value
    data = _as_mapping(value)
    if "action" in data and not isinstance(data.get("action_type"), str):
        return _coerce_action(data["action"])
    return Overload108Action.model_validate(data)


def _extract_actions(*sources: Any) -> list[Overload108Action]:
    actions: list[Overload108Action] = []
    for source in sources:
        if source is None:
            continue
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            for item in source:
                try:
                    actions.append(_coerce_action(item))
                except Exception:
                    continue
        else:
            try:
                actions.append(_coerce_action(source))
            except Exception:
                continue
    return actions


def _extract_task_spec(task_name: str | None, task_spec: Any | None = None) -> Overload108TaskSpec:
    if task_spec is not None:
        if isinstance(task_spec, Overload108TaskSpec):
            return task_spec
        return Overload108TaskSpec.model_validate(_as_mapping(task_spec))
    if task_name is None:
        raise ValueError("task_name is required for grading")
    return get_task_spec(task_name)


def _weighted(binary_value: bool, weight: float) -> float:
    return weight if binary_value else 0.0


def _weighted_ratio(count: int, threshold: int, weight: float) -> float:
    if threshold <= 0:
        return weight
    return weight * min(1.0, max(0.0, count / threshold))


def _score_easy(
    initial_state: Overload108Observation,
    final_state: Overload108Observation,
    actions: Sequence[Overload108Action],
    trajectory: Sequence[Any],
) -> GradingResult:
    dispatch_critical = any(
        a.action_type == "dispatch_ambulance"
        and str(a.params.get("priority_level") or a.params.get("priority") or "").lower() == "critical"
        for a in actions
    )
    surge_or_fatigue = any(a.action_type in {"handle_surge", "manage_fatigue"} for a in actions)
    queue_reduced = final_state.queue_length < initial_state.queue_length
    fatigue_under = final_state.operator_fatigue < 0.7

    components = {
        "critical_call_dispatched": _weighted(dispatch_critical, 0.25),
        "surge_or_fatigue_handled": _weighted(surge_or_fatigue, 0.25),
        "queue_reduced": _weighted(queue_reduced, 0.25),
        "final_fatigue_under": _weighted(fatigue_under, 0.25),
    }
    score = _clamp_score(sum(components.values()))
    return GradingResult(task_name="EASY", score=score, breakdown=components)


def _score_medium(
    initial_state: Overload108Observation,
    final_state: Overload108Observation,
    actions: Sequence[Overload108Action],
    trajectory: Sequence[Any],
) -> GradingResult:
    cascade_contained = final_state.incident_cascade_risk < 0.5
    mutual_or_escalation = any(a.action_type in {"request_mutual_aid", "escalate_incident"} for a in actions)

    triage_used = any(a.action_type == "triage_call" for a in actions)
    fatigue_managed = any(
        a.action_type == "manage_fatigue"
        for a in actions
    )
    ambulances_remain = final_state.ambulances_available > 0

    components = {
        "cascade_contained": _weighted(cascade_contained, 0.20),
        "mutual_aid_or_escalation": _weighted(mutual_or_escalation, 0.20),
        "triage_used": _weighted(triage_used, 0.20),
        "fatigue_managed": _weighted(fatigue_managed, 0.20),
        "ambulances_not_exhausted": _weighted(ambulances_remain, 0.20),
    }
    score = _clamp_score(sum(components.values()))
    return GradingResult(task_name="MEDIUM", score=score, breakdown=components)


def _score_hard(
    initial_state: Overload108Observation,
    final_state: Overload108Observation,
    actions: Sequence[Overload108Action],
    trajectory: Sequence[Any],
) -> GradingResult:
    close_shift_thorough = sum(
        1 for a in actions
        if a.action_type == "close_shift"
        and str(a.params.get("handoff_quality", "")).lower() == "thorough"
    )
    cascade_handled = sum(
        1 for a in actions
        if a.action_type in {"handle_surge", "escalate_incident"}
    )
    mutual_aid_count = sum(1 for a in actions if a.action_type == "request_mutual_aid")

    triage_categories = set()
    for a in actions:
        if a.action_type == "triage_call":
            cat = str(a.params.get("category", "")).strip().lower()
            if cat in SEVERITY_CATEGORIES:
                triage_categories.add(cat)

    streak_improved = final_state.streak > initial_state.streak
    strong_close = (
        final_state.operator_fatigue < 0.5
        and final_state.incident_cascade_risk < 0.4
        and final_state.queue_length < 20
    )

    components = {
        "close_shift_thorough_x2": _weighted_ratio(close_shift_thorough, 2, 0.15),
        "cascade_events_handled_x3": _weighted_ratio(cascade_handled, 3, 0.15),
        "mutual_aid_requested_x2": _weighted_ratio(mutual_aid_count, 2, 0.15),
        "four_categories_triaged": _weighted_ratio(len(triage_categories), 4, 0.15),
        "streak_maintained": _weighted(streak_improved, 0.20),
        "strong_close": _weighted(strong_close, 0.20),
    }

    # Anti-spam: penalize if any single action type dominates > 40% of trajectory
    if actions:
        action_counts = Counter(a.action_type for a in actions)
        most_common_ratio = action_counts.most_common(1)[0][1] / len(actions)
        if most_common_ratio > 0.4:
            score_penalty = (most_common_ratio - 0.4) * 0.5
            score = _clamp_score(sum(components.values()) - score_penalty)
        else:
            score = _clamp_score(sum(components.values()))
    else:
        score = _clamp_score(sum(components.values()))

    return GradingResult(task_name="HARD", score=score, breakdown=components)


def _normalize_grading_inputs(*args: Any, **kwargs: Any) -> tuple[str, Overload108Observation, Overload108Observation, list[Overload108Action], list[Any], Overload108TaskSpec]:
    task_name = kwargs.pop("task_name", None) or kwargs.pop("task", None)
    if task_name is not None:
        task_name = str(task_name).strip().upper()

    task_spec = kwargs.pop("task_spec", None)
    initial_state = kwargs.pop("initial_state", None)
    final_state = kwargs.pop("final_state", None)
    state = kwargs.pop("state", None)
    environment_state = kwargs.pop("environment_state", None)
    actions = kwargs.pop("actions", None)
    trajectory = kwargs.pop("trajectory", None)
    if trajectory is None:
        trajectory = kwargs.pop("history", None) or kwargs.pop("transitions", None) or kwargs.pop("step_results", None)

    if args:
        if task_name is None:
            task_name = str(args[0]).strip().upper()
        if len(args) > 1 and final_state is None and state is None:
            final_state = args[1]
        if len(args) > 2 and initial_state is None:
            initial_state = args[2]
        if len(args) > 3 and actions is None:
            actions = args[3]
        if len(args) > 4 and trajectory is None:
            trajectory = args[4]

    if final_state is None:
        final_state = state if state is not None else environment_state

    spec = _extract_task_spec(task_name, task_spec)
    initial_full = spec.initial_state

    final_obs = _coerce_observation(final_state if final_state is not None else initial_full.to_observation())
    initial_obs = _coerce_observation(initial_state if initial_state is not None else initial_full.to_observation())

    action_list = _extract_actions(actions)
    trajectory_list = list(trajectory) if trajectory else []

    if not action_list and trajectory_list:
        for record in trajectory_list:
            data = _as_mapping(record)
            for key in ("action", "step_action", "move"):
                if key in data:
                    try:
                        action_list.append(_coerce_action(data[key]))
                    except Exception:
                        continue
                    break

    return spec.task_name, initial_obs, final_obs, action_list, trajectory_list, spec


def grade_easy(
    initial_state: Overload108Observation,
    final_state: Overload108Observation,
    actions: Sequence[Overload108Action],
    trajectory: Sequence[Any] | None = None,
) -> GradingResult:
    return _score_easy(initial_state, final_state, actions, trajectory or [])


def grade_medium(
    initial_state: Overload108Observation,
    final_state: Overload108Observation,
    actions: Sequence[Overload108Action],
    trajectory: Sequence[Any] | None = None,
) -> GradingResult:
    return _score_medium(initial_state, final_state, actions, trajectory or [])


def grade_hard(
    initial_state: Overload108Observation,
    final_state: Overload108Observation,
    actions: Sequence[Overload108Action],
    trajectory: Sequence[Any] | None = None,
) -> GradingResult:
    return _score_hard(initial_state, final_state, actions, trajectory or [])


def grade_task(*args: Any, **kwargs: Any) -> GradingResult:
    task_name, initial_state, final_state, actions, trajectory, _spec = _normalize_grading_inputs(*args, **kwargs)
    if task_name == "EASY":
        return grade_easy(initial_state, final_state, actions, trajectory)
    if task_name == "MEDIUM":
        return grade_medium(initial_state, final_state, actions, trajectory)
    if task_name == "HARD":
        return grade_hard(initial_state, final_state, actions, trajectory)
    raise ValueError(f"unsupported task: {task_name}")


def grade(*args: Any, **kwargs: Any) -> GradingResult:
    return grade_task(*args, **kwargs)


def grade_episode(*args: Any, **kwargs: Any) -> GradingResult:
    return grade_task(*args, **kwargs)


def score_breakdown(*args: Any, **kwargs: Any) -> dict[str, float]:
    result = grade_task(*args, **kwargs)
    return result.breakdown


def build_grader(task_name: str) -> Any:
    graders = {"EASY": grade_easy, "MEDIUM": grade_medium, "HARD": grade_hard}
    return graders.get(task_name.upper(), grade_task)


def get_task_grader(task_name: str) -> Any:
    return build_grader(task_name)


TASK_GRADERS: dict[str, Any] = {
    "EASY": grade_easy,
    "MEDIUM": grade_medium,
    "HARD": grade_hard,
}


__all__ = [
    "GradingResult",
    "TASK_GRADERS",
    "build_grader",
    "get_task_grader",
    "grade",
    "grade_easy",
    "grade_episode",
    "grade_hard",
    "grade_medium",
    "grade_task",
    "score_breakdown",
]
