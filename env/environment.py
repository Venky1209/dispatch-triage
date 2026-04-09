from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .models import (
    ACTION_TYPES,
    Action,
    Overload108Action,
    Overload108EnvironmentState,
    Overload108FullState,
    Overload108Observation,
    Observation,
    State,
    SEVERITY_CATEGORIES,
    VALID_EVENT_FLAGS,
    VALID_SURGE_REDIRECTS,
    VALID_FATIGUE_STYLES,
    VALID_ESCALATION_TARGETS,
    VALID_HANDOFF_QUALITIES,
    VALID_PRIORITY_LEVELS,
)
from .tasks import get_default_task_name, get_task_spec, list_task_names


EVENT_INJECTION_POOL = ("monsoon_surge", "festival_traffic", "power_outage", "mass_casualty")


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp_reward(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _normalize_task_name(task_name: str | None) -> str:
    candidate = str(task_name or get_default_task_name()).strip().upper()
    return candidate if candidate in list_task_names() else get_default_task_name()


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_lower(value: Any) -> str:
    return str(value).strip().lower()


def _coerce_action(value: Any) -> Overload108Action:
    if isinstance(value, Overload108Action):
        return value
    if isinstance(value, dict):
        return Overload108Action.model_validate(value)
    if hasattr(value, "model_dump"):
        return Overload108Action.model_validate(value.model_dump())
    raise TypeError(f"Unsupported action type: {type(value)!r}")


def _clean_event_flags(flags: list[str]) -> list[str]:
    deduped = []
    for flag in flags:
        if flag in VALID_EVENT_FLAGS and flag not in deduped:
            deduped.append(flag)
    return deduped


@dataclass(slots=True)
class _StepOutcome:
    observation: Overload108Observation
    full_state: Overload108FullState
    reward: float
    done: bool
    info: dict[str, Any]


class Overload108Env:
    def __init__(self, task_name: str = "EASY", seed: int = 42) -> None:
        self._rng = random.Random(seed)
        random.seed(seed)
        self._task_name = _normalize_task_name(task_name)
        self._spec = get_task_spec(self._task_name)
        self._initial_full_state = self._spec.initial_state.model_copy(deep=True)
        self._full_state = self._spec.initial_state.model_copy(deep=True)
        self._state = self._build_state()

    def _build_state(self) -> Overload108EnvironmentState:
        fs = self._full_state.model_copy(deep=True)
        fs.event_flags = _clean_event_flags(fs.event_flags)
        fs.current_call_id = fs.current_call_id or f"{self._task_name.lower()}-call-001"
        observation = fs.to_observation()
        return Overload108EnvironmentState(
            task_name=self._task_name,
            step_count=0,
            max_steps=self._spec.max_steps,
            success_threshold=self._spec.success_threshold,
            observation=observation,
            full_state=fs,
            reward_history=[],
            action_history=[],
            done=False,
            success=False,
        )

    def reset(self, task_name: str | None = None, seed: int | None = None, **_: Any) -> Overload108Observation:
        if seed is not None:
            random.seed(seed)
            self._rng.seed(seed)

        if task_name is not None:
            self._task_name = _normalize_task_name(task_name)

        self._spec = get_task_spec(self._task_name)
        self._initial_full_state = self._spec.initial_state.model_copy(deep=True)
        self._full_state = self._spec.initial_state.model_copy(deep=True)
        self._state = self._build_state()
        return self._state.observation.model_copy(deep=True)

    def state(self) -> Overload108EnvironmentState:
        return self._state.model_copy(deep=True)

    def _apply_passive_dynamics(self, fs: Overload108FullState, step_count: int) -> None:
        """Apply passive dynamics every step regardless of action."""
        fs.operator_fatigue = _clamp_unit_interval(fs.operator_fatigue + 0.05)

        is_surge = any(f in fs.event_flags for f in ("monsoon_surge", "mass_casualty"))
        fs.queue_length = max(0, fs.queue_length + self._rng.randint(0, 2 if is_surge else 1))

        if fs.queue_length > 20:
            fs.incident_cascade_risk = _clamp_unit_interval(fs.incident_cascade_risk + 0.03)
        else:
            fs.incident_cascade_risk = _clamp_unit_interval(fs.incident_cascade_risk - 0.01)

        if step_count % 3 == 0 and fs.ambulances_en_route > 0:
            returned = min(fs.ambulances_en_route, 1)
            fs.ambulances_en_route -= returned
            fs.ambulances_available = min(20, fs.ambulances_available + returned)

        fs.response_time_pressure = _clamp_unit_interval(
            0.3 * (fs.queue_length / 50.0)
            + 0.3 * fs.incident_cascade_risk
            + 0.2 * fs.operator_fatigue
            + 0.2 * (1.0 - fs.ambulances_available / 20.0)
        )

    def _inject_event_if_needed(self, step_count: int, fs: Overload108FullState) -> str | None:
        if step_count % 4 != 0:
            return None

        context_weights = {
            "normal": {"monsoon_surge": 0.1, "festival_traffic": 0.1, "power_outage": 0.2, "mass_casualty": 0.05},
            "monsoon_season": {"monsoon_surge": 0.4, "festival_traffic": 0.05, "power_outage": 0.35, "mass_casualty": 0.15},
            "festival_day": {"monsoon_surge": 0.05, "festival_traffic": 0.45, "power_outage": 0.1, "mass_casualty": 0.25},
            "disaster_zone": {"monsoon_surge": 0.2, "festival_traffic": 0.1, "power_outage": 0.25, "mass_casualty": 0.4},
        }
        weights = context_weights.get(fs.city_context, context_weights["normal"])
        available = [e for e in EVENT_INJECTION_POOL if e not in fs.event_flags]
        if not available:
            return None

        candidates = [(e, weights.get(e, 0.1)) for e in available]
        total = sum(w for _, w in candidates)
        r = self._rng.random() * total
        cumulative = 0.0
        chosen = candidates[0][0]
        for event, weight in candidates:
            cumulative += weight
            if r <= cumulative:
                chosen = event
                break

        fs.event_flags.append(chosen)
        fs.event_flags = _clean_event_flags(fs.event_flags)
        return chosen

    def _apply_action(self, fs: Overload108FullState, action: Overload108Action, step_count: int) -> tuple[Overload108FullState, float, dict[str, Any]]:
        next_fs = fs.model_copy(deep=True)
        reward = 0.0
        info: dict[str, Any] = {"action_type": action.action_type, "reward_components": {}}

        params = dict(action.params)
        action_type = action.action_type

        if action_type == "dispatch_ambulance":
            category = _safe_lower(params.get("severity_category") or params.get("category") or "")
            priority = _safe_lower(params.get("priority_level") or params.get("priority") or "medium")
            backup = bool(params.get("backup_requested", False))

            if next_fs.ambulances_available > 0 and category in SEVERITY_CATEGORIES:
                next_fs.ambulances_available -= 1
                next_fs.ambulances_en_route += 1
                next_fs.queue_length = max(0, next_fs.queue_length - 1)
                next_fs.current_call_id = f"{self._task_name.lower()}-call-{step_count:03d}"

                true_sev = next_fs.true_severity.get(category, 0.5)
                priority_map = {"low": 0.2, "medium": 0.5, "high": 0.75, "critical": 0.95}
                priority_val = priority_map.get(priority, 0.5)

                if abs(priority_val - true_sev) < 0.15:
                    reward += 0.30
                    info["reward_components"]["priority_match"] = 0.30
                elif abs(priority_val - true_sev) < 0.3:
                    reward += 0.10
                    info["reward_components"]["priority_near"] = 0.10

                if backup and next_fs.incident_cascade_risk > 0.6:
                    reward += 0.10
                    info["reward_components"]["backup_during_cascade"] = 0.10

                if priority == "low" and true_sev > 0.7:
                    reward -= 0.25
                    info["reward_components"]["critical_undertriaged"] = -0.25

                next_fs.recent_dispatch_accuracy = _clamp_unit_interval(
                    next_fs.recent_dispatch_accuracy + (0.1 if abs(priority_val - true_sev) < 0.2 else -0.05)
                )
                next_fs.streak += 1
            else:
                reward -= 0.10
                info["reward_components"]["no_ambulance_or_bad_category"] = -0.10

        elif action_type == "triage_call":
            assessed = float(params.get("assessed_severity", 0.5))
            category = _safe_lower(params.get("category") or "")
            escalate = bool(params.get("escalate", False))

            if category in SEVERITY_CATEGORIES:
                true_sev = next_fs.true_severity.get(category, 0.5)
                if abs(assessed - true_sev) < 0.15:
                    reward += 0.20
                    info["reward_components"]["triage_accurate"] = 0.20
                elif abs(assessed - true_sev) < 0.25:
                    reward += 0.10
                    info["reward_components"]["triage_near"] = 0.10

                if escalate and true_sev > 0.7:
                    reward += 0.10
                    info["reward_components"]["escalate_correct"] = 0.10

                if assessed < true_sev - 0.2:
                    reward -= 0.15
                    info["reward_components"]["under_triaged"] = -0.15

                next_fs.recent_dispatch_accuracy = _clamp_unit_interval(
                    next_fs.recent_dispatch_accuracy + (0.08 if abs(assessed - true_sev) < 0.2 else -0.03)
                )

        elif action_type == "handle_surge":
            redirect = _safe_lower(params.get("redirect_to") or params.get("redirect") or "")
            has_surge = any(f in next_fs.event_flags for f in ("monsoon_surge", "festival_traffic", "mass_casualty"))

            if has_surge and redirect in VALID_SURGE_REDIRECTS:
                reward += 0.25
                info["reward_components"]["surge_handled"] = 0.25
                next_fs.incident_cascade_risk = _clamp_unit_interval(next_fs.incident_cascade_risk - 0.1)
                if redirect == "defer_non_critical":
                    next_fs.queue_length = max(0, next_fs.queue_length - 3)
                elif redirect == "request_backup":
                    next_fs.ambulances_available = min(20, next_fs.ambulances_available + 2)
            elif has_surge:
                reward -= 0.10
                info["reward_components"]["bad_surge_redirect"] = -0.10
            else:
                reward -= 0.20
                info["reward_components"]["no_surge_present"] = -0.20

        elif action_type == "manage_fatigue":
            style = _safe_lower(params.get("style") or "")
            if next_fs.operator_fatigue > 0.7:
                reward += 0.15
                info["reward_components"]["fatigue_critical_managed"] = 0.15
                next_fs.operator_fatigue = _clamp_unit_interval(next_fs.operator_fatigue - 0.15)
            elif next_fs.operator_fatigue > 0.5:
                reward += 0.05
                info["reward_components"]["fatigue_proactive"] = 0.05
                next_fs.operator_fatigue = _clamp_unit_interval(next_fs.operator_fatigue - 0.10)
            else:
                reward += 0.02
                next_fs.operator_fatigue = _clamp_unit_interval(next_fs.operator_fatigue - 0.05)

            if style in VALID_FATIGUE_STYLES:
                next_fs.operator_fatigue = _clamp_unit_interval(next_fs.operator_fatigue - 0.03)

        elif action_type == "escalate_incident":
            notify = _safe_lower(params.get("notify") or "")
            has_mass_casualty = "mass_casualty" in next_fs.event_flags

            if has_mass_casualty and notify in VALID_ESCALATION_TARGETS:
                reward += 0.20
                info["reward_components"]["escalation_correct"] = 0.20
                next_fs.event_flags = [f for f in next_fs.event_flags if f != "mass_casualty"]
                next_fs.incident_cascade_risk = _clamp_unit_interval(next_fs.incident_cascade_risk - 0.15)
            elif notify in VALID_ESCALATION_TARGETS:
                reward += 0.05
                info["reward_components"]["escalation_preventive"] = 0.05
            else:
                reward -= 0.10
                info["reward_components"]["bad_escalation"] = -0.10

        elif action_type == "defer_call":
            reason = _safe_str(params.get("reason") or "")
            highest_cat = max(next_fs.true_severity, key=next_fs.true_severity.get)
            avg_true = sum(next_fs.true_severity.values()) / len(next_fs.true_severity)

            if avg_true < 0.3:
                reward += 0.10
                info["reward_components"]["defer_low_severity"] = 0.10
                next_fs.queue_length = max(0, next_fs.queue_length - 1)
            elif next_fs.true_severity.get(highest_cat, 0) > 0.6:
                reward -= 0.25
                info["reward_components"]["defer_critical_penalty"] = -0.25
            else:
                reward -= 0.05

            if len(reason) > 200:
                reason = reason[:200]

        elif action_type == "request_mutual_aid":
            from_district = _safe_str(params.get("from_district") or "")
            if next_fs.ambulances_available < 3:
                reward += 0.20
                info["reward_components"]["mutual_aid_needed"] = 0.20
                next_fs.ambulances_available = min(20, next_fs.ambulances_available + 3)
            elif next_fs.ambulances_available > 10:
                reward -= 0.10
                info["reward_components"]["mutual_aid_wasteful"] = -0.10
            else:
                reward += 0.05
                next_fs.ambulances_available = min(20, next_fs.ambulances_available + 1)

        elif action_type == "close_shift":
            handoff = _safe_lower(params.get("handoff_quality") or "standard")
            if handoff == "thorough" and next_fs.streak > 3 and next_fs.operator_fatigue < 0.6:
                reward += 0.35
                info["reward_components"]["excellent_close"] = 0.35
            elif handoff == "standard":
                reward += 0.10
                info["reward_components"]["standard_close"] = 0.10
            elif handoff == "poor":
                reward -= 0.20
                info["reward_components"]["poor_close"] = -0.20
            else:
                reward += 0.10

            next_fs.operator_fatigue = _clamp_unit_interval(next_fs.operator_fatigue - 0.1)
            next_fs.queue_length = max(0, next_fs.queue_length - 2)

        if next_fs.operator_fatigue > 0.8 and action_type != "manage_fatigue":
            reward -= 0.10
            info["reward_components"]["fatigue_passive_penalty"] = -0.10

        return next_fs, _clamp_reward(reward), info

    def step(self, action: Action | Overload108Action | dict[str, Any]) -> tuple[Observation, float, bool, dict[str, Any]]:
        action_model = _coerce_action(action)
        next_step_count = self._state.step_count + 1

        next_fs, reward, info = self._apply_action(self._full_state, action_model, next_step_count)
        self._apply_passive_dynamics(next_fs, next_step_count)
        injected = self._inject_event_if_needed(next_step_count, next_fs)
        if injected:
            info["injected_event"] = injected

        self._full_state = next_fs
        observation = next_fs.to_observation()

        done = next_step_count >= self._state.max_steps
        score = self._episode_score(next_fs)
        success = done and score >= self._spec.success_threshold

        self._state = Overload108EnvironmentState(
            task_name=self._task_name,
            step_count=next_step_count,
            max_steps=self._spec.max_steps,
            success_threshold=self._spec.success_threshold,
            observation=observation.model_copy(deep=True),
            full_state=next_fs.model_copy(deep=True),
            reward_history=[*self._state.reward_history, reward],
            action_history=[*self._state.action_history, action_model.model_copy(deep=True)],
            done=done,
            success=success,
        )

        info.update({"episode_score": score, "done": done, "success": success})
        return observation.model_copy(deep=True), reward, done, info

    def _episode_score(self, fs: Overload108FullState) -> float:
        return _clamp_unit_interval(
            0.30 * (1.0 - fs.operator_fatigue)
            + 0.25 * (fs.ambulances_available / 20.0)
            + 0.20 * fs.recent_dispatch_accuracy
            + 0.15 * (1.0 - fs.incident_cascade_risk)
            + 0.10 * min(1.0, fs.streak / 10.0)
        )


State = Overload108EnvironmentState


__all__ = ["Overload108Env", "State"]
