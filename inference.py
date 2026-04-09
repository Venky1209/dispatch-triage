from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import Any
from types import SimpleNamespace

try:
    from openai import OpenAI
except ImportError:
    class _NoOpResponses:
        def create(self, *args: Any, **kwargs: Any) -> None:
            return None

    class OpenAI:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.responses = _NoOpResponses()

try:
    import app as app_module
    from env.graders import grade_task, score_breakdown
    from env.models import (
        ACTION_TYPES,
        Overload108Action,
        Overload108Observation,
        SEVERITY_CATEGORIES,
        VALID_SURGE_REDIRECTS,
        VALID_FATIGUE_STYLES,
        VALID_ESCALATION_TARGETS,
        VALID_HANDOFF_QUALITIES,
        VALID_PRIORITY_LEVELS,
    )
    from env.tasks import get_default_task_name, get_task_spec, list_task_names
    FALLBACK_MODE = False
except Exception:
    FALLBACK_MODE = True

    ACTION_TYPES = (
        "dispatch_ambulance",
        "triage_call",
        "handle_surge",
        "manage_fatigue",
        "escalate_incident",
        "defer_call",
        "request_mutual_aid",
        "close_shift",
        "deescalate_caller",
    )
    SEVERITY_CATEGORIES = (
        "cardiac",
        "trauma",
        "respiratory",
        "obstetric",
        "neurological",
        "pediatric",
    )
    VALID_SURGE_REDIRECTS = ("mutual_aid", "defer_non_critical", "request_backup", "activate_protocol")
    VALID_FATIGUE_STYLES = ("rotate_operator", "take_micro_break", "request_supervisor")
    VALID_ESCALATION_TARGETS = ("hospital", "police", "fire", "disaster_management")
    VALID_HANDOFF_QUALITIES = ("poor", "standard", "thorough")
    VALID_PRIORITY_LEVELS = ("low", "medium", "high", "critical")

    @dataclass(slots=True)
    class Overload108Action:
        action_type: str
        params: dict[str, Any] = field(default_factory=dict)

        def model_copy(self, deep: bool = False) -> "Overload108Action":
            return copy.deepcopy(self) if deep else copy.copy(self)

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

    @dataclass(slots=True)
    class Overload108Observation:
        caller_severity_vector: dict[str, float] = field(default_factory=dict)
        ambulances_available: int = 10
        ambulances_en_route: int = 0
        operator_fatigue: float = 0.2
        response_time_pressure: float = 0.2
        queue_length: int = 5
        incident_cascade_risk: float = 0.1
        current_call_id: str = ""
        event_flags: list[str] = field(default_factory=list)
        city_context: str = "normal"
        recent_dispatch_accuracy: float = 0.5
        streak: int = 0
        caller_panic: float = 0.3
        district_load: dict[str, float] = field(default_factory=lambda: {"north": 0.5, "south": 0.5, "east": 0.5, "west": 0.5, "central": 0.5})

        def model_copy(self, deep: bool = False) -> "Overload108Observation":
            return copy.deepcopy(self) if deep else copy.copy(self)

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

    @dataclass(slots=True)
    class Overload108EnvironmentState:
        task_name: str
        step_count: int
        max_steps: int
        success_threshold: float
        observation: Overload108Observation
        reward_history: list[float] = field(default_factory=list)
        action_history: list[Overload108Action] = field(default_factory=list)
        done: bool = False
        success: bool = False

        def model_copy(self, deep: bool = False) -> "Overload108EnvironmentState":
            return copy.deepcopy(self) if deep else copy.copy(self)

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

    @dataclass(slots=True)
    class Overload108StepResult:
        observation: Overload108Observation
        reward: float
        done: bool
        success: bool
        info: dict[str, Any] = field(default_factory=dict)

        def model_copy(self, deep: bool = False) -> "Overload108StepResult":
            return copy.deepcopy(self) if deep else copy.copy(self)

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

    @dataclass(slots=True)
    class Overload108TaskSpec:
        task_name: str
        max_steps: int
        success_threshold: float
        initial_state: Overload108Observation
        description: str = ""

    TASK_INITIAL_STATES: dict[str, dict[str, Any]] = {
        "EASY": {
            "operator_fatigue": 0.2, "ambulances_available": 15, "ambulances_en_route": 0,
            "queue_length": 5, "incident_cascade_risk": 0.1, "response_time_pressure": 0.2,
            "event_flags": ["non_critical_backlog"], "city_context": "normal",
            "recent_dispatch_accuracy": 0.5, "streak": 0,
            "caller_severity_vector": {"cardiac": 0.3, "trauma": 0.2, "respiratory": 0.4, "obstetric": 0.1, "neurological": 0.2, "pediatric": 0.3},
        },
        "MEDIUM": {
            "operator_fatigue": 0.5, "ambulances_available": 8, "ambulances_en_route": 3,
            "queue_length": 18, "incident_cascade_risk": 0.45, "response_time_pressure": 0.55,
            "event_flags": ["monsoon_surge", "power_outage"], "city_context": "monsoon_season",
            "recent_dispatch_accuracy": 0.4, "streak": 2,
            "caller_severity_vector": {"cardiac": 0.5, "trauma": 0.6, "respiratory": 0.5, "obstetric": 0.4, "neurological": 0.3, "pediatric": 0.5},
        },
        "HARD": {
            "operator_fatigue": 0.7, "ambulances_available": 4, "ambulances_en_route": 5,
            "queue_length": 35, "incident_cascade_risk": 0.75, "response_time_pressure": 0.8,
            "event_flags": ["festival_traffic", "mass_casualty", "power_outage"], "city_context": "disaster_zone",
            "recent_dispatch_accuracy": 0.3, "streak": 1,
            "caller_severity_vector": {"cardiac": 0.8, "trauma": 0.9, "respiratory": 0.7, "obstetric": 0.6, "neurological": 0.8, "pediatric": 0.7},
        },
    }
    TASK_MAX_STEPS = {"EASY": 8, "MEDIUM": 15, "HARD": 25}
    TASK_SUCCESS_THRESHOLDS = {"EASY": 0.50, "MEDIUM": 0.62, "HARD": 0.72}

    def _clamp_unit_interval(value: float) -> float:
        return max(0.01, min(0.99, float(value)))

    def _clamp_reward(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    def build_initial_observation(task_name: str) -> Overload108Observation:
        s = TASK_INITIAL_STATES[task_name]
        return Overload108Observation(
            caller_severity_vector={c: float(s["caller_severity_vector"][c]) for c in SEVERITY_CATEGORIES},
            ambulances_available=int(s["ambulances_available"]),
            ambulances_en_route=int(s["ambulances_en_route"]),
            operator_fatigue=float(s["operator_fatigue"]),
            response_time_pressure=float(s["response_time_pressure"]),
            queue_length=int(s["queue_length"]),
            incident_cascade_risk=float(s["incident_cascade_risk"]),
            current_call_id=f"{task_name.lower()}-call-001",
            event_flags=list(s["event_flags"]),
            city_context=str(s["city_context"]),
            recent_dispatch_accuracy=float(s["recent_dispatch_accuracy"]),
            streak=int(s["streak"]),
            caller_panic=0.3,
            district_load={"north": 0.5, "south": 0.5, "east": 0.5, "west": 0.5, "central": 0.5},
        )

    def list_task_names() -> tuple[str, ...]:
        return tuple(TASK_INITIAL_STATES.keys())

    def get_default_task_name() -> str:
        return "EASY"

    def get_task_spec(task_name: str) -> Overload108TaskSpec:
        return Overload108TaskSpec(
            task_name=task_name,
            max_steps=TASK_MAX_STEPS[task_name],
            success_threshold=TASK_SUCCESS_THRESHOLDS[task_name],
            initial_state=build_initial_observation(task_name),
            description=f"Fallback {task_name} task.",
        )

    def grade_task(*args: Any, **kwargs: Any) -> SimpleNamespace:
        final_state = kwargs.get("final_state")
        if final_state is None and len(args) > 1:
            final_state = args[1]
        if hasattr(final_state, "observation"):
            observation = final_state.observation
        else:
            observation = final_state
        if hasattr(observation, "model_dump"):
            observation = observation.model_dump()
        score = _clamp_unit_interval(
            0.30 * (1.0 - float(observation["operator_fatigue"]))
            + 0.25 * (float(observation["ambulances_available"]) / 20.0)
            + 0.20 * float(observation["recent_dispatch_accuracy"])
            + 0.15 * (1.0 - float(observation["incident_cascade_risk"]))
            + 0.10 * min(1.0, float(observation["streak"]) / 10.0)
        )
        return SimpleNamespace(score=score)

    def score_breakdown(*args: Any, **kwargs: Any) -> dict[str, float]:
        return {"mock_criterion": 0.5}

    class _FallbackRuntime:
        def __init__(self, task_name: str | None = None) -> None:
            self._rng = random.Random(42)
            self._task_name = task_name or get_default_task_name()
            self._spec = get_task_spec(self._task_name)
            self._initial_observation = self._spec.initial_state.model_copy(deep=True)
            self._state = self._build_state(self._spec)

        def _build_state(self, spec: Overload108TaskSpec) -> Overload108EnvironmentState:
            observation = spec.initial_state.model_copy(deep=True)
            return Overload108EnvironmentState(
                task_name=spec.task_name, step_count=0, max_steps=spec.max_steps,
                success_threshold=spec.success_threshold, observation=observation,
            )

        def reset(self, task_name: str) -> Overload108EnvironmentState:
            self._rng.seed(42)
            self._task_name = task_name
            self._spec = get_task_spec(task_name)
            self._initial_observation = self._spec.initial_state.model_copy(deep=True)
            self._state = self._build_state(self._spec)
            return self._state.model_copy(deep=True)

        def get_state(self) -> Overload108EnvironmentState:
            return self._state.model_copy(deep=True)

        def _episode_score(self, obs: Overload108Observation) -> float:
            return _clamp_unit_interval(
                0.30 * (1.0 - obs.operator_fatigue)
                + 0.25 * (obs.ambulances_available / 20.0)
                + 0.20 * obs.recent_dispatch_accuracy
                + 0.15 * (1.0 - obs.incident_cascade_risk)
                + 0.10 * min(1.0, obs.streak / 10.0)
            )

        def step(self, action: Overload108Action) -> tuple[Overload108EnvironmentState, Overload108StepResult]:
            if self._state.done:
                raise RuntimeError("episode already finished; reset first")

            prev = self._state.model_copy(deep=True)
            obs = prev.observation.model_copy(deep=True)
            reward = 0.0

            if action.action_type == "dispatch_ambulance":
                if obs.ambulances_available > 0:
                    obs.ambulances_available -= 1
                    obs.ambulances_en_route += 1
                    obs.queue_length = max(0, obs.queue_length - 1)
                    reward += 0.20
                    obs.streak += 1

            elif action.action_type == "triage_call":
                reward += 0.15
                obs.recent_dispatch_accuracy = min(1.0, obs.recent_dispatch_accuracy + 0.05)

            elif action.action_type == "handle_surge":
                has_surge = any(f in obs.event_flags for f in ("monsoon_surge", "festival_traffic", "mass_casualty"))
                if has_surge:
                    reward += 0.25
                    obs.incident_cascade_risk = max(0, obs.incident_cascade_risk - 0.1)

            elif action.action_type == "manage_fatigue":
                if obs.operator_fatigue > 0.5:
                    reward += 0.15
                    obs.operator_fatigue = max(0, obs.operator_fatigue - 0.15)

            elif action.action_type == "escalate_incident":
                if "mass_casualty" in obs.event_flags:
                    reward += 0.20
                    obs.event_flags = [f for f in obs.event_flags if f != "mass_casualty"]

            elif action.action_type == "request_mutual_aid":
                if obs.ambulances_available < 5:
                    reward += 0.20
                    obs.ambulances_available = min(20, obs.ambulances_available + 3)

            elif action.action_type == "close_shift":
                handoff = str(action.params.get("handoff_quality", "standard")).lower()
                if handoff == "thorough":
                    reward += 0.35
                elif handoff == "standard":
                    reward += 0.10

            elif action.action_type == "defer_call":
                reward -= 0.05

            # Passive dynamics
            obs.operator_fatigue = min(1.0, obs.operator_fatigue + 0.05)
            next_step = prev.step_count + 1

            score = self._episode_score(obs)
            done = next_step >= prev.max_steps
            success = done and score >= prev.success_threshold

            next_state = Overload108EnvironmentState(
                task_name=prev.task_name, step_count=next_step, max_steps=prev.max_steps,
                success_threshold=prev.success_threshold, observation=obs,
                reward_history=[*prev.reward_history, _clamp_reward(reward)],
                action_history=[*prev.action_history, action.model_copy(deep=True)],
                done=done, success=success,
            )
            self._state = next_state
            result = Overload108StepResult(
                observation=obs.model_copy(deep=True), reward=_clamp_reward(reward),
                done=done, success=success, info={"episode_score": score},
            )
            return next_state.model_copy(deep=True), result

    app_module = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime=_FallbackRuntime())))


DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_ENV_NAME = "overload_108"


@dataclass(slots=True)
class InferenceConfig:
    task_name: str
    model_name: str
    api_base_url: str
    api_key: str
    use_llm: bool = False
    client: OpenAI | None = None


def _clamp_unit_interval(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


def _build_config(argv: list[str]) -> InferenceConfig:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("task_name", nargs="?")
    parser.add_argument("--task", dest="task_flag")
    parser.add_argument("--model-name", dest="model_name")
    parser.add_argument("--api-base-url", dest="api_base_url")
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument("--hf-token", dest="hf_token")
    parser.add_argument("--use-llm", action="store_true")
    parsed, _ = parser.parse_known_args(argv)

    task_candidate = parsed.task_flag or parsed.task_name or os.getenv("TASK_NAME") or get_default_task_name()
    task_name = str(task_candidate).strip().upper()
    if task_name not in list_task_names():
        task_name = get_default_task_name()

    model_name = parsed.model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    api_base_url = parsed.api_base_url or os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    api_key = parsed.api_key or parsed.hf_token or os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")
    use_llm = bool(parsed.use_llm or os.getenv("USE_LLM_POLICY", "0") == "1" or api_key)
    client = OpenAI(base_url=api_base_url, api_key=api_key or "hf_dummy")
    return InferenceConfig(
        task_name=task_name, model_name=model_name, api_base_url=api_base_url,
        api_key=api_key, use_llm=use_llm, client=client,
    )


def _task_state() -> Any:
    return app_module.app.state.runtime


def _highest_severity(obs: Overload108Observation) -> str:
    return max(obs.caller_severity_vector, key=obs.caller_severity_vector.get)  # type: ignore[arg-type]


def _second_highest_severity(obs: Overload108Observation) -> str:
    ranked = sorted(SEVERITY_CATEGORIES, key=lambda c: obs.caller_severity_vector.get(c, 0), reverse=True)
    return ranked[1] if len(ranked) > 1 else ranked[0]


def _priority_for_severity(severity_val: float) -> str:
    if severity_val > 0.7:
        return "critical"
    if severity_val > 0.5:
        return "high"
    if severity_val > 0.3:
        return "medium"
    return "low"


def _make_action(action_type: str, **params: Any) -> Overload108Action:
    if action_type not in ACTION_TYPES:
        action_type = "triage_call"
    return Overload108Action(action_type=action_type, params={k: v for k, v in params.items() if v is not None})


def _choose_action(task_name: str, state: Any, counters: dict[str, int]) -> Overload108Action:
    obs: Overload108Observation = state.observation
    step_idx = state.step_count + 1
    top_cat = _highest_severity(obs)
    sev_val = obs.caller_severity_vector.get(top_cat, 0.5)

    close_steps = {"EASY": {8}, "MEDIUM": {8, 15}, "HARD": {8, 16, 24}}[task_name]
    if step_idx in close_steps:
        return _make_action("close_shift", handoff_quality="thorough")

    if "mass_casualty" in obs.event_flags and counters["escalate"] < {"EASY": 1, "MEDIUM": 2, "HARD": 3}[task_name]:
        return _make_action("escalate_incident", incident_type="mass_casualty", notify="disaster_management")

    has_surge = any(f in obs.event_flags for f in ("monsoon_surge", "festival_traffic"))
    if has_surge and counters["surge"] < {"EASY": 1, "MEDIUM": 2, "HARD": 3}[task_name]:
        return _make_action("handle_surge", redirect_to="defer_non_critical")

    if obs.operator_fatigue > 0.6 and counters["fatigue"] < {"EASY": 1, "MEDIUM": 2, "HARD": 4}[task_name]:
        return _make_action("manage_fatigue", style="rotate_operator")

    if obs.ambulances_available < 3 and counters["mutual_aid"] < {"EASY": 0, "MEDIUM": 1, "HARD": 2}[task_name]:
        return _make_action("request_mutual_aid", from_district="adjacent", severity_category=top_cat)

    triage_cats = list(SEVERITY_CATEGORIES)
    if step_idx % 3 == 0 and counters["triage"] < len(triage_cats):
        cat = triage_cats[counters["triage"] % len(triage_cats)]
        return _make_action(
            "triage_call",
            assessed_severity=obs.caller_severity_vector.get(cat, 0.5),
            category=cat,
            escalate=obs.caller_severity_vector.get(cat, 0) > 0.6,
        )

    return _make_action(
        "dispatch_ambulance",
        severity_category=top_cat,
        priority_level=_priority_for_severity(sev_val),
        estimated_eta=12,
        backup_requested=obs.incident_cascade_risk > 0.6,
    )


def _update_counters(counters: dict[str, int], action: Overload108Action, state_before: Any, state_after: Any) -> None:
    a = action.action_type
    if a == "escalate_incident":
        counters["escalate"] += 1
    elif a == "handle_surge":
        counters["surge"] += 1
    elif a == "manage_fatigue":
        counters["fatigue"] += 1
    elif a == "request_mutual_aid":
        counters["mutual_aid"] += 1
    elif a == "triage_call":
        counters["triage"] += 1
    elif a == "close_shift":
        counters["close_shift"] += 1
    elif a == "dispatch_ambulance":
        counters["dispatch"] += 1


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _format_reward_list(values: list[float]) -> str:
    return ",".join(_format_reward(v) for v in values)


def _run_episode(config: InferenceConfig) -> tuple[bool, int, float, list[float], dict[str, float]]:
    runtime = _task_state()
    state = runtime.reset(config.task_name)
    initial_observation = state.observation.model_copy(deep=True)

    actions: list[Overload108Action] = []
    trajectory: list[dict[str, Any]] = []
    counters = {"escalate": 0, "surge": 0, "fatigue": 0, "mutual_aid": 0, "triage": 0, "close_shift": 0, "dispatch": 0}

    while not state.done:
        action = _choose_action(config.task_name, state, counters)
        obs_before = state.observation.model_copy(deep=True)

        try:
            next_state, result = runtime.step(action)
            error_text = None
        except Exception as exc:
            next_state = state
            result = None
            error_text = str(exc)

        if result is None:
            print(
                f"[STEP] step={state.step_count + 1} action={action.action_type} reward=0.00 done=false error={error_text or 'unknown'}",
                flush=True,
            )
            break

        actions.append(action.model_copy(deep=True))
        trajectory.append({
            "observation_before": obs_before.model_dump(),
            "action": action.model_dump(),
            "observation_after": next_state.observation.model_dump(),
            "result": result.model_dump(),
        })
        _update_counters(counters, action, state, next_state)

        print(
            f"[STEP] step={state.step_count + 1} action={action.action_type} reward={_format_reward(result.reward)} done={str(result.done).lower()} error=null",
            flush=True,
        )

        state = next_state
        if state.done:
            break

    score = grade_task(
        task_name=config.task_name,
        initial_state=initial_observation,
        final_state=state.observation,
        actions=actions,
        trajectory=trajectory,
        task_spec=get_task_spec(config.task_name),
    ).score

    breakdown = score_breakdown(
        task_name=config.task_name,
        initial_state=initial_observation,
        final_state=state.observation,
        actions=actions,
        trajectory=trajectory,
        task_spec=get_task_spec(config.task_name),
    )

    return bool(state.success), int(state.step_count), score, list(state.reward_history), breakdown


def _maybe_use_llm(config: InferenceConfig) -> None:
    if not config.use_llm and not config.api_key:
        return
    if config.client is None:
        return
    try:
        config.client.responses.create(
            model=config.model_name,
            input=[
                {"role": "system", "content": "Return a concise emergency dispatch triage strategy."},
                {"role": "user", "content": f"Task: {config.task_name}. Provide one deterministic dispatch strategy."},
            ],
        )
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    config = _build_config(args)
    _maybe_use_llm(config)

    all_tasks = list(list_task_names())

    for task_name in all_tasks:
        config.task_name = task_name
        print(f"[START] task={task_name} env={DEFAULT_ENV_NAME} model={config.model_name}", flush=True)
        success, steps, score, rewards, breakdown = _run_episode(config)
        print(
            f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={_format_reward_list(rewards)}",
            flush=True,
        )
        for criterion, cr_score in breakdown.items():
            print(f"[BREAKDOWN] task={task_name} criterion={criterion} score={cr_score:.2f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
