from __future__ import annotations

import random
import threading
from typing import Any, Final

from fastapi import Body, FastAPI, HTTPException

from env.models import (
    ACTION_TYPES,
    Overload108Action,
    Overload108EnvironmentState,
    Overload108FullState,
    Overload108Observation,
    Overload108ResetResponse,
    Overload108StepResponse,
    Overload108StepResult,
    SEVERITY_CATEGORIES,
    VALID_EVENT_FLAGS,
    VALID_SURGE_REDIRECTS,
    VALID_FATIGUE_STYLES,
    VALID_ESCALATION_TARGETS,
    VALID_HANDOFF_QUALITIES,
    VALID_PRIORITY_LEVELS,
)
from env.tasks import build_initial_full_state, get_default_task_name, get_task_spec, list_task_names, list_task_specs
from env.graders import TASK_GRADERS, grade_task
from env.environment import Overload108Env, _clamp_unit_interval, _clamp_reward, _clean_event_flags


EVENT_INJECTION_POOL: Final[tuple[str, ...]] = ("monsoon_surge", "festival_traffic", "power_outage", "mass_casualty")


def _safe_lower(value: Any) -> str:
    return str(value).strip().lower()


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


class Overload108Runtime:
    def __init__(self, task_name: str | None = None) -> None:
        self._lock = threading.Lock()
        self._env = Overload108Env(task_name=task_name or get_default_task_name(), seed=42)

    def reset(self, task_name: str) -> Overload108EnvironmentState:
        with self._lock:
            self._env.reset(task_name=task_name, seed=42)
            return self._env.state()

    def get_state(self) -> Overload108EnvironmentState:
        with self._lock:
            return self._env.state()

    def step(self, action: Overload108Action) -> tuple[Overload108EnvironmentState, Overload108StepResult]:
        with self._lock:
            state_before = self._env.state()
            if state_before.done:
                raise HTTPException(status_code=409, detail="episode already finished; call /reset to start a new task")

            observation, reward, done, info = self._env.step(action)
            state_after = self._env.state()

            result = Overload108StepResult(
                observation=observation,
                reward=reward,
                done=done,
                success=state_after.success,
                info=info,
            )
            return state_after, result


def create_app() -> FastAPI:
    application = FastAPI(title="108-Overload", version="1.0.0")
    application.state.runtime = Overload108Runtime()

    @application.get("/")
    def root() -> dict[str, Any]:
        runtime: Overload108Runtime = application.state.runtime
        state = runtime.get_state()
        tasks_with_graders = [t for t in list_task_names() if t in TASK_GRADERS]
        return {
            "name": "overload-108",
            "env": "overload_108",
            "tasks": list(list_task_names()),
            "tasks_with_graders": tasks_with_graders,
            "tasks_with_graders_count": len(tasks_with_graders),
            "current_task": state.task_name,
        }

    @application.get("/metadata")
    def metadata() -> dict[str, Any]:
        tasks_with_graders = [t for t in list_task_names() if t in TASK_GRADERS]
        detailed_tasks = []
        difficulty_map = {"EASY": "easy", "MEDIUM": "medium", "HARD": "hard"}
        for spec in list_task_specs():
            detailed_tasks.append({
                "task_id": spec.task_name,
                "task_name": spec.task_name,
                "difficulty": difficulty_map.get(spec.task_name, "unknown"),
                "grader": spec.grader,
                "has_grader": spec.task_name in TASK_GRADERS,
            })
        return {
            "name": "overload-108",
            "description": "108-Overload: National Emergency Ambulance Dispatch Simulator with hidden severity and passive dynamics.",
            "tasks": detailed_tasks,
            "tasks_with_graders": tasks_with_graders,
            "tasks_with_graders_count": len(tasks_with_graders),
        }

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "env": "overload_108"}

    @application.get("/tasks")
    def tasks() -> dict[str, Any]:
        tasks_payload: list[dict[str, Any]] = []
        difficulty_map = {"EASY": "easy", "MEDIUM": "medium", "HARD": "hard"}
        for spec in list_task_specs():
            task_name = spec.task_name
            tasks_payload.append({
                **spec.model_dump(),
                "task_id": task_name,
                "difficulty": difficulty_map.get(task_name, "unknown"),
                "grader": spec.grader,
                "grader_name": f"grade_{task_name.lower()}",
                "has_grader": task_name in TASK_GRADERS,
            })
        tasks_with_graders = [t for t in tasks_payload if t.get("has_grader")]
        return {
            "tasks": tasks_payload,
            "tasks_with_graders": tasks_with_graders,
            "tasks_with_graders_count": len(tasks_with_graders),
            "default_task": get_default_task_name(),
        }

    @application.get("/state", response_model=Overload108ResetResponse)
    def get_state() -> Overload108ResetResponse:
        runtime: Overload108Runtime = application.state.runtime
        return Overload108ResetResponse(state=runtime.get_state())

    @application.post("/state", response_model=Overload108ResetResponse)
    def post_state() -> Overload108ResetResponse:
        runtime: Overload108Runtime = application.state.runtime
        return Overload108ResetResponse(state=runtime.get_state())

    @application.post("/reset", response_model=Overload108ResetResponse)
    def reset(payload: dict[str, Any] = Body(default={})) -> Overload108ResetResponse:
        runtime: Overload108Runtime = application.state.runtime
        request_body = payload
        task_name = request_body.get("task_name") or request_body.get("task_id") or request_body.get("task") or get_default_task_name()
        state = runtime.reset(str(task_name).strip().upper())
        return Overload108ResetResponse(state=state)

    @application.post("/step", response_model=Overload108StepResponse)
    def step(payload: dict[str, Any]) -> Overload108StepResponse:
        runtime: Overload108Runtime = application.state.runtime
        action_payload = payload.get("action", payload)
        action = Overload108Action.model_validate(action_payload)
        state, result = runtime.step(action)
        return Overload108StepResponse(state=state, result=result)

    @application.post("/grader")
    def grader(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request_body = payload or {}
        task_name = str(request_body.get("task_name") or request_body.get("task_id") or request_body.get("task") or get_default_task_name()).strip().upper()
        if task_name not in TASK_GRADERS:
            raise HTTPException(status_code=404, detail=f"unknown task: {task_name}")

        initial_state = request_body.get("initial_state")
        final_state = request_body.get("final_state")
        actions = request_body.get("actions")
        trajectory = request_body.get("trajectory")

        if initial_state is None:
            initial_state = build_initial_full_state(task_name).to_observation().model_dump()
        if final_state is None:
            final_state = build_initial_full_state(task_name).to_observation().model_dump()
        if actions is None:
            actions = []
        if trajectory is None:
            trajectory = []

        result = grade_task(
            task_name=task_name,
            initial_state=initial_state,
            final_state=final_state,
            actions=actions,
            trajectory=trajectory,
        )
        return {
            "task_name": result.task_name,
            "task_id": result.task_name,
            "grader": f"grade_{result.task_name.lower()}",
            "score": result.score,
            "breakdown": result.breakdown,
            "has_grader": True,
        }

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
