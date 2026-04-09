---
title: Dispatch Triage
emoji: 🚑
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
license: mit
tags:
  - openenv
short_description: "108-Overload: Emergency Dispatch Simulator"
---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,14,20&height=200&section=header&text=108-Overload&fontSize=50&animation=fadeIn&fontAlignY=35&desc=National%20Emergency%20Ambulance%20Dispatch%20Simulator&descAlignY=55&descAlign=50" alt="108-Overload Header" />
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/OpenEnv-v1.0-brightgreen?style=for-the-badge" alt="OpenEnv Spec" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Hugging_Face-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white" alt="HF Space" />
  <img src="https://img.shields.io/badge/License-MIT-gray?style=for-the-badge" alt="License" />
</div>

<br>

Most RL environments test if an agent can solve a problem. **108-Overload tests if an agent can save lives under impossible constraints.**

India's 108 emergency ambulance service handles **94,000+ calls per day**, responds to **39,000 emergencies daily**, and rescues **800+ lives every single day**. During monsoon surges, festival stampedes, and mass-casualty events, operators face cascading failures with limited ambulances, exhausted staff, and patients deteriorating by the second. This environment simulates that exact high-stakes dispatch problem as a deterministic OpenEnv for the Meta × Hugging Face × Scaler Hackathon.

---

## ❖ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FastAPI Server                     │
│              /reset  /step  /state  /grader           │
├─────────────────────────────────────────────────────┤
│                  Overload108Runtime                   │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐ │
│  │  EASY Task   │  │ MEDIUM Task│  │  HARD Task   │ │
│  │ Normal Shift │  │Monsoon Surg│  │Mass Casualty │ │
│  │  8 steps     │  │ 15 steps   │  │  25 steps    │ │
│  └──────────────┘  └────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────┤
│              Overload108Env (Core Engine)             │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Passive Dynamics Engine (runs EVERY step)      │ │
│  │  • operator_fatigue += 0.05                     │ │
│  │  • queue_length grows during surge              │ │
│  │  • cascade_risk compounds if queue > 20         │ │
│  │  • ambulances return from en_route every 3 steps│ │
│  └─────────────────────────────────────────────────┘ │
│  ┌───────────────┐  ┌─────────────────────────────┐ │
│  │ Hidden State  │  │ Event Injection (every 4th) │ │
│  │ true_severity │  │ weighted by city_context     │ │
│  │ ≠ caller_sev  │  │ monsoon|festival|power|mass │ │
│  └───────────────┘  └─────────────────────────────┘ │
├─────────────────────────────────────────────────────┤
│                  Graders (per-task)                   │
│  Binary criteria + ratio-weighted + anti-spam penalty │
│  Scores clamped [0.01, 0.99]                         │
└─────────────────────────────────────────────────────┘
```

## ✦ Why This Matters

Current OpenEnv tasks model static, transactional goals — parse an email, solve an equation, fill out a form. They fail to evaluate an agent's ability to **triage under resource scarcity**, **manage cascading failures**, and **make life-or-death prioritization decisions in real time**.

**108-Overload fills a critical gap in agentic evaluation.** By simulating a dispatch operator whose fleet depletes, whose fatigue compounds passively, and whose patient conditions deteriorate every tick the agent hesitates, this environment forces frontier LLMs to exhibit **multi-objective optimization under uncertainty** — not just greedy action selection.

## ⟡ How It's Different

*   **Hidden State (Information Asymmetry):** The agent sees `caller_severity_vector` — what the panicked caller *reports*. The environment tracks `true_severity` — what the patient *actually* has. These differ by up to ±0.2. The grader evaluates against ground truth, so an agent that blindly trusts caller reports will under-triage critical patients and get penalized.
*   **Passive Dynamics (Time Kills):** Every single `step()` — regardless of what action the agent takes — increases `operator_fatigue` by +0.05, grows the call queue, and compounds `incident_cascade_risk` if the queue exceeds 20. Doing nothing is actively punished. This is not a "choose the right answer" environment — it's a "choose the least-bad option under time pressure" environment.
*   **Anti-Exploit Design:** Graders evaluate behavioral criteria (did the agent triage 4+ categories? did it request mutual aid when fleet was depleted?) rather than just final-state outcomes. An anti-spam penalty reduces scores if any single action type exceeds 40% of the trajectory, preventing degenerate "spam dispatch" strategies.

---

## ⌕ Observation Space

The agent receives a structured numeric snapshot of the dispatch center's live state. **Critically, `true_severity` is hidden — the agent must infer patient acuity from the noisy `caller_severity_vector`.**

| Field | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| `caller_severity_vector` | `Dict[str, float]` | `0.0 - 1.0` | Reported severity across 6 categories (cardiac, trauma, respiratory, obstetric, neurological, pediatric). **Noisy — differs from true severity.** |
| `ambulances_available` | `int` | `0 - 20` | Fleet currently at the station. Depletes with dispatches, replenishes as en-route units return. |
| `ambulances_en_route` | `int` | `0 - 20` | Units currently responding. Return to available every 3 steps. |
| `operator_fatigue` | `float` | `0.0 - 1.0` | **Passively increases +0.05 every step.** High fatigue triggers passive reward penalties. |
| `response_time_pressure` | `float` | `0.0 - 1.0` | Composite pressure metric derived from queue length, cascade risk, fatigue, and fleet depletion. |
| `queue_length` | `int` | `0 - 50` | Pending calls. **Grows passively during surge events.** |
| `incident_cascade_risk` | `float` | `0.0 - 1.0` | Risk of systemic failure. Compounds if queue > 20, decays slowly otherwise. |
| `event_flags` | `List[str]` | `Enum` | Active crisis modifiers: `monsoon_surge`, `festival_traffic`, `power_outage`, `mass_casualty`, `non_critical_backlog`. |
| `city_context` | `str` | `Enum` | Environmental stress state: `normal`, `monsoon_season`, `festival_day`, `disaster_zone`. |
| `recent_dispatch_accuracy` | `float` | `0.0 - 1.0` | Rolling accuracy of recent triage decisions against hidden `true_severity`. |
| `streak` | `int` | `0 - ∞` | Consecutive successful dispatches. |
| `caller_panic` | `float` | `0.0 - 1.0` | Urgency state reported by the caller. Increases when queue > 15, decreases when en-route > 3. |
| `district_load` | `Dict[str, float]` | `0.0 - 1.0` | Geographical load distributions across 5 districts. Drifts randomly ±0.05 per step. |

## ⚙ Action Space

The agent selects from 8 discrete dispatch operations, each with structured parameters.

| Action Type | Parameters | Description |
| :--- | :--- | :--- |
| `dispatch_ambulance` | `severity_category`, `priority_level`, `estimated_eta`, `backup_requested`, `district` | Deploy an ambulance. `district` targeting yields +0.10 if highest load. Priority must match `true_severity` for maximum reward. |
| `triage_call` | `assessed_severity`, `category`, `escalate` | Assess a caller. Reward scales with proximity to hidden `true_severity`. |
| `handle_surge` | `redirect_to` | Redirect resources during active surge events. Options: `mutual_aid`, `defer_non_critical`, `request_backup`, `activate_protocol`. |
| `manage_fatigue` | `style` | Address operator fatigue. Styles: `rotate_operator`, `take_micro_break`, `request_supervisor`. |
| `escalate_incident` | `incident_type`, `notify` | Escalate to external agencies: `hospital`, `police`, `fire`, `disaster_management`. |
| `defer_call` | `reason`, `callback_eta` | Defer a call. Penalized heavily if deferred call has high `true_severity` or if `caller_panic` > 0.6. |
| `request_mutual_aid` | `from_district`, `severity_category` | Request ambulances from adjacent districts. Reward (+0.20 base) scales up with `district_load` max values when fleet < 3, penalized when fleet > 10. |
| `deescalate_caller` | None | Actively calms the caller, reducing `caller_panic` by 0.2 and yielding +0.10 reward. |
| `close_shift` | `handoff_quality` | End a shift block. `thorough` handoff with streak > 3 and fatigue < 0.6 yields maximum reward and appends shift continuity notes. |

---

## ⌖ Reward Design Deep-Dive

108-Overload implements **dense, non-sparse reward shaping** mapped to real-world dispatch effectiveness:

- **Priority Matching:** Dispatching an ambulance whose `priority_level` maps within 0.15 of the hidden `true_severity` yields +0.30. Priority mismatch (sending low-priority to a critical patient) yields -0.25.
- **Triage Accuracy:** Assessing a call within 0.15 of `true_severity` yields +0.20. Under-triaging (assessed < true by > 0.2) penalizes -0.15.
- **Cascade Prevention:** Successfully handling a surge event reduces `incident_cascade_risk` and yields +0.25. Ignoring an active surge penalizes -0.20.
- **Fatigue Management:** Managing fatigue when > 0.7 yields +0.15, but every step where fatigue > 0.8 without management applies a passive -0.10 penalty.
- **Urgency & District Balancing:** Dispatching correctly to the highest-load district yields +0.10, and dispatching fast when `caller_panic > 0.7` adds an urgent response bonus points (+0.10).
- **Shift Continuity:** A `thorough` `close_shift` generates shift notes, successfully closing the gap and reducing `response_time_pressure` by -0.05 on the succeeding shift `reset`.

---

## ▤ Task Difficulty Breakdown

| Task | Max Steps | Simulation Context | Success Threshold | Core Challenge |
| :--- | :--- | :--- | :--- | :--- |
| **`EASY`** | 8 | Normal shift. 15 ambulances, 5 pending calls. | `> 0.50` | Basic dispatch prioritization, minor fatigue management. |
| **`MEDIUM`** | 15 | Monsoon surge + power outage. 8 ambulances, 18 pending calls. | `> 0.62` | Resource scarcity under active surge, cascade prevention, fleet conservation. |
| **`HARD`** | 25 | Mass casualty + festival traffic + power outage. 4 ambulances, 35 pending calls. Disaster zone. | `> 0.72` | Triage under impossible constraints: 4 ambulances for 35 calls, cascade containment, mutual aid coordination, fatigue triage across 25 steps of passive decay. |

---

## ▶ Quick Start

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Start the FastAPI Server:**
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

**3. Run the baseline inference script:**
```bash
python inference.py
```

## ◘ Docker

```bash
docker build -t dispatch-triage .
docker run --rm -p 7860:7860 dispatch-triage
```

## ✓ OpenEnv Validation

```bash
./validate.sh https://your-space-name.hf.space
```
