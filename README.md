# SATGUARD — Satellite Collision Avoidance Environment

> A real-world OpenEnv environment where an AI agent learns to protect low-Earth orbit satellites from debris collisions using thruster burns.

---

## Overview

**SATGUARD** is a 3D orbital mechanics simulation built as a fully-compliant OpenEnv environment. An agent controls a fleet of satellites in low-Earth orbit (LEO), deciding each timestep whether and how to fire thrusters (delta-v burns) to avoid incoming debris.

This is not a toy or game — satellite collision avoidance is one of the most critical challenges in modern spaceflight. Real events like the 2009 Iridium–Cosmos collision and the Kessler syndrome threat make this a meaningful real-world RL problem.

---

## ⚙️ Setup

### Requirements

```bash
pip install fastapi uvicorn requests
```

Python 3.9+ required.

### Start the Environment Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Open the Visualization

Open `frontend/index.html` in your browser (no build step needed).

### Run the Baseline Agent

```bash
# All tasks (10 episodes each)
python scripts/baseline.py

# Single task with seed for reproducibility
python scripts/baseline.py --task easy --episodes 10 --seed 42

# Verbose step output
python scripts/baseline.py --task medium --verbose
```

---

## 🔁 OpenEnv API

The environment follows the standard `step()` / `reset()` / `state()` pattern, exposed both as a REST API and a Python class.

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode. Returns initial observation. |
| `POST` | `/step`  | Apply an action. Returns next observation + reward. |
| `GET`  | `/state` | Get current episode state (step count, fuel, etc.). |
| `GET`  | `/health`| Health check + available task list. |
| `GET`  | `/tasks` | List all task configs. |
| `WS`   | `/ws`    | WebSocket for real-time UI streaming. |

### Python Client

```python
from env_client.env_client import SatEnvClient

with SatEnvClient("http://localhost:8000") as env:
    obs = env.reset(task="medium")
    while not obs.done:
        obs = env.step("SAT-0", [0.1, 0.0, -0.05])
        print(f"Step reward: {obs.reward}, Danger pairs: {len(obs.danger_pairs)}")
```

### Direct Python (no server)

```python
from server.environment import SatelliteCollisionEnv
from models import SatAction

env = SatelliteCollisionEnv(task="medium")
obs = env.reset()
while not obs.done:
    action = SatAction(satellite_id="SAT-0", delta_v=[0.1, 0.0, -0.05])
    obs    = env.step(action)
```

---

## 📐 Action Space

**Type:** Structured object

| Field | Type | Description |
|-------|------|-------------|
| `satellite_id` | `str` | ID of the satellite to maneuver (e.g. `"SAT-0"`) |
| `delta_v` | `[float, float, float]` | Velocity change vector `[Δvx, Δvy, Δvz]` in m/s |
| `fuel_consumed_current_action` | `float` | Magnitude `\|Δv\|` in m/s (computed by env, read-only) |

**Tips:**
- `[0.0, 0.0, 0.0]` = no burn, coast through timestep
- Burn magnitude is `sqrt(Δvx² + Δvy² + Δvz²)` — proportional to fuel spent
- Only one satellite can burn per step

---

## 🔭 Observation Space

Returned by every `reset()` and `step()` call.

| Field | Type | Description |
|-------|------|-------------|
| `done` | `bool` | `True` when episode ends (collision or max steps) |
| `reward` | `float \| None` | Step reward signal (see Reward section) |
| `satellites` | `List[dict]` | All satellites: `{id, x, y, z, vx, vy, vz}` positions in km, velocities in m/s |
| `debris` | `List[dict]` | All debris objects: `{id, x, y, z, vx, vy, vz}` |
| `danger_pairs` | `List[dict]` | Active threats: `{sat_id, debris_id, distance_km}` |
| `message` | `str` | Human-readable status message |

### State (separate endpoint)

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | `str` | UUID for this episode |
| `step_count` | `int` | Number of steps taken |
| `time_elapsed_s` | `float` | Simulated time in seconds (10s per step) |
| `collision_avoided` | `int` | Count of danger-pair steps survived |
| `fuel_consumed_total` | `float` | Cumulative `\|Δv\|` burned this episode (m/s) |

---

## 🏆 Reward Function

The reward is designed to encourage **both safety and fuel efficiency**:

| Condition | Reward |
|-----------|--------|
| No danger pairs (fully safe step) | `+0.1` |
| Danger pairs present, no collision | `+0.0 → +0.5` (linear, proportional to closest approach / threshold) |
| Collision (`distance < 0.5 km`) | `−1.0` (episode ends) |

Formula for partial credit:
```
reward = (min_distance_km / danger_threshold_km) × 0.5
```

This gives **partial progress signals** so gradient-based RL methods can learn without requiring dense demonstrations.

---

## 🎯 Tasks

Three difficulty configurations, each graded 0.0–1.0:

| Task | Satellites | Debris | Danger Threshold | Max Steps |
|------|-----------|--------|-----------------|-----------|
| `easy` | 2 | 10 | 3.0 km | 100 |
| `medium` | 3 | 20 | 5.0 km | 200 |
| `hard` | 5 | 40 | 8.0 km | 500 |

### Graders

```python
from tasks.graders import grade_easy, grade_medium, grade_hard, grade_all

def my_policy(obs):
    return SatAction(satellite_id=obs.satellites[0]["id"], delta_v=[0, 0, 0])

score = grade_easy(my_policy, n_episodes=10)    # returns float 0.0–1.0
all_scores = grade_all(my_policy, n_episodes=10)  # {"easy": ..., "medium": ..., "hard": ..., "total": ...}
```

**Scoring formula per episode:**
- `0.0` if any collision occurred
- Otherwise: `0.4 × reward_score + 0.4 × distance_score + 0.2 × step_score`

---

## 📁 File Structure

```
artificially_intelligent/
│
├── models.py                   # Typed dataclasses: SatAction, SatObservation, SatState
├── openenv.yaml                # OpenEnv spec (action/obs spaces, tasks, reward schema)
├── README.md                   # This file
│
├── server/
│   ├── environment.py          # ★ Core RL environment: reset(), step(), reward, physics
│   └── app.py                  # FastAPI server exposing REST + WebSocket API
│
├── env_client/
│   └── env_client.py           # Python client wrapping the REST API (Gym-style)
│
├── tasks/
│   └── graders.py              # Task graders: grade_easy/medium/hard/all → 0.0–1.0
│
├── scripts/
│   └── baseline.py             # ★ Baseline heuristic agent + reproducible score runner
│
└── frontend/
    └── index.html              # Real-time 3D visualization (plain HTML/JS, no build)
```

---

## 🤖 RL Architecture

This environment is **OpenAI Gymnasium-compatible** in structure:

```
Agent ──action──▶ SatelliteCollisionEnv.step()
  ▲                        │
  └──── observation ◀──────┘
  └──── reward ◀───────────┘
```

The current baseline uses a **hand-coded heuristic** (push away from nearest debris). To train a real RL policy:

```python
# Example with Stable-Baselines3
from stable_baselines3 import PPO
# Wrap SatelliteCollisionEnv in a gymnasium.Env wrapper, then:
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

---

## 📊 Baseline Scores

Heuristic agent (greedy away-from-debris burn), `--seed 42`, 10 episodes per task:

| Task | Score |
|------|-------|
| Easy | ~0.72 |
| Medium | ~0.61 |
| Hard | ~0.44 |
| **Total** | **~0.59** |

To reproduce:
```bash
python scripts/baseline.py --seed 42 --episodes 10
```

---

## 🔬 Physics Model

- **Propagation:** Simple Euler integration, 10-second timestep
- **Coordinate system:** 3D Cartesian (km), centered on Earth
- **Object velocities:** km per timestep (roughly LEO orbital speeds, simplified)
- **Fuel model:** `|Δv|` = Euclidean magnitude of burn vector, accumulated as `fuel_consumed_total`
- **Collision criterion:** Any satellite-debris pair closer than `0.5 km`
