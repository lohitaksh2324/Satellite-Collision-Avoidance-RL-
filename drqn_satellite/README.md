# DRQN Satellite — Collision Avoidance Simulation

A standalone **Deep Recurrent Q-Network** simulation of satellite trajectory
management with real-time 3D visualisation. Completely separate from the
main project.

## Architecture

```
drqn_satellite/
├── collision_predictor.py  — Analytical TCA (Time of Closest Approach)
├── environment.py          — Physics + trajectory + fuel model
├── drqn_agent.py           — DRQN with LSTM + Dueling heads + Double DQN
├── train.py                — Standalone CLI training script
├── server.py               — FastAPI + WebSocket server (real-time streaming)
├── checkpoints/            — Saved agent weights (auto-created)
├── frontend/
│   └── index.html          — 3D Three.js live visualisation
└── requirements.txt
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r drqn_satellite/requirements.txt
```

### 2 — Launch the server (training + live UI)

```bash
python drqn_satellite/server.py
```

Open **http://localhost:8001** in your browser.

### 3 — Use the UI

| Button | Action |
|---|---|
| **Easy / Medium / Hard** | Select difficulty (1 / 3 / 5 threats) |
| **Start Training** | Train DRQN; streams selected episodes live to 3D canvas |
| **Run Inference** | Load checkpoint and stream every step at real-time speed |
| **Stop** | Halt training or inference immediately |

### 4 — Pre-train without the UI (optional)

```bash
python drqn_satellite/train.py --level easy   --episodes 500
python drqn_satellite/train.py --level medium --episodes 600
python drqn_satellite/train.py --level hard   --episodes 800
```

---

## Simulation Design

### Scenario
- The **agent satellite** travels from a random **SOURCE** to a **DESTINATION**.
- A nominal *straight-line* trajectory is pre-computed as a reference.
- **Threats** (debris or other satellites) follow their own constant-velocity paths,
  some deliberately aimed near the nominal trajectory.

### DRQN Architecture
```
obs(t) → Linear Encoder (256) → LSTM (256, 1 layer) → Dueling Head → Q-values (19)
```
The **LSTM hidden state** persists across timesteps, letting the agent remember
its manoeuvre history and plan *deviate → avoid → return* sequences.

### Action Space (19 discrete actions)
| # | Description |
|---|---|
| 0 | **Coast** — no burn |
| 1–3 | +X burn at 0.05 / 0.20 / 0.80 km/s |
| 4–6 | −X burn … |
| 7–9 | +Y burn … |
| 10–12 | −Y burn … |
| 13–15 | +Z burn … |
| 16–18 | −Z burn … |

### Observation Vector (50 features)
- **[0:6]** — satellite position + velocity (normalised)
- **[6:46]** — 5 threat slots × 8 features each: relative pos/vel + TCA time + miss distance
- **[46:50]** — dist-to-dest, fuel-remaining, deviation-from-nominal, steps-remaining

### Difficulty Levels
| Level | Threats | Notes |
|-------|---------|-------|
| Easy  | 1 | 1 collision-course threat |
| Medium| 3 | 2 collision-course + 1 random |
| Hard  | 5 | 3–4 collision-course + 1–2 random |

### Reward Function
| Event | Reward |
|---|---|
| Reach destination | +10 to +15 (fuel efficiency bonus) |
| Collision | −50 |
| Fuel exhausted | −20 |
| Per step — deviation penalty | −0.005 × deviation_km |
| Per step — fuel penalty | −0.05 × Δv_used |
| Per step — proximity | −0.5 × (1 − dist/danger_dist) |
| Per step — progress | +0.02 × (1 − dist_to_dest/total) |

### Collision Prediction
Every step the **TCA solver** analytically computes for each threat:
- **Time of Closest Approach** (seconds)
- **Miss distance** at TCA (km)
- **Predicted positions** of satellite and threat at TCA

These are fed into the observation and displayed in the *Threat Forecasts* panel.

---

## 3D Visualisation Legend

| Object | Appearance |
|---|---|
| Agent satellite | **Glowing blue sphere** with pulsing aura |
| Source | Green cone |
| Destination | Gold octahedron (rotating) |
| Nominal trajectory | White semi-transparent tube |
| Actual trail | Blue line |
| Debris threat | Red icosahedron |
| Satellite threat | Orange sphere |
| TCA collision zone | Transparent red sphere at predicted impact point |
| Burn vector | Yellow arrow (disappears when coasting) |
| Nominal position | Dim blue dot (where satellite *should* be) |
