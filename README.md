# 🛰️ Orbital Guardian — Satellite Collision Avoidance System

A reinforcement learning–based satellite collision avoidance system that simulates a multi-satellite orbital environment with dynamic debris fields and inter-satellite conjunction threats. The system trains a **PPO (Proximal Policy Optimization)** agent in Python and deploys it in a real-time **3D browser visualizer** powered by Three.js.

---

## 📁 Project Structure

```
RL-main/
├── index.html              # Entry point for the browser simulation
├── main.js                 # Game loop — orchestrates sim, agent, and UI
├── agent.js                # Heuristic + Neural (PPO) collision avoidance agents
├── simulation.js           # JavaScript orbital physics engine (browser-side)
├── renderer.js             # Three.js 3D scene manager
├── ui.js                   # UI panels, logs, stats, and sound engine
├── index.css               # Stylesheet for the simulation UI
├── trained_weights.json    # Pre-trained PPO neural network weights
└── training/
    ├── train.py            # PPO training script (Stable-Baselines3)
    ├── sat_env.py          # Gymnasium environment for satellite avoidance
    ├── sim_engine.py       # NumPy-vectorized orbital physics (training-side)
    ├── export_weights.py   # Exports trained model weights to JSON
    └── requirements.txt    # Python dependencies
```

---

## 🚀 Features

- **8-satellite constellation** across multiple orbital planes with realistic inclinations and altitudes
- **30 dynamic debris objects** spawned at randomized orbital parameters
- **Vectorized threat computation** — calculates time-to-closest-approach (TCA) and miss distance for every satellite–debris and satellite–satellite pair each step
- **PPO agent** trained via Stable-Baselines3 with a 256-dim observation space and 24-dim continuous action space (delta-v per satellite)
- **Layered reward function** penalising collisions, proximity violations, and fuel expenditure
- **Heuristic fallback agent** for real-time avoidance when neural weights are unavailable
- **3D browser visualizer** with danger zone rendering, manoeuvre popups, mission logs, and time-warp control

---

## 🧠 How It Works

### Observation Space (256-dim)
Each of the 8 satellites contributes 32 features:
- Fuel ratio, normalized altitude, velocity components (vx, vy, vz)
- 3 nearest threats × 9 features each: relative position, relative velocity, distance, miss distance, TCA

### Action Space (24-dim)
- 3 delta-v components per satellite, scaled by `MAX_DV = 0.04`
- Actions are clipped to `[-1, 1]` and scaled continuously

### Reward Structure
| Signal | Value |
|---|---|
| Survival bonus (per step) | +0.1 |
| Collision penalty | −500.0 |
| Fuel cost | −5.0 × total Δv |
| Critical proximity (miss < 0.35) | −3.0 per pair |
| Warning proximity (miss < 0.7) | −0.5 per pair |

### Threat Zones
| Zone | Radius |
|---|---|
| Collision | 0.08 units |
| Critical | 0.35 units |
| Warning | 0.70 units |
| Caution | 1.20 units |

---

## 🛠️ Setup & Usage

### Running the Browser Visualizer

Simply open `index.html` in a modern browser (Chrome/Firefox recommended). No build step required — the simulation runs entirely client-side.

> The visualizer will automatically load `trained_weights.json` and run the Neural PPO agent. If weights fail to load, it falls back to the heuristic agent.

---

### Training the PPO Agent (Python)

#### 1. Install dependencies

```bash
cd training/
pip install -r requirements.txt
```

#### 2. Train the model

```bash
# Default: 500,000 timesteps
python train.py

# Custom run
python train.py --timesteps 2000000 --lr 1e-4
```

Checkpoints are saved every 50,000 steps to `./checkpoints/`. TensorBoard logs are also generated.

#### 3. Export weights to JSON

```bash
python export_weights.py
```

This produces `trained_weights.json`, which can be dropped into the root directory for use in the browser visualizer.

---

## 📦 Dependencies

### Python (training)
See `training/requirements.txt`. Key packages:
- `stable-baselines3`
- `gymnasium`
- `numpy`
- `torch`

### Browser (visualizer)
- **Three.js** — 3D rendering
- No build tools or npm required

---

## 📊 Training Tips

- Start with `--timesteps 500000` to verify the environment is working
- Use `--timesteps 2000000` or more for a well-converged policy
- Monitor collision count and fuel mean in the console output printed every 10,000 steps
- Lower `--lr` (e.g. `1e-4`) for more stable but slower convergence

---

## 📄 License

This project was developed as part of the **AAIES (Autonomous AI Embedded Systems)** coursework.
