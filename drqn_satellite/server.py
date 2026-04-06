"""
FastAPI WebSocket server — streams DRQN inference results in real time.

Start:  python drqn_satellite/server.py
UI:     http://localhost:8001/
WS:     ws://localhost:8001/ws

REST endpoints:
  GET /api/model-info?level=easy|medium|hard  → checkpoint metadata

Client → Server commands:
  {"cmd": "run_inference",  "level": "easy"|"medium"|"hard", "n_episodes": 3}
  {"cmd": "set_speed", "speed": 1.0|2.5|5.0}
  {"cmd": "stop"}

Server → Client events:
  connected | episode_start | step | episode_end | inference_done
  stopped | error | info | speed_changed
"""

import asyncio
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from environment import SatelliteTrajectoryEnv, N_ACTIONS, OBS_DIM, ACTIONS, FUEL_BUDGET
from drqn_agent import DRQNAgent

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="DRQN Satellite Sim", version="2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# Serve static frontend
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


def ckpt_path(level: str) -> str:
    return str(CKPT_DIR / f"drqn_{level}.pth")


# ── REST: Model Info ──────────────────────────────────────────────────────────

@app.get("/api/model-info")
async def model_info(level: str = "easy"):
    """Return checkpoint metadata for the given difficulty level."""
    path = ckpt_path(level)
    if not os.path.exists(path):
        return JSONResponse({
            "level": level,
            "checkpoint_exists": False,
            "episode_count": 0,
            "epsilon": 1.0,
        })
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return JSONResponse({
            "level": level,
            "checkpoint_exists": True,
            "episode_count": ckpt.get("episode_count", 0),
            "epsilon": round(ckpt.get("epsilon", 1.0), 6),
        })
    except Exception as e:
        return JSONResponse({
            "level": level,
            "checkpoint_exists": True,
            "episode_count": 0,
            "epsilon": 1.0,
            "error": str(e),
        })


# ── Global state (one session at a time) ─────────────────────────────────────
_active       = False
_stop_event   = threading.Event()
_msg_queue    = None        # asyncio.Queue — set inside WS handler
_loop         = None        # running event loop — set inside WS handler
_speed_factor = 1.0         # multiplier: 1=normal, 2.5=fast, 5=fastest


def _put(msg: dict):
    """Thread-safe: push JSON message onto the asyncio queue."""
    if _msg_queue and _loop:
        asyncio.run_coroutine_threadsafe(
            _msg_queue.put(json.dumps(msg, default=float)), _loop
        )


# ── Inference thread ──────────────────────────────────────────────────────────

def _inference_worker(level: str, n_episodes: int):
    global _active
    try:
        env   = SatelliteTrajectoryEnv(level=level)
        agent = DRQNAgent(OBS_DIM, N_ACTIONS)
        agent.epsilon = 0.0   # fully greedy

        ckpt = ckpt_path(level)
        if os.path.exists(ckpt):
            agent.load(ckpt)
            agent.epsilon = 0.0
            _put({'event': 'info',
                  'message': f'Loaded checkpoint: {agent.episode_count} episodes trained'})
        else:
            _put({'event': 'info',
                  'message': f'No checkpoint for {level} — using random policy'})

        for ep in range(n_episodes):
            if _stop_event.is_set():
                break

            obs    = env.reset()
            hidden = agent.init_hidden()
            total_r = 0.0
            steps   = 0
            info    = {}
            max_deviation = 0.0
            total_fuel_consumed = 0.0
            deviation_history = []

            state = env.get_render_state()
            # Send full nominal trajectory (all points) at episode start
            full_nom = [p.tolist() for p in env._nom_traj]
            _put({
                'event': 'episode_start',
                'episode': ep,
                'total_episodes': n_episodes,
                'level': level,
                'epsilon': 0.0,
                'episodes_trained': agent.episode_count,
                'full_nominal_trajectory': full_nom,
                **state,
            })

            done = False
            while not done and not _stop_event.is_set():
                action, hidden = agent.select_action(obs, hidden)
                obs, reward, done, info = env.step(action)
                total_r += reward
                steps   += 1

                # Track deviation & fuel
                dev = info.get('deviation', 0.0)
                max_deviation = max(max_deviation, dev)
                fuel_cost = info.get('fuel_cost_step', 0.0)
                total_fuel_consumed += fuel_cost
                deviation_history.append(round(dev, 2))

                state = env.get_render_state()
                _put({
                    'event':        'step',
                    'step':         steps,
                    'episode':      ep,
                    'action':       ACTIONS[action].tolist(),
                    'reward':       round(reward, 4),
                    'total_reward': round(total_r, 4),
                    'info':         {k: v for k, v in info.items()
                                     if isinstance(v, (int, float, bool))},
                    'max_deviation':       round(max_deviation, 3),
                    'total_fuel_consumed': round(total_fuel_consumed, 5),
                    'fuel_budget':         FUEL_BUDGET,
                    **state,
                })
                base_delay = 0.30   # 0.30s per step at 1x speed
                time.sleep(base_delay / max(0.1, _speed_factor))

            # Send full actual trail at episode end
            _put({
                'event':        'episode_end',
                'episode':      ep,
                'total_episodes': n_episodes,
                'total_reward': round(total_r, 4),
                'steps':        steps,
                'reached_dest': bool(info.get('reached_dest', False)),
                'collision':    bool(info.get('collision', False)),
                'timeout':      bool(info.get('timeout', False)),
                'fuel_exhausted': bool(info.get('fuel_exhausted', False)),
                'fuel_remaining': round(info.get('fuel_remaining', 0), 4),
                'max_deviation':  round(max_deviation, 3),
                'total_fuel_consumed': round(total_fuel_consumed, 5),
                'deviation_history': deviation_history,
                'full_actual_trail': env._actual_trail,
                'level': level,
            })
            time.sleep(0.5)  # brief pause between episodes

        _put({'event': 'inference_done', 'n_episodes': n_episodes})
    except Exception as e:
        _put({'event': 'error', 'message': str(e), 'trace': traceback.format_exc()})
    finally:
        _active = False


# ── WebSocket handler ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    global _active, _msg_queue, _loop, _stop_event, _speed_factor

    await websocket.accept()
    _loop      = asyncio.get_running_loop()
    _msg_queue = asyncio.Queue()
    _stop_event.clear()

    forward_task = None

    async def _forward():
        """Read from queue and push to WebSocket."""
        while True:
            msg = await _msg_queue.get()
            try:
                await websocket.send_text(msg)
            except Exception:
                return

    try:
        # Announce capabilities
        await websocket.send_json({
            'event':     'connected',
            'n_actions': N_ACTIONS,
            'obs_dim':   OBS_DIM,
            'levels':    ['easy', 'medium', 'hard'],
        })

        forward_task = asyncio.create_task(_forward())

        while True:
            raw  = await websocket.receive_text()
            data = json.loads(raw)
            cmd  = data.get('cmd', '')

            if cmd == 'run_inference':
                if _active:
                    await websocket.send_json(
                        {'event': 'error', 'message': 'Already running — stop first'})
                    continue
                level    = data.get('level', 'easy')
                n_eps    = int(data.get('n_episodes', 3))
                _active  = True
                _stop_event.clear()
                threading.Thread(
                    target=_inference_worker,
                    args=(level, n_eps),
                    daemon=True,
                ).start()
                await websocket.send_json(
                    {'event': 'inference_started', 'level': level, 'n_episodes': n_eps})

            elif cmd == 'set_speed':
                _speed_factor = float(data.get('speed', 1.0))
                await websocket.send_json(
                    {'event': 'speed_changed', 'speed': _speed_factor})

            elif cmd == 'stop':
                _stop_event.set()
                _active = False
                await websocket.send_json({'event': 'stopped'})

            else:
                await websocket.send_json(
                    {'event': 'error', 'message': f'Unknown command: {cmd}'})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        traceback.print_exc()
    finally:
        _stop_event.set()
        _active = False
        if forward_task:
            forward_task.cancel()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("🛰  DRQN Satellite Server  →  http://localhost:8001")
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False,
                app_dir=str(Path(__file__).parent))
