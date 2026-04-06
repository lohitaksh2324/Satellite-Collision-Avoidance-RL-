"""
FastAPI server — exposes /reset, /step, /state, /health, /ws
"""
import json
import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import SatAction
from server.environment import SatelliteCollisionEnv, TASK_CONFIGS

app = FastAPI(title="Satellite Collision Avoidance Env", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Per-session env store (keyed by session_id) ───────────────────────────
_envs: dict = {}


def _get_env(session_id: str, task: str = "medium") -> SatelliteCollisionEnv:
    if session_id not in _envs:
        _envs[session_id] = SatelliteCollisionEnv(task=task)
    return _envs[session_id]


# ── Pydantic request bodies ───────────────────────────────────────────────
class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    task: str = "medium"

class StepRequest(BaseModel):
    session_id: str
    satellite_id: str
    delta_v: List[float]


# ── REST endpoints ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "tasks": list(TASK_CONFIGS.keys())}


@app.post("/reset")
def reset(req: ResetRequest):
    sid = req.session_id or str(uuid.uuid4())
    env = SatelliteCollisionEnv(task=req.task)
    _envs[sid] = env
    obs = env.reset()
    return {"session_id": sid, **obs.__dict__}


@app.post("/step")
def step(req: StepRequest):
    env = _get_env(req.session_id)
    action = SatAction(
        satellite_id=req.satellite_id,
        delta_v=req.delta_v,
    )
    obs = env.step(action)
    return obs.__dict__


@app.get("/state")
def state(session_id: str = Query(...)):
    env = _get_env(session_id)
    return env.state.__dict__


@app.get("/tasks")
def tasks():
    return TASK_CONFIGS


# ── WebSocket endpoint (for real-time UI) ────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    env: Optional[SatelliteCollisionEnv] = None

    try:
        while True:
            raw  = await websocket.receive_text()
            msg  = json.loads(raw)
            cmd  = msg.get("command")

            if cmd == "reset":
                task = msg.get("task", "medium")
                env  = SatelliteCollisionEnv(task=task)
                _envs[session_id] = env
                obs  = env.reset()
                await websocket.send_json({"event": "reset", "session_id": session_id,
                                           **obs.__dict__})

            elif cmd == "step":
                if env is None:
                    await websocket.send_json({"error": "call reset first"})
                    continue
                action = SatAction(
                    satellite_id=msg.get("satellite_id", "SAT-0"),
                    delta_v=msg.get("delta_v", [0.0, 0.0, 0.0]),
                )
                obs = env.step(action)
                await websocket.send_json({"event": "step", **obs.__dict__})

            elif cmd == "state":
                if env:
                    await websocket.send_json({"event": "state",
                                               **env.state.__dict__})

            else:
                await websocket.send_json({"error": f"unknown command: {cmd}"})

    except WebSocketDisconnect:
        _envs.pop(session_id, None)


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)