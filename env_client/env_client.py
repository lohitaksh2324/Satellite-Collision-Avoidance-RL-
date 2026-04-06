"""
SatelliteCollisionEnv client — wraps the REST/WS server.
Usage (sync):
    with SatEnvClient(base_url="http://localhost:8000").sync() as env:
        result = env.reset(task="hard")
        while not result.done:
            result = env.step("SAT-0", [0.1, 0.0, -0.05])
"""
import json
import requests
from dataclasses import dataclass
from typing import List, Optional
from contextlib import contextmanager


# ── Response wrappers ─────────────────────────────────────────────────────
@dataclass
class StepResult:
    done: bool
    reward: Optional[float]
    satellites: List[dict]
    debris: List[dict]
    danger_pairs: List[dict]
    step: int
    time_elapsed_s: float
    message: str
    fuel_consumed_total: float = 0.0   # cumulative |Δv| (m/s) spent this episode

    @classmethod
    def from_dict(cls, d: dict) -> "StepResult":
        return cls(
            done          = d.get("done", False),
            reward        = d.get("reward"),
            satellites    = d.get("satellites", []),
            debris        = d.get("debris", []),
            danger_pairs  = d.get("danger_pairs", []),
            step          = d.get("step", 0),
            time_elapsed_s= d.get("time_elapsed_s", 0.0),
            message       = d.get("message", ""),
            fuel_consumed_total = d.get("fuel_consumed_total", 0.0),
        )


# ── Sync client ───────────────────────────────────────────────────────────
class SatEnvClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url   = base_url.rstrip("/")
        self.session_id: Optional[str] = None

    def reset(self, task: str = "medium") -> StepResult:
        r = requests.post(f"{self.base_url}/reset",
                          json={"task": task, "session_id": self.session_id})
        r.raise_for_status()
        data = r.json()
        self.session_id = data["session_id"]
        return StepResult.from_dict(data)

    def step(self, satellite_id: str, delta_v: List[float]) -> StepResult:
        assert self.session_id, "Call reset() first."
        r = requests.post(f"{self.base_url}/step",
                          json={"session_id": self.session_id,
                                "satellite_id": satellite_id,
                                "delta_v": delta_v})
        r.raise_for_status()
        return StepResult.from_dict(r.json())

    def state(self) -> dict:
        assert self.session_id, "Call reset() first."
        r = requests.get(f"{self.base_url}/state",
                         params={"session_id": self.session_id})
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        return requests.get(f"{self.base_url}/health").json()

    # Context manager support
    def sync(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass    # stateless REST — nothing to close