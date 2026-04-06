from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class SatAction:
    satellite_id: str          # which satellite to nudge
    delta_v: List[float]       # [dx, dy, dz] m/s burn vector
    fuel_consumed_current_action: float = 0.0  # |Δv| magnitude (m/s) — proxy for propellant used this step
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SatObservation:
    done: bool
    reward: Optional[float]
    satellites: List[Dict]     # [{id, x, y, z, vx, vy, vz}, ...]
    debris: List[Dict]         # [{id, x, y, z}, ...]
    danger_pairs: List[Dict]   # [{sat_id, debris_id, distance_km}, ...]
    message: str

@dataclass
class SatState:
    episode_id: Optional[str] = None
    step_count: int = 0
    time_elapsed_s: float = 0.0
    collision_avoided: int = 0
    fuel_consumed_total: float = 0.0   # cumulative |Δv| (m/s) spent across the episode