"""
Analytical collision prediction using Time of Closest Approach (TCA).
Given satellite and threat position/velocity, computes WHEN and HOW CLOSE
they will be — so the DRQN agent can avoid before the collision happens.
"""
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class CollisionForecast:
    threat_id: str
    threat_type: str          # 'debris' or 'satellite'
    time_to_collision: float  # seconds until closest approach
    miss_distance: float      # km at closest approach
    predicted_sat_pos: list   # [x,y,z] of satellite at TCA
    predicted_threat_pos: list# [x,y,z] of threat at TCA
    is_critical: bool         # miss_distance < DANGER_DIST


class CollisionPredictor:
    def __init__(self, collision_dist_km: float = 0.5, danger_dist_km: float = 5.0):
        self.collision_dist = collision_dist_km
        self.danger_dist = danger_dist_km

    def predict(self, sat_pos: np.ndarray, sat_vel: np.ndarray,
                threats: list) -> List[CollisionForecast]:
        """
        Compute CollisionForecast for every threat.
        Returns list sorted by time_to_collision (most imminent first).
        """
        forecasts = []
        for t in threats:
            t_pos = np.asarray(t['pos'], dtype=float)
            t_vel = np.asarray(t['vel'], dtype=float)
            tca_time, miss_dist, sat_tca, thr_tca = self._tca(
                np.asarray(sat_pos, dtype=float),
                np.asarray(sat_vel, dtype=float),
                t_pos, t_vel,
            )
            forecasts.append(CollisionForecast(
                threat_id=t['id'],
                threat_type=t.get('type', 'debris'),
                time_to_collision=float(tca_time),
                miss_distance=float(miss_dist),
                predicted_sat_pos=sat_tca.tolist(),
                predicted_threat_pos=thr_tca.tolist(),
                is_critical=miss_dist < self.danger_dist,
            ))
        return sorted(forecasts, key=lambda f: f.time_to_collision)

    # ── internal ────────────────────────────────────────────────────────────
    def _tca(self, pos_s, vel_s, pos_t, vel_t):
        """Compute Time of Closest Approach analytically."""
        dr = pos_s - pos_t
        dv = vel_s - vel_t
        dv_sq = float(np.dot(dv, dv))
        if dv_sq < 1e-12:
            t_star = 0.0
        else:
            t_star = float(max(0.0, -np.dot(dr, dv) / dv_sq))
        sat_tca  = pos_s + vel_s * t_star
        thr_tca  = pos_t + vel_t * t_star
        miss     = float(np.linalg.norm(sat_tca - thr_tca))
        return t_star, miss, sat_tca, thr_tca
