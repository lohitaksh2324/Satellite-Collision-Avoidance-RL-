"""
Satellite Trajectory Environment for DRQN training.

Simulates ONE agent satellite travelling from SOURCE → DESTINATION.
Threats (debris / other satellites) follow their own trajectories.
The DRQN must deviate minimally from the nominal path to avoid collisions
and still reach the destination with fuel to spare.

Difficulty:
  easy   → 1 threat
  medium → 3 threats
  hard   → 5 threats
"""

import numpy as np
import random
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from collision_predictor import CollisionPredictor

# ── Constants ────────────────────────────────────────────────────────────────
DT              = 10.0    # seconds per step
MAX_STEPS       = 200     # steps per episode
FUEL_BUDGET     = 5.0     # km/s total Δv budget (small burns only)
FUEL_THRESHOLD  = 0.05    # episode fails if fuel drops below this
COLLISION_DIST  = 8.0     # km – physics collision declared (matches visual 3D scale)
DANGER_DIST     = 25.0    # km – "danger zone" for reward shaping
MAX_THREATS     = 5       # padded observation size (always 5 slots)
SCALE           = 300.0   # normalisation; positions ÷ SCALE

DIFFICULTY = {"easy": 1, "medium": 3, "hard": 5}

# ── Discrete action space ────────────────────────────────────────────────────
# 1 coast + 6 directions × 3 magnitudes = 19 actions
_DV_MAGNITUDES = [0.02, 0.05, 0.1]  # km/s — larger maneuvers to dodge 15km hitboxes
_DV_DIRECTIONS = [
    np.array([ 1.,  0.,  0.]),
    np.array([-1.,  0.,  0.]),
    np.array([ 0.,  1.,  0.]),
    np.array([ 0., -1.,  0.]),
    np.array([ 0.,  0.,  1.]),
    np.array([ 0.,  0., -1.]),
]
ACTIONS = [np.zeros(3)]    # action 0 = coast
for _d in _DV_DIRECTIONS:
    for _m in _DV_MAGNITUDES:
        ACTIONS.append(_d * _m)
N_ACTIONS = len(ACTIONS)   # 19

# obs = sat(6) + per_threat(8) × MAX_THREATS + mission(4)
OBS_DIM = 6 + MAX_THREATS * 8 + 4   # = 50


class SatelliteTrajectoryEnv:
    def __init__(self, level: str = "easy"):
        if level not in DIFFICULTY:
            raise ValueError(f"level must be one of {list(DIFFICULTY)}")
        self.level     = level
        self.n_threats = DIFFICULTY[level]
        self._pred     = CollisionPredictor(COLLISION_DIST, DANGER_DIST)

        # filled by reset()
        self._sat       = {}
        self._threats   = []
        self._src       = np.zeros(3)
        self._dst       = np.zeros(3)
        self._nom_traj  = []      # [MAX_STEPS+1] nominal positions
        self._step      = 0
        self._fuel      = FUEL_BUDGET
        self._done      = False
        self._total_rew = 0.0
        self._col_avoided = 0
        self._actual_trail = []   # [(x,y,z)] for visualisation

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        # Random source / destination 200-500 km apart
        self._src = np.random.uniform(-150, 150, 3)
        while True:
            self._dst = np.random.uniform(-150, 150, 3)
            dist = np.linalg.norm(self._dst - self._src)
            if 200 < dist < 450:
                break

        direction = (self._dst - self._src) / dist
        # speed to reach destination comfortably in ~150 steps (leaving buffer)
        speed = dist / (150 * DT)
        nom_vel = direction * speed

        self._sat = {'pos': self._src.copy(), 'vel': nom_vel.copy()}

        # Pre-compute nominal straight-line trajectory
        self._nom_traj = [self._src + nom_vel * DT * i for i in range(MAX_STEPS + 1)]

        # Spawn threats — some on collision course, some nearby
        self._threats = []
        n_coll = max(1, int(self.n_threats * 0.7))  # majority on collision course
        for i in range(self.n_threats):
            if i < n_coll:
                t = self._make_collision_threat(i)
            else:
                t = self._make_random_threat(i)
            self._threats.append(t)

        self._step       = 0
        self._fuel       = FUEL_BUDGET
        self._done       = False
        self._total_rew  = 0.0
        self._col_avoided = 0
        self._actual_trail = [self._src.tolist()]

        return self._obs()

    def step(self, action_idx: int):
        if self._done:
            raise RuntimeError("Episode done — call reset()")

        # Apply Δv burn
        dv        = ACTIONS[action_idx].copy()
        fuel_cost = float(np.linalg.norm(dv))
        self._fuel -= fuel_cost
        self._sat['vel'] = self._sat['vel'] + dv

        # Propagate
        self._sat['pos'] = self._sat['pos'] + self._sat['vel'] * DT
        for t in self._threats:
            t['pos'] = t['pos'] + t['vel'] * DT

        self._step += 1
        self._actual_trail.append(self._sat['pos'].tolist())

        # Metrics
        deviation = self._geometric_deviation(self._sat['pos'])
        dist_dest = float(np.linalg.norm(self._sat['pos'] - self._dst))

        # Collision check
        collision = False
        min_threat_dist = float('inf')
        for t in self._threats:
            d = float(np.linalg.norm(self._sat['pos'] - t['pos']))
            min_threat_dist = min(min_threat_dist, d)
            if d < COLLISION_DIST:
                collision = True

        fuel_exhausted = self._fuel < FUEL_THRESHOLD
        reached_dest   = dist_dest < 10.0
        timeout        = self._step >= MAX_STEPS

        # Reward
        reward = self._reward(
            collision, fuel_exhausted, reached_dest,
            deviation, min_threat_dist, fuel_cost, dist_dest
        )
        self._done = collision or fuel_exhausted or reached_dest or timeout
        self._total_rew += reward

        if not collision and min_threat_dist < DANGER_DIST:
            self._col_avoided += 1

        info = {
            'collision':         collision,
            'fuel_exhausted':    fuel_exhausted,
            'reached_dest':      reached_dest,
            'timeout':           timeout,
            'deviation':         round(deviation, 3),
            'dist_to_dest':      round(dist_dest, 3),
            'min_threat_dist':   round(min_threat_dist, 3),
            'fuel_remaining':    round(self._fuel, 4),
            'fuel_cost_step':    round(fuel_cost, 4),
        }
        return self._obs(), reward, self._done, info

    def get_render_state(self) -> dict:
        """Full state dict for JSON streaming to the frontend."""
        forecasts = self._pred.predict(
            self._sat['pos'], self._sat['vel'], self._threats
        )
        nom_pos = self._nom_traj[min(self._step, MAX_STEPS)]
        return {
            'step':           self._step,
            'max_steps':      MAX_STEPS,
            'satellite':      {'pos': self._sat['pos'].tolist(),
                               'vel': self._sat['vel'].tolist()},
            'source':         self._src.tolist(),
            'destination':    self._dst.tolist(),
            'nominal_pos':    nom_pos.tolist(),
            'nominal_trajectory': [p.tolist() for p in self._nom_traj[::4]],
            'actual_trail':   self._actual_trail[-60:],   # last 60 positions
            'threats': [
                {'id':   t['id'], 'type': t['type'],
                 'pos':  t['pos'].tolist(), 'vel': t['vel'].tolist()}
                for t in self._threats
            ],
            'forecasts': [
                {'threat_id':          f.threat_id,
                 'threat_type':        f.threat_type,
                 'time_to_collision':  round(f.time_to_collision, 1),
                 'miss_distance':      round(f.miss_distance, 3),
                 'predicted_sat_pos':  f.predicted_sat_pos,
                 'predicted_threat_pos': f.predicted_threat_pos,
                 'is_critical':        f.is_critical}
                for f in forecasts
            ],
            'fuel_remaining': round(self._fuel, 4),
            'fuel_budget':    FUEL_BUDGET,
            'total_reward':   round(self._total_rew, 4),
            'done':           self._done,
            'level':          self.level,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _geometric_deviation(self, pos: np.ndarray) -> float:
        """Perpendicular distance from `pos` to the source→destination line.
        This is speed-independent: the agent can accelerate or decelerate
        along the track without penalty — only lateral offset counts."""
        ab = self._dst - self._src
        ab_len = np.linalg.norm(ab)
        if ab_len < 1e-9:
            return float(np.linalg.norm(pos - self._src))
        ap = pos - self._src
        cross = np.cross(ab, ap)
        return float(np.linalg.norm(cross) / ab_len)

    def _obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # Satellite state [0:6]
        obs[0:3] = self._sat['pos'] / SCALE
        obs[3:6] = self._sat['vel'] / 1.0    # vel already small (< 1 km/s)

        # Threats [6 : 6+MAX_THREATS*8]
        forecasts = self._pred.predict(
            self._sat['pos'], self._sat['vel'], self._threats
        )
        for i, f in enumerate(forecasts[:MAX_THREATS]):
            t = next(th for th in self._threats if th['id'] == f.threat_id)
            off = 6 + i * 8
            rel_pos = (t['pos'] - self._sat['pos']) / SCALE
            rel_vel = (t['vel'] - self._sat['vel']) / 1.0
            obs[off:off+3] = np.clip(rel_pos, -1, 1)
            obs[off+3:off+6] = np.clip(rel_vel, -1, 1)
            obs[off+6] = min(f.time_to_collision, MAX_STEPS * DT) / (MAX_STEPS * DT)
            obs[off+7] = min(f.miss_distance, 100.0) / 100.0

        # Mission [OBS_DIM-4 : OBS_DIM]
        deviation = self._geometric_deviation(self._sat['pos'])
        dist_dest = float(np.linalg.norm(self._sat['pos'] - self._dst))
        total_dist = float(np.linalg.norm(self._dst - self._src)) + 1e-6
        obs[-4] = dist_dest / total_dist
        obs[-3] = self._fuel / FUEL_BUDGET
        obs[-2] = min(deviation, SCALE) / SCALE
        obs[-1] = 1.0 - self._step / MAX_STEPS
        return obs

    def _reward(self, collision, fuel_exhausted, reached_dest,
                deviation, min_threat_dist, fuel_cost, dist_dest) -> float:
        total_dist = float(np.linalg.norm(self._dst - self._src)) + 1e-6

        # ── Terminal rewards ──
        if collision:      return -100.0
        if fuel_exhausted: return -30.0
        if reached_dest:
            eff = self._fuel / FUEL_BUDGET
            return 100.0 + 50.0 * eff     # big payoff for arriving

        # Timeout: penalise proportional to how far from destination
        if self._step >= MAX_STEPS:
            closeness = 1.0 - min(dist_dest / total_dist, 1.0)
            return -20.0 + 15.0 * closeness  # -20 if far, -5 if close

        # ── Per-step shaping ──
        r  = 0.0
        # Progress: reward getting closer to destination
        r += 0.3 * (1.0 - dist_dest / total_dist)
        # Deviation penalty: 10x stronger — forces return to path
        r -= 0.02 * min(deviation, SCALE)
        # Return-to-path bonus: reward when deviation is shrinking
        if deviation < 3.0:
            r += 0.1  # small bonus for staying near the nominal line
        # Fuel: light penalty for burns
        r -= 0.5 * fuel_cost
        # Threat proximity penalty
        if min_threat_dist < DANGER_DIST:
            r -= 1.0 * (1.0 - min_threat_dist / DANGER_DIST)
        return float(r)

    def _make_collision_threat(self, idx: int) -> dict:
        """Create a threat whose straight-line trajectory brings it close
        to the satellite's nominal path at a random future moment."""
        step_t = random.randint(30, MAX_STEPS - 30)
        sat_pos_at_t = self._nom_traj[step_t]

        # Near-miss offset (0.1 – 0.3 km so it's dangerous but not always a hit)
        offset = np.random.randn(3)
        offset /= (np.linalg.norm(offset) + 1e-9)
        offset *= random.uniform(0.1, 0.3)
        collision_pt = sat_pos_at_t + offset

        # Approach from a random direction
        direction = np.random.randn(3)
        direction /= (np.linalg.norm(direction) + 1e-9)

        t_collision = step_t * DT
        speed = random.uniform(0.05, 0.3)   # km/s
        start_pos = collision_pt - direction * speed * t_collision
        start_pos += np.random.normal(0, 5, 3)  # small jitter in start position

        return {
            'id':   f'THREAT-{idx}',
            'type': random.choice(['debris', 'satellite']),
            'pos':  start_pos,
            'vel':  direction * speed,
        }

    def _make_random_threat(self, idx: int) -> dict:
        """Create a threat with a random (likely non-colliding) trajectory."""
        pos = np.random.uniform(-200, 200, 3)
        vel = np.random.uniform(-0.1, 0.1, 3)
        return {
            'id':   f'THREAT-{idx}',
            'type': random.choice(['debris', 'satellite']),
            'pos':  pos,
            'vel':  vel,
        }
