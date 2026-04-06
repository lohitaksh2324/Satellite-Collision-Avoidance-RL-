import uuid, random
from models import SatAction, SatObservation, SatState

# ── Physics constants (imported by graders.py) ───────────────────────────────
COLLISION_KM = 0.5    # distance at which a collision is declared
DANGER_KM    = 5.0    # reference danger threshold (medium task)

# ── Task presets (imported by app.py) ────────────────────────────────────────
TASK_CONFIGS = {
    "easy":   {"debris_count": 10, "sat_count": 2,  "danger_threshold_km": 3.0, "max_steps": 100},
    "medium": {"debris_count": 20, "sat_count": 3,  "danger_threshold_km": 5.0, "max_steps": 200},
    "hard":   {"debris_count": 40, "sat_count": 5,  "danger_threshold_km": 8.0, "max_steps": 500},
}

# Defaults (used when no task is specified)
DEBRIS_COUNT        = 20
SAT_COUNT           = 3
DANGER_THRESHOLD_KM = 5.0

class SatelliteCollisionEnv:

    def __init__(self, task: str = "medium"):
        cfg = TASK_CONFIGS.get(task, TASK_CONFIGS["medium"])
        self._task              = task
        self._debris_count      = cfg["debris_count"]
        self._sat_count         = cfg["sat_count"]
        self._danger_threshold  = cfg["danger_threshold_km"]
        self._max_steps         = cfg["max_steps"]
        self._sats:   list = []
        self._debris: list = []
        self._state = SatState()

    def reset(self) -> SatObservation:
        self._sats   = [self._rand_sat(f"SAT-{i}")   for i in range(self._sat_count)]
        self._debris = [self._rand_debris(f"DEB-{i}") for i in range(self._debris_count)]
        self._state  = SatState(episode_id=str(uuid.uuid4()), step_count=0)
        return self._observe(reward=None, done=False, msg="Episode started.")

    def step(self, action: SatAction) -> SatObservation:
        self._state.step_count += 1
        self._state.time_elapsed_s += 10.0   # 10s timestep

        # Compute fuel cost: |Δv| = Euclidean magnitude of the burn vector
        dv = action.delta_v
        fuel_this_step = (dv[0]**2 + dv[1]**2 + dv[2]**2) ** 0.5 if len(dv) >= 3 else 0.0
        action.fuel_consumed_current_action = round(fuel_this_step, 6)
        self._state.fuel_consumed_total = round(
            self._state.fuel_consumed_total + fuel_this_step, 6
        )

        # Apply delta-v nudge
        sat = next((s for s in self._sats if s["id"] == action.satellite_id), None)
        if sat:
            for i, axis in enumerate(["vx", "vy", "vz"]):
                sat[axis] += action.delta_v[i] if i < len(action.delta_v) else 0

        # Propagate all objects (simple Euler for now)
        for obj in self._sats + self._debris:
            obj["x"] += obj["vx"] * 10
            obj["y"] += obj["vy"] * 10
            obj["z"] += obj["vz"] * 10

        # Check collisions
        danger_pairs = self._check_danger()
        collision = any(p["distance_km"] < COLLISION_KM for p in danger_pairs)
        done = collision or self._state.step_count >= self._max_steps
        if not collision and danger_pairs:
            self._state.collision_avoided += 1

        reward = self._reward(danger_pairs, collision)
        msg = "Collision!" if collision else f"{len(danger_pairs)} danger pairs"
        return self._observe(reward=reward, done=done, msg=msg)

    @property
    def state(self) -> SatState:
        return self._state

    # ── helpers ──────────────────────────────────────────────────
    def _rand_sat(self, sid):
        return dict(id=sid,
            x=random.uniform(-500,500), y=random.uniform(-500,500),
            z=random.uniform(-500,500),
            vx=random.uniform(-0.1,0.1), vy=random.uniform(-0.1,0.1),
            vz=random.uniform(-0.1,0.1))

    def _rand_debris(self, did):
        return dict(id=did,
            x=random.uniform(-600,600), y=random.uniform(-600,600),
            z=random.uniform(-600,600),
            vx=random.uniform(-0.2,0.2), vy=random.uniform(-0.2,0.2),
            vz=random.uniform(-0.2,0.2))

    def _check_danger(self):
        pairs = []
        for s in self._sats:
            for d in self._debris:
                dist = ((s["x"]-d["x"])**2+(s["y"]-d["y"])**2+(s["z"]-d["z"])**2)**0.5
                if dist < self._danger_threshold:
                    pairs.append({"sat_id":s["id"],"debris_id":d["id"],"distance_km":round(dist,3)})
        return pairs

    def _reward(self, danger_pairs, collision):
        if collision: return -1.0
        if not danger_pairs: return 0.1            # safe — small positive
        min_dist = min(p["distance_km"] for p in danger_pairs)
        return round((min_dist / self._danger_threshold) * 0.5, 4)   # partial credit

    def _observe(self, reward, done, msg):
        return SatObservation(
            done=done, reward=reward,
            satellites=self._sats, debris=self._debris,
            danger_pairs=self._check_danger(), message=msg)