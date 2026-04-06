"""
Baseline inference script — Satellite Collision Avoidance Environment
======================================================================
Runs a deterministic heuristic agent across all three task grades
(easy / medium / hard) and reports reproducible 0.0–1.0 scores.

Usage:
    python scripts/baseline.py                    # run all tasks, 10 episodes each
    python scripts/baseline.py --task easy        # single task
    python scripts/baseline.py --episodes 5       # fewer episodes
    python scripts/baseline.py --seed 42          # fix random seed
    python scripts/baseline.py --verbose          # step-by-step output

The heuristic policy:
    For each step, find the most critical danger pair (closest debris).
    Fire a burn in the direction AWAY from that debris, scaled by proximity.
    If no danger pairs exist, fire zero burn (coast).
"""

import sys, os, argparse, random, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SatAction
from tasks.graders import grade_easy, grade_medium, grade_hard, grade_all


# ── Heuristic policy ──────────────────────────────────────────────────────────
def heuristic_policy(obs) -> SatAction:
    """
    Greedy collision-avoidance policy:
      1. Find the closest (most critical) danger pair.
      2. Compute the unit vector FROM debris TO satellite.
      3. Scale burn by 1/distance (stronger burn = closer debris).
      4. Clamp to max 2.0 m/s per axis to conserve fuel.
    """
    # No threat — coast with zero burn
    if not obs.danger_pairs:
        sat_id = obs.satellites[0]["id"] if obs.satellites else "SAT-0"
        return SatAction(satellite_id=sat_id, delta_v=[0.0, 0.0, 0.0])

    # Sort by distance ascending → handle the most critical pair first
    pairs = sorted(obs.danger_pairs, key=lambda p: p["distance_km"])
    pair  = pairs[0]

    sat = next((s for s in obs.satellites if s["id"] == pair["sat_id"]), None)
    deb = next((d for d in obs.debris     if d["id"] == pair["debris_id"]), None)

    if sat is None or deb is None:
        return SatAction(satellite_id=pair["sat_id"], delta_v=[0.0, 0.0, 0.0])

    # Vector from debris → satellite (direction to move away)
    rx = sat["x"] - deb["x"]
    ry = sat["y"] - deb["y"]
    rz = sat["z"] - deb["z"]
    dist = math.sqrt(rx*rx + ry*ry + rz*rz) or 1.0

    # Burn magnitude: stronger when closer; capped at 2.0 m/s
    MAX_DV  = 2.0
    scale   = min(MAX_DV, 5.0 / dist)

    dv = [
        max(-MAX_DV, min(MAX_DV, (rx / dist) * scale)),
        max(-MAX_DV, min(MAX_DV, (ry / dist) * scale)),
        max(-MAX_DV, min(MAX_DV, (rz / dist) * scale)),
    ]
    return SatAction(satellite_id=pair["sat_id"], delta_v=dv)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Baseline heuristic agent for SATGUARD env")
    p.add_argument("--task",     choices=["easy", "medium", "hard", "all"],
                   default="all", help="Which task(s) to grade (default: all)")
    p.add_argument("--episodes", type=int, default=10,
                   help="Number of episodes per task (default: 10)")
    p.add_argument("--seed",     type=int, default=None,
                   help="Random seed for reproducibility (default: no seed)")
    p.add_argument("--verbose",  action="store_true",
                   help="Print step-by-step output")
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"[seed] random seed set to {args.seed}\n")

    print("=" * 52)
    print("  SATGUARD — Baseline Heuristic Agent")
    print("  Policy : greedy away-from-debris burn")
    print(f"  Task   : {args.task.upper()}")
    print(f"  N eps  : {args.episodes}")
    print("=" * 52)
    print()

    if args.task == "all":
        results = grade_all(heuristic_policy, n_episodes=args.episodes,
                            verbose=args.verbose)
        print()
        print("┌─────────────────────────────────┐")
        print(f"│  EASY   : {results['easy']:.4f}             │")
        print(f"│  MEDIUM : {results['medium']:.4f}             │")
        print(f"│  HARD   : {results['hard']:.4f}             │")
        print(f"│  ──────────────────────         │")
        print(f"│  TOTAL  : {results['total']:.4f}             │")
        print("└─────────────────────────────────┘")

    elif args.task == "easy":
        s = grade_easy(heuristic_policy, n_episodes=args.episodes,
                       verbose=args.verbose)
        print(f"\n  EASY score: {s:.4f}")

    elif args.task == "medium":
        s = grade_medium(heuristic_policy, n_episodes=args.episodes,
                         verbose=args.verbose)
        print(f"\n  MEDIUM score: {s:.4f}")

    elif args.task == "hard":
        s = grade_hard(heuristic_policy, n_episodes=args.episodes,
                       verbose=args.verbose)
        print(f"\n  HARD score: {s:.4f}")


if __name__ == "__main__":
    main()
