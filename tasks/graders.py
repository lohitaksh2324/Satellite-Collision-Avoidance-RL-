"""
Agent graders — easy → medium → hard, scores 0.0–1.0.
Each grader runs N episodes and returns a normalized score.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Callable, Optional
from server.environment import SatelliteCollisionEnv, COLLISION_KM, DANGER_KM
from models import SatAction


# ── Policy type ───────────────────────────────────────────────────────────
Policy = Callable[[object], SatAction]   # obs → SatAction


# ── Episode runner ────────────────────────────────────────────────────────
def _run_episode(env: SatelliteCollisionEnv,
                 policy: Policy,
                 task: str,
                 verbose: bool = False) -> dict:
    obs = env.reset()
    total_reward     = 0.0
    steps            = 0
    collisions       = 0
    danger_events    = 0
    min_dist         = 9999.0

    while not obs.done:
        action = policy(obs)
        obs    = env.step(action)
        steps += 1

        if obs.reward is not None:
            total_reward += obs.reward

        if obs.danger_pairs:
            danger_events += 1
            d = min(p["distance_km"] for p in obs.danger_pairs)
            if d < min_dist:
                min_dist = d
            if d < COLLISION_KM:
                collisions += 1

        if verbose:
            print(f"  step {steps:3d} | reward {obs.reward or 0:.4f} | {obs.message}")

    return {
        "task":           task,
        "total_reward":   total_reward,
        "steps":          steps,
        "collisions":     collisions,
        "danger_events":  danger_events,
        "min_dist_km":    min_dist if min_dist < 9999 else None,
        "survived":       collisions == 0,
    }


# ── Scoring helpers ───────────────────────────────────────────────────────
def _normalize(value: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _score_episode(ep: dict, max_steps: int) -> float:
    """Convert raw episode stats to a 0.0–1.0 score."""
    if not ep["survived"]:
        return 0.0    # collision = zero score

    # Component 1: average reward per step (0–1)
    avg_r = ep["total_reward"] / max(ep["steps"], 1)
    reward_score = _normalize(avg_r, 0.0, 0.1)

    # Component 2: how close did things get?  further = better
    if ep["min_dist_km"] is not None:
        dist_score = _normalize(ep["min_dist_km"], COLLISION_KM, DANGER_KM)
    else:
        dist_score = 1.0   # never entered danger zone

    # Component 3: step efficiency (finishing sooner is not rewarded — use all steps)
    step_score = ep["steps"] / max_steps

    # Weighted composite
    score = 0.4 * reward_score + 0.4 * dist_score + 0.2 * step_score
    return round(score, 4)


# ── Graders ───────────────────────────────────────────────────────────────
def grade_easy(policy: Policy, n_episodes: int = 10,
               verbose: bool = False) -> float:
    """
    Easy: 1 satellite, 5 debris, 100 steps.
    Score 0.0–1.0 averaged over n_episodes.
    """
    env    = SatelliteCollisionEnv(task="easy")
    scores = []
    for i in range(n_episodes):
        ep    = _run_episode(env, policy, "easy", verbose=verbose)
        score = _score_episode(ep, max_steps=100)
        scores.append(score)
        if verbose:
            print(f"[easy] ep {i+1}: score={score:.4f} survived={ep['survived']}")
    final = round(sum(scores) / len(scores), 4)
    print(f"[EASY]   avg score over {n_episodes} eps: {final:.4f}")
    return final


def grade_medium(policy: Policy, n_episodes: int = 10,
                 verbose: bool = False) -> float:
    """
    Medium: 3 satellites, 20 debris, 200 steps.
    Score 0.0–1.0 averaged over n_episodes.
    """
    env    = SatelliteCollisionEnv(task="medium")
    scores = []
    for i in range(n_episodes):
        ep    = _run_episode(env, policy, "medium", verbose=verbose)
        score = _score_episode(ep, max_steps=200)
        scores.append(score)
        if verbose:
            print(f"[medium] ep {i+1}: score={score:.4f} survived={ep['survived']}")
    final = round(sum(scores) / len(scores), 4)
    print(f"[MEDIUM] avg score over {n_episodes} eps: {final:.4f}")
    return final


def grade_hard(policy: Policy, n_episodes: int = 10,
               verbose: bool = False) -> float:
    """
    Hard: 5 satellites, 50 debris, simultaneous conjunctions, 500 steps.
    Score 0.0–1.0 averaged over n_episodes.
    """
    env    = SatelliteCollisionEnv(task="hard")
    scores = []
    for i in range(n_episodes):
        ep    = _run_episode(env, policy, "hard", verbose=verbose)
        score = _score_episode(ep, max_steps=500)
        scores.append(score)
        if verbose:
            print(f"[hard] ep {i+1}: score={score:.4f} survived={ep['survived']}")
    final = round(sum(scores) / len(scores), 4)
    print(f"[HARD]   avg score over {n_episodes} eps: {final:.4f}")
    return final


def grade_all(policy: Policy, n_episodes: int = 10,
              verbose: bool = False) -> dict:
    """Run all three graders and return a summary dict."""
    easy   = grade_easy(policy,   n_episodes, verbose)
    medium = grade_medium(policy, n_episodes, verbose)
    hard   = grade_hard(policy,   n_episodes, verbose)
    total  = round((easy + medium + hard) / 3, 4)
    result = {
        "easy":   easy,
        "medium": medium,
        "hard":   hard,
        "total":  total,
    }
    print(f"\n{'='*40}")
    print(f"TOTAL SCORE: {total:.4f}")
    print(f"{'='*40}")
    return result