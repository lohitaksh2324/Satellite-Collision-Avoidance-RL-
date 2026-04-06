"""
Standalone training script (no server).
Run this to pre-train checkpoints before launching the live UI.

Usage:
  python drqn_satellite/train.py --level easy --episodes 500
  python drqn_satellite/train.py --level all  --episodes 600
"""

import argparse, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from environment import SatelliteTrajectoryEnv, N_ACTIONS, OBS_DIM
from drqn_agent import DRQNAgent

CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)


def train(level: str, n_episodes: int, verbose: bool = True):
    print(f"\n{'='*52}")
    print(f"  DRQN Training — {level.upper()}  |  {n_episodes} episodes")
    print(f"{'='*52}")

    env   = SatelliteTrajectoryEnv(level=level)
    agent = DRQNAgent(OBS_DIM, N_ACTIONS)

    ckpt = os.path.join(CKPT_DIR, f'drqn_{level}.pth')
    if os.path.exists(ckpt):
        agent.load(ckpt)
        print(f"  Resumed from {ckpt}  (ep {agent.episode_count}, ε={agent.epsilon:.3f})")

    reward_history, success_count = [], 0

    for ep in range(n_episodes):
        obs    = env.reset()
        hidden = agent.init_hidden()
        transitions, total_r = [], 0.0

        done = False
        while not done:
            action, hidden = agent.select_action(obs, hidden)
            next_obs, reward, done, info = env.step(action)
            transitions.append((obs, action, reward, next_obs, float(done)))
            obs = next_obs
            total_r += reward

        agent.buffer.add(transitions)
        agent.episode_count += 1
        loss = agent.update()
        agent.decay_epsilon()
        if ep % agent.target_update == 0:
            agent.update_target()

        reward_history.append(total_r)
        if info.get('reached_dest'): success_count += 1

        if verbose and (ep % 20 == 0 or ep == n_episodes - 1):
            avg = np.mean(reward_history[-30:]) if len(reward_history) >= 30 else np.mean(reward_history)
            sr  = success_count / (ep + 1) * 100
            loss_str = f"{loss:.4f}" if loss is not None else "—"
            print(f"  ep {ep:4d} | rew {total_r:7.2f} | avg {avg:7.2f} | "
                  f"ε {agent.epsilon:.3f} | success {sr:4.1f}% | "
                  f"loss {loss_str:>8}" )

        if ep > 0 and ep % 100 == 0:
            agent.save(ckpt)
            print(f"  ✔ Checkpoint saved → {ckpt}")

    agent.save(ckpt)
    sr = success_count / n_episodes * 100
    print(f"\n  ✅ Done. Success rate: {sr:.1f}%  |  Checkpoint: {ckpt}\n")
    return sr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--level',    choices=['easy','medium','hard','all'], default='easy')
    p.add_argument('--episodes', type=int, default=400)
    p.add_argument('--quiet',    action='store_true')
    args = p.parse_args()

    levels = ['easy','medium','hard'] if args.level == 'all' else [args.level]
    for lv in levels:
        train(lv, args.episodes, verbose=not args.quiet)


if __name__ == '__main__':
    main()
