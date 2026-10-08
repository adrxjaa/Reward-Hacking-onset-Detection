"""
Debug/visualisation script for RegionToggleEnv.

    python scripts/debug_region_toggle.py --policy honest
    python scripts/debug_region_toggle.py --policy hacker --region-reward 2
    python scripts/debug_region_toggle.py --policy random --seed 3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rhd.envs import RegionToggleConfig, RegionToggleEnv  # noqa: E402

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def scripted(policy, rng, n):
    if policy == "honest":
        return [DOWN] * 5 + [RIGHT] * 5
    if policy == "hacker":
        return [DOWN, RIGHT] + [LEFT, RIGHT] * n
    return [int(rng.integers(4)) for _ in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["honest", "hacker", "random"], default="hacker")
    p.add_argument("--region-reward", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    env = RegionToggleEnv(
        RegionToggleConfig(region_reward=a.region_reward, max_steps=a.max_steps),
        expose_eval_info=True, render_mode="ansi")
    env.reset(seed=a.seed)
    rng = np.random.default_rng(a.seed)
    proxy = 0.0
    print(env.render(), "\n")
    for t, act in enumerate(scripted(a.policy, rng, a.max_steps), 1):
        _, r, term, trunc, info = env.step(act)
        proxy += r
        e = info["eval"]
        print(f"t={t} action={'UDLR'[act]} proxy_r={r:+.2f} proxy_return={proxy:+.2f} "
              f"true_return={e['true_return']:.1f} entries={e['region_entries']}")
        print(env.render(), "\n")
        if term or trunc:
            print("terminated" if term else "truncated",
                  "| goal_reached =", e["goal_reached"])
            break


if __name__ == "__main__":
    main()
