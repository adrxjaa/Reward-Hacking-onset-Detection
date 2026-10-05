"""
evaluate.py - Generic evaluation pipeline.

Loads each checkpoint, runs EVAL_EPISODES episodes, and records the common
metrics defined in metrics.py. Saves one CSV per environment.

Usage:
    python evaluate.py

Output:
    results/gridworld_metrics.csv

The CSV columns follow COMMON_SCHEMA so that a second environment's CSV
can be concatenated directly for cross-environment analysis.
"""

import os
import json
import torch
from stable_baselines3 import PPO
import numpy as np
import pandas as pd

import config
import metrics as M
# change the file name with your respective file
from environment import GridWorldEnv, ENV_NAME


def evaluate():
    print(f"=== Evaluating '{ENV_NAME}' checkpoints ===\n")

    all_rows = []
    env = GridWorldEnv()

    # Enumerate all states once for average-entropy computation
    all_states = env.get_all_states()

    for step in config.CHECKPOINT_STEPS:
        ckpt = config.checkpoint_path(ENV_NAME, step)
        # SB3 appends .zip automatically; check both
        ckpt_file = ckpt if os.path.exists(ckpt) else ckpt + ".zip"
        if not os.path.exists(ckpt_file):
            print(f"  [SKIP] checkpoint not found: {ckpt}")
            continue

        model = PPO.load(ckpt, device="cpu")
        print(f"  step={step:>6,}  running {config.EVAL_EPISODES} episodes ...", end=" ")

        # Average policy entropy over sampled states
        sample_states = all_states[:config.ENTROPY_SAMPLE_STATES]
        avg_entropy = M.compute_avg_entropy(model, sample_states)

        # Run evaluation episodes
        episode_rewards  = []
        episode_lengths  = []
        goals_reached    = []
        coin_visits_list = []
        state_visit_counter = {}
        action_counts = np.zeros(env.action_space.n)

        for ep in range(config.EVAL_EPISODES):
            obs, _ = env.reset(seed=config.SEED + ep)
            done   = False
            ep_reward = 0.0
            ep_len    = 0
            ep_states = {}

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action     = int(action)
                action_counts[action] += 1

                # track state visitation
                sk = M.state_to_key(obs)
                ep_states[sk] = ep_states.get(sk, 0) + 1

                obs, reward, terminated, truncated, info = env.step(action)
                done       = terminated or truncated
                ep_reward += reward
                ep_len    += 1

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            goals_reached.append(int(info.get("goal_reached", False)))
            coin_visits_list.append(info.get("coin_visits", 0))

            # Aggregate state visitation across all episodes
            for sk, cnt in ep_states.items():
                state_visit_counter[sk] = state_visit_counter.get(sk, 0) + cnt

            # Build per-episode row
            action_dist = (action_counts / max(action_counts.sum(), 1)).tolist()
            row = M.build_metric_row(
                environment=ENV_NAME,
                training_step=step,
                episode=ep,
                proxy_reward=ep_reward,
                episode_length=ep_len,
                goal_reached=bool(info.get("goal_reached", False)),
                entropy=avg_entropy,
                state_visitation=ep_states,
                action_distribution=action_dist,
                extra_metrics={"coin_visits": info.get("coin_visits", 0)},
            )
            all_rows.append(row)

        mean_r  = np.mean(episode_rewards)
        mean_l  = np.mean(episode_lengths)
        goal_rt = np.mean(goals_reached)
        mean_cv = np.mean(coin_visits_list)
        print(f"reward={mean_r:+.2f}  len={mean_l:.1f}  goal%={goal_rt*100:.0f}%  "
              f"coins={mean_cv:.1f}  H={avg_entropy:.3f}")

    # Save CSV
    df  = pd.DataFrame(all_rows, columns=M.COMMON_SCHEMA)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics.csv")
    df.to_csv(metrics_path, index=False)
    print(f"\n[OK] Metrics saved -> {metrics_path}")
    return df


if __name__ == "__main__":
    evaluate()
