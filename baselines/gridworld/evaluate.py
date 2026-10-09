"""
evaluate.py - Evaluation pipeline for GridWorld baseline.

Loads each checkpoint, runs EVAL_EPISODES episodes, and records the common
metrics defined in metrics.py. Saves results to results/metrics.csv.
"""

import os
import json
import csv
import torch
from stable_baselines3 import PPO
import numpy as np

import config
import metrics as M
from environment import GridWorldEnv, ENV_NAME


def evaluate():
    print(f"=== Evaluating '{ENV_NAME}' checkpoints ===\n")

    all_rows = []
    env = GridWorldEnv()

    # Enumerate all states once for average-entropy computation
    all_states = env.get_all_states()

    for step in config.CHECKPOINT_STEPS:
        ckpt = config.checkpoint_path(ENV_NAME, step)
        ckpt_file = ckpt if os.path.exists(ckpt) else ckpt + ".zip"
        if not os.path.exists(ckpt_file):
            print(f"  [SKIP] checkpoint not found: {ckpt}")
            continue

        model = PPO.load(ckpt, device="cpu")
        print(f"  step={step:>6,}  running {config.EVAL_EPISODES} episodes ...", end=" ")

        # Average policy entropy over sampled states
        avg_entropy = M.compute_avg_entropy(
            model,
            all_states[:config.ENTROPY_SAMPLE_STATES],
        )

        episode_rewards = []
        episode_lengths = []
        goals_reached   = []
        coin_visits_list= []

        # Run evaluation episodes
        for ep in range(config.EVAL_EPISODES):
            obs, info = env.reset(seed=config.SEED + ep)
            done = False
            ep_reward = 0.0
            ep_length = 0
            ep_states = {}
            action_counts = np.zeros(env.action_space.n)

            while not done:
                s_key = M.state_to_key(obs)
                ep_states[s_key] = ep_states.get(s_key, 0) + 1

                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                action_counts[action] += 1

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_length += 1
                done = terminated or truncated

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            goals_reached.append(float(info.get("goal_reached", False)))
            coin_visits_list.append(info.get("coin_visits", 0))

            action_dist = (
                (action_counts / action_counts.sum()).tolist()
                if action_counts.sum() > 0
                else []
            )

            row = M.build_metric_row(
                environment=ENV_NAME,
                training_step=step,
                episode=ep,
                proxy_reward=ep_reward,
                episode_length=ep_length,
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
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics.csv")
    try:
        import pandas as pd
        df = pd.DataFrame(all_rows, columns=M.COMMON_SCHEMA)
        df.to_csv(metrics_path, index=False)
    except ImportError:
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=M.COMMON_SCHEMA)
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"\n[OK] Metrics saved -> {metrics_path}")
    return all_rows


if __name__ == "__main__":
    evaluate()
