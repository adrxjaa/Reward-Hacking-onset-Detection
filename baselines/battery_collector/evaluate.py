"""
evaluate.py - Evaluation pipeline for Battery Collector.

Loads PPO checkpoints and records the shared metrics schema.
Results are saved to results/battery_collector_metrics.csv.
"""

import json
import os
import sys
import csv
import numpy as np
import torch
from stable_baselines3 import PPO

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import config
from environment import BatteryCollectorEnv, ENV_NAME

COMMON_SCHEMA = [
    "environment",
    "training_step",
    "episode",
    "proxy_reward",
    "episode_length",
    "goal_reached",
    "entropy",
    "state_visitation",
    "action_distribution",
    "extra_metrics",
]


def make_json_safe(value):
    """Convert NumPy values and nested structures to JSON-safe values."""
    if isinstance(value, dict):
        return {
            str(make_json_safe(key)): make_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def compute_avg_entropy(model, states):
    """Compute average policy entropy across sampled states."""
    obs_tensor = torch.as_tensor(np.array(states, dtype=np.float32)).to(model.device)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()
        probs = np.clip(probs, 1e-10, 1.0)
        entropies = -np.sum(probs * np.log(probs), axis=-1)
        return float(np.mean(entropies))


def evaluate():
    print(f"=== Evaluating '{ENV_NAME}' checkpoints ===\n")

    all_rows = []
    env = BatteryCollectorEnv()

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
        sample_states = all_states[:config.ENTROPY_SAMPLE_STATES]
        avg_entropy = compute_avg_entropy(model, sample_states)

        episode_rewards = []
        episode_lengths = []
        goals_reached = []
        battery_visits_list = []

        for ep in range(config.EVAL_EPISODES):
            obs, info = env.reset(seed=config.SEED + ep)
            done = False
            ep_reward = 0.0
            ep_length = 0
            ep_states = {}
            action_counts = np.zeros(env.action_space.n)

            while not done:
                s_key = str((int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3])))
                ep_states[s_key] = ep_states.get(s_key, 0) + 1

                action, _ = model.predict(obs, deterministic=True)
                action_counts[action] += 1

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_length += 1
                done = terminated or truncated

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            goals_reached.append(float(info.get("goal_reached", False)))
            battery_visits_list.append(info.get("battery_visits", 0))

            action_dist = (
                (action_counts / action_counts.sum()).tolist()
                if action_counts.sum() > 0
                else []
            )

            row = {
                "environment": ENV_NAME,
                "training_step": step,
                "episode": ep,
                "proxy_reward": float(ep_reward),
                "episode_length": int(ep_length),
                "goal_reached": bool(info.get("goal_reached", False)),
                "entropy": float(avg_entropy),
                "state_visitation": json.dumps(make_json_safe(ep_states)),
                "action_distribution": json.dumps(make_json_safe(action_dist)),
                "extra_metrics": json.dumps({"battery_visits": int(info.get("battery_visits", 0))}),
            }
            all_rows.append(row)

        mean_r = np.mean(episode_rewards)
        mean_l = np.mean(episode_lengths)
        goal_rt = np.mean(goals_reached)
        mean_bv = np.mean(battery_visits_list)
        print(f"reward={mean_r:+.2f}  len={mean_l:.1f}  goal%={goal_rt*100:.0f}%  "
              f"batteries={mean_bv:.1f}  H={avg_entropy:.3f}")

    # Save CSV
    out_path = config.results_csv_path(ENV_NAME)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        import pandas as pd
        df = pd.DataFrame(all_rows, columns=COMMON_SCHEMA)
        df.to_csv(out_path, index=False)
    except ImportError:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COMMON_SCHEMA)
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\n[OK] Metrics saved -> {out_path}")
    return all_rows


if __name__ == "__main__":
    evaluate()
