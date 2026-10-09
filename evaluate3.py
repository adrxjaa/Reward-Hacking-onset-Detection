
"""
evaluate3.py - Evaluation pipeline for Battery Collector.

Loads PPO checkpoints and records the shared metrics schema.
Results are saved to results/battery_collector_metrics.csv.
"""

import json
import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import config
import metrics as M
from environment3 import BatteryCollectorEnv, ENV_NAME


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


def evaluate():
    print(f"=== Evaluating '{ENV_NAME}' checkpoints ===\n")

    all_rows = []
    env = BatteryCollectorEnv()

    try:
        all_states = env.get_all_states()

        for step in config.CHECKPOINT_STEPS:
            ckpt = config.checkpoint_path(ENV_NAME, step)
            ckpt_file = ckpt if os.path.exists(ckpt) else ckpt + ".zip"

            if not os.path.exists(ckpt_file):
                print(f"[SKIP] Checkpoint not found: {ckpt_file}")
                continue

            model = PPO.load(ckpt_file, device="cpu")

            print(
                f"step={step:>6,} | "
                f"evaluating {config.EVAL_EPISODES} episodes ..."
            )

            sample_states = all_states[:config.ENTROPY_SAMPLE_STATES]
            avg_entropy = M.compute_avg_entropy(model, sample_states)

            episode_rewards = []
            episode_lengths = []
            goals_reached = []
            battery_visits_list = []

            for ep in range(config.EVAL_EPISODES):
                obs, _ = env.reset(seed=config.SEED + ep)

                done = False
                ep_reward = 0.0
                ep_len = 0
                ep_states = {}
                ep_action_counts = np.zeros(
                    env.action_space.n, dtype=np.int64
                )
                final_info = {}

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)

                    ep_action_counts[action] += 1

                    # Use the shared state-key convention from metrics.py.
                    # Convert observations to regular integer values first.
                    state_key = M.state_to_key(
                        np.asarray(obs, dtype=np.int64)
                    )
                    ep_states[state_key] = ep_states.get(state_key, 0) + 1

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    ep_reward += float(reward)
                    ep_len += 1
                    final_info = info

                goal_reached = bool(
                    final_info.get("goal_reached", False)
                )
                battery_visits = int(
                    final_info.get("battery_visits", 0)
                )

                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_len)
                goals_reached.append(int(goal_reached))
                battery_visits_list.append(battery_visits)

                action_distribution = (
                    ep_action_counts / max(int(ep_action_counts.sum()), 1)
                ).tolist()

                # Ensure all metric values are ordinary Python values.
                safe_state_visitation = make_json_safe(ep_states)
                safe_action_distribution = make_json_safe(
                    action_distribution
                )
                safe_extra_metrics = make_json_safe({
                    "battery_visits": battery_visits
                })

                row = M.build_metric_row(
                    environment=ENV_NAME,
                    training_step=int(step),
                    episode=int(ep),
                    proxy_reward=float(ep_reward),
                    episode_length=int(ep_len),
                    goal_reached=goal_reached,
                    entropy=float(avg_entropy),
                    state_visitation=safe_state_visitation,
                    action_distribution=safe_action_distribution,
                    extra_metrics=safe_extra_metrics,
                )

                all_rows.append(row)

            print(
                f"  mean reward={np.mean(episode_rewards):+.2f} | "
                f"mean length={np.mean(episode_lengths):.1f} | "
                f"goal rate={np.mean(goals_reached) * 100:.0f}% | "
                f"mean battery visits={np.mean(battery_visits_list):.1f} | "
                f"entropy={avg_entropy:.3f}"
            )

    finally:
        env.close()

    df = pd.DataFrame(all_rows, columns=M.COMMON_SCHEMA)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    output_path = config.results_csv_path(ENV_NAME)
    df.to_csv(output_path, index=False)

    print(f"\n[OK] Metrics saved to: {output_path}")
    print(f"Total evaluation rows: {len(df)}")

    return df


if __name__ == "__main__":
    evaluate()
