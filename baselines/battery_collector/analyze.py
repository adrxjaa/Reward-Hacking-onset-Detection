"""
analyze.py - Visualize reward-hacking behaviour in Battery Collector environment.
"""

import os
import json
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def main():
    csv_path = config.results_csv_path("battery_collector")
    out_dir = config.results_plots_dir("battery_collector")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metrics file not found: {csv_path}. Run evaluate.py first.")

    os.makedirs(out_dir, exist_ok=True)

    step_data = defaultdict(lambda: {
        "proxy_reward": [],
        "goal_reached": [],
        "episode_length": [],
        "battery_visits": [],
        "entropy": []
    })

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = int(row["training_step"])
            reward = float(row["proxy_reward"])
            goal = 1.0 if str(row["goal_reached"]).lower() in ["true", "1"] else 0.0
            length = float(row["episode_length"])
            entropy = float(row["entropy"])

            bv = 0.0
            if "extra_metrics" in row and row["extra_metrics"]:
                try:
                    bv = float(json.loads(row["extra_metrics"]).get("battery_visits", 0))
                except Exception:
                    bv = 0.0

            step_data[s]["proxy_reward"].append(reward)
            step_data[s]["goal_reached"].append(goal)
            step_data[s]["episode_length"].append(length)
            step_data[s]["battery_visits"].append(bv)
            step_data[s]["entropy"].append(entropy)

    sorted_steps = sorted(step_data.keys())
    steps = np.array(sorted_steps)
    mean_proxy = np.array([np.mean(step_data[s]["proxy_reward"]) for s in sorted_steps])
    mean_goal = np.array([np.mean(step_data[s]["goal_reached"]) * 100.0 for s in sorted_steps])
    mean_bv = np.array([np.mean(step_data[s]["battery_visits"]) for s in sorted_steps])
    mean_entropy = np.array([np.mean(step_data[s]["entropy"]) for s in sorted_steps])

    # Plot 1: Proxy Reward
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(steps, mean_proxy, marker="o", color="#1f77b4", linewidth=2)
    ax.set_title("Proxy Reward over Training - Battery Collector", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Proxy Reward")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_proxy_reward.png"))
    plt.close(fig)

    # Plot 2: Goal Completion
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(steps, mean_goal, marker="s", color="#2ca02c", linewidth=2)
    ax.set_title("Goal Completion Rate (%) - Battery Collector", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Goal Completion (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_goal_completion.png"))
    plt.close(fig)

    # Plot 3: Battery Visits
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(steps, mean_bv, marker="^", color="#d62728", linewidth=2)
    ax.set_title("Battery Visits over Training - Battery Collector", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Battery Visits")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_battery_visits.png"))
    plt.close(fig)

    # Plot 4: Entropy
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(steps, mean_entropy, marker="d", color="#9467bd", linewidth=2)
    ax.set_title("Policy Entropy over Training - Battery Collector", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Policy Entropy")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_policy_entropy.png"))
    plt.close(fig)

    print(f"\n[OK] Battery Collector plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
