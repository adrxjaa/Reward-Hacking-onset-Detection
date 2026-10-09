"""
analyze.py - Plotting and analysis for the GridWorld baseline experiment.

Loads evaluation metrics from results/metrics.csv and generates the
specified data plots:
    - reward_over_training.png
    - goal_success_over_training.png
    - coin_visits_over_training.png
    - entropy_over_training.png

All plots are saved directly to results/.
"""

import os
import json
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import config


def load_and_aggregate_metrics(csv_path: str = None):
    """Load evaluation metrics CSV and compute per-step stats."""
    if csv_path is None:
        primary = os.path.join(config.RESULTS_DIR, "metrics.csv")
        fallback = os.path.join(config.RESULTS_DIR, "gridworld_metrics.csv")
        if os.path.exists(primary):
            csv_path = primary
        elif os.path.exists(fallback):
            csv_path = fallback
        else:
            raise FileNotFoundError(
                f"Metrics file not found at {primary} or {fallback}. "
                "Please run evaluate.py first."
            )

    print(f"Loading metrics from: {csv_path}")

    step_data = defaultdict(lambda: {
        "proxy_reward": [],
        "goal_reached": [],
        "coin_visits": [],
        "entropy": []
    })

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["training_step"])
            reward = float(row["proxy_reward"])
            goal = 1.0 if str(row["goal_reached"]).lower() in ["true", "1"] else 0.0
            entropy = float(row["entropy"])

            # Extract coin visits
            coin_visits = 0
            if "coin_visits" in row:
                try:
                    coin_visits = float(row["coin_visits"])
                except ValueError:
                    coin_visits = 0
            elif "extra_metrics" in row and row["extra_metrics"]:
                try:
                    extra = json.loads(row["extra_metrics"])
                    coin_visits = float(extra.get("coin_visits", 0))
                except Exception:
                    coin_visits = 0

            step_data[step]["proxy_reward"].append(reward)
            step_data[step]["goal_reached"].append(goal)
            step_data[step]["coin_visits"].append(coin_visits)
            step_data[step]["entropy"].append(entropy)

    sorted_steps = sorted(step_data.keys())
    agg = {
        "training_step": np.array(sorted_steps),
        "proxy_reward_mean": np.array([np.mean(step_data[s]["proxy_reward"]) for s in sorted_steps]),
        "goal_reached_mean": np.array([np.mean(step_data[s]["goal_reached"]) for s in sorted_steps]),
        "coin_visits_mean": np.array([np.mean(step_data[s]["coin_visits"]) for s in sorted_steps]),
        "entropy_mean": np.array([np.mean(step_data[s]["entropy"]) for s in sorted_steps]),
    }
    return agg


def plot_reward_over_training(agg: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    steps = agg["training_step"]
    rewards = agg["proxy_reward_mean"]

    ax.plot(steps, rewards, marker="o", color="#1f77b4", linewidth=2, markersize=6, label="Mean Proxy Reward")
    ax.set_title("Proxy Reward over Training", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Average Proxy Reward per Episode", fontsize=11)
    ax.set_xticks(steps)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "reward_over_training.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_goal_success_over_training(agg: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    steps = agg["training_step"]
    goal_rate = agg["goal_reached_mean"] * 100.0

    ax.plot(steps, goal_rate, marker="s", color="#2ca02c", linewidth=2, markersize=6, label="Goal Success Rate (%)")
    ax.set_title("Goal Success Rate over Training", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Goal Success Rate (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.set_xticks(steps)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "goal_success_over_training.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_coin_visits_over_training(agg: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    steps = agg["training_step"]
    coin_visits = agg["coin_visits_mean"]

    ax.plot(steps, coin_visits, marker="^", color="#d62728", linewidth=2, markersize=6, label="Average Coin Visits")
    ax.set_title("Coin Visits over Training", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Average Coin Visits per Episode", fontsize=11)
    ax.set_xticks(steps)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "coin_visits_over_training.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_entropy_over_training(agg: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    steps = agg["training_step"]
    entropy = agg["entropy_mean"]

    ax.plot(steps, entropy, marker="d", color="#9467bd", linewidth=2, markersize=6, label="Policy Entropy")
    ax.set_title("Policy Entropy over Training", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Average Policy Entropy", fontsize=11)
    ax.set_xticks(steps)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "entropy_over_training.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def analyze(results_dir: str = config.RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    agg = load_and_aggregate_metrics()

    print("\n--- Aggregated Training Dynamics ---")
    print(f"{'Step':>8} | {'Proxy':>8} | {'Goal%':>8} | {'CoinVisits':>10} | {'Entropy':>8}")
    print("-" * 52)
    for i, s in enumerate(agg["training_step"]):
        print(f"{s:8d} | {agg['proxy_reward_mean'][i]:8.2f} | {agg['goal_reached_mean'][i]*100:7.1f}% | {agg['coin_visits_mean'][i]:10.1f} | {agg['entropy_mean'][i]:8.4f}")
    print("------------------------------------\n")

    plot_reward_over_training(agg, results_dir)
    plot_goal_success_over_training(agg, results_dir)
    plot_coin_visits_over_training(agg, results_dir)
    plot_entropy_over_training(agg, results_dir)

    print(f"\n[OK] All plots successfully generated in: {results_dir}")


if __name__ == "__main__":
    analyze()
