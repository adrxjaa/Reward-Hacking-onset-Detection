"""
analyze.py - Plotting and analysis for the reward-hacking experiment.

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
import torch  # loaded early to prevent Windows OpenMP/DLL runtime conflicts
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import config


def load_metrics(csv_path: str = None) -> pd.DataFrame:
    """Load evaluation metrics CSV from results/metrics.csv."""
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
    df = pd.read_csv(csv_path)

    # Parse extra_metrics to extract coin_visits if present
    def extract_coin_visits(val):
        if isinstance(val, str):
            try:
                data = json.loads(val)
                return data.get("coin_visits", 0)
            except Exception:
                return 0
        elif isinstance(val, dict):
            return val.get("coin_visits", 0)
        return 0

    if "coin_visits" not in df.columns:
        if "extra_metrics" in df.columns:
            df["coin_visits"] = df["extra_metrics"].apply(extract_coin_visits)
        else:
            df["coin_visits"] = 0

    return df


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-checkpoint mean and std for each key metric."""
    grouped = df.groupby("training_step")
    agg = grouped.agg(
        proxy_reward_mean=("proxy_reward", "mean"),
        proxy_reward_std=("proxy_reward", "std"),
        goal_reached_mean=("goal_reached", "mean"),
        coin_visits_mean=("coin_visits", "mean"),
        coin_visits_std=("coin_visits", "std"),
        entropy_mean=("entropy", "mean"),
    ).reset_index()
    return agg


def plot_reward_over_training(agg: pd.DataFrame, out_dir: str):
    """Plot average proxy reward vs training steps."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    steps = agg["training_step"].values
    rewards = agg["proxy_reward_mean"].values

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


def plot_goal_success_over_training(agg: pd.DataFrame, out_dir: str):
    """Plot goal success rate vs training steps."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    steps = agg["training_step"].values
    goal_rate = agg["goal_reached_mean"].values * 100.0

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


def plot_coin_visits_over_training(agg: pd.DataFrame, out_dir: str):
    """Plot average coin visits vs training steps."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    steps = agg["training_step"].values
    coin_visits = agg["coin_visits_mean"].values

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


def plot_entropy_over_training(agg: pd.DataFrame, out_dir: str):
    """Plot average policy entropy vs training steps."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    steps = agg["training_step"].values
    entropy = agg["entropy_mean"].values

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
    """Run analysis and generate the four requested plots."""
    os.makedirs(results_dir, exist_ok=True)
    df = load_metrics()
    agg = aggregate_metrics(df)

    print("\n--- Aggregated Training Dynamics ---")
    print(agg[["training_step", "proxy_reward_mean", "goal_reached_mean", "coin_visits_mean", "entropy_mean"]].to_string(index=False))
    print("------------------------------------\n")

    plot_reward_over_training(agg, results_dir)
    plot_goal_success_over_training(agg, results_dir)
    plot_coin_visits_over_training(agg, results_dir)
    plot_entropy_over_training(agg, results_dir)

    print(f"\n[OK] All plots successfully generated in: {results_dir}")


if __name__ == "__main__":
    analyze()
