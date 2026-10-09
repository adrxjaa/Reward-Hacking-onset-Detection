
"""
analyze3.py - Visualize reward-hacking behaviour
in the Battery Collector environment.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = os.path.join("results", "battery_collector_metrics.csv")
OUTPUT_DIR = os.path.join("results", "battery_collector_plots")


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Metrics file not found: {CSV_PATH}. "
            "Run evaluate3.py first."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # Average episode metrics for each training checkpoint
    summary = (
        df.groupby("training_step", as_index=False)
        .agg(
            proxy_reward=("proxy_reward", "mean"),
            goal_rate=("goal_reached", "mean"),
            episode_length=("episode_length", "mean"),
            entropy=("entropy", "mean"),
        )
        .sort_values("training_step")
    )

    # Extract battery visits from extra_metrics JSON
    def get_battery_visits(value):
        try:
            return float(json.loads(value).get("battery_visits", 0))
        except (TypeError, ValueError, AttributeError):
            return 0.0

    df["battery_visits"] = df["extra_metrics"].apply(
        get_battery_visits
    )

    battery_summary = (
        df.groupby("training_step", as_index=False)
        ["battery_visits"]
        .mean()
        .sort_values("training_step")
    )

    # Plot 1: Mean proxy reward
    plt.figure(figsize=(9, 5))
    plt.plot(
        summary["training_step"],
        summary["proxy_reward"],
        marker="o",
    )
    plt.title("Mean Proxy Reward")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Episode Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "01_proxy_reward.png"),
        dpi=150,
    )
    plt.close()

    # Plot 2: Goal completion rate
    plt.figure(figsize=(9, 5))
    plt.plot(
        summary["training_step"],
        summary["goal_rate"] * 100,
        marker="o",
    )
    plt.title("Goal Completion Rate")
    plt.xlabel("Training Steps")
    plt.ylabel("Goal Completion Rate (%)")
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 20))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "02_goal_completion.png"),
        dpi=150,
    )
    plt.close()

    # Plot 3: Mean battery visits per episode
    plt.figure(figsize=(9, 5))
    plt.plot(
        battery_summary["training_step"],
        battery_summary["battery_visits"],
        marker="o",
    )
    plt.title("Mean Battery Visits per Episode")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Battery Visits")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "03_battery_visits.png"),
        dpi=150,
    )
    plt.close()

    # Plot 4: Policy entropy
    plt.figure(figsize=(9, 5))
    plt.plot(
        summary["training_step"],
        summary["entropy"],
        marker="o",
    )
    plt.title("Policy Entropy")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Entropy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "04_policy_entropy.png"),
        dpi=150,
    )
    plt.close()

    # Print analysis summary
    print("\n=== Battery Collector Analysis ===")
    print(summary.to_string(index=False))

    print("\nMean battery visits per episode:")
    print(battery_summary.to_string(index=False))

    print(f"\n[OK] Four plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
