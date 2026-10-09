"""
combine_metrics.py
Combine GridWorld and Battery Collector evaluation datasets into a unified CSV.
"""

import json
import os
import csv

GRIDWORLD_PATH = os.path.join("results", "metrics.csv")
BATTERY_PATH = os.path.join("results", "battery_collector_metrics.csv")
OUTPUT_PATH = os.path.join("results", "combined_metrics.csv")

EXPECTED_COLUMNS = [
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


def combine():
    for path in [GRIDWORLD_PATH, BATTERY_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}")

    combined_rows = []
    for path in [GRIDWORLD_PATH, BATTERY_PATH]:
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                combined_rows.append(row)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"[OK] Combined metrics successfully created: {OUTPUT_PATH} ({len(combined_rows)} rows)")


if __name__ == "__main__":
    combine()
