
"""
combine_metrics.py
Combine GridWorld and Battery Collector evaluation datasets.
"""

import json
import os
import pandas as pd

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


def main():
    # Confirm that both input datasets exist
    for path in [GRIDWORLD_PATH, BATTERY_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}")

    gridworld = pd.read_csv(GRIDWORLD_PATH)
    battery = pd.read_csv(BATTERY_PATH)

    # Verify the common schema
    for name, df in [
        ("GridWorld", gridworld),
        ("Battery Collector", battery),
    ]:
        if df.columns.tolist() != EXPECTED_COLUMNS:
            raise ValueError(
                f"{name} schema mismatch.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found: {df.columns.tolist()}"
            )

        if df.isna().any().any():
            raise ValueError(f"{name} contains missing values.")

        for column in [
            "state_visitation",
            "action_distribution",
            "extra_metrics",
        ]:
            for value in df[column]:
                json.loads(value)

    # Verify environment labels
    if set(gridworld["environment"].unique()) != {"gridworld"}:
        raise ValueError("Unexpected environment label in metrics.csv.")

    if set(battery["environment"].unique()) != {"battery_collector"}:
        raise ValueError(
            "Unexpected environment label in battery_collector_metrics.csv."
        )

    # Combine without modifying either original dataset
    combined = pd.concat(
        [gridworld, battery],
        ignore_index=True,
    )

    # Confirm checkpoint schedules match
    expected_steps = [0, 5000, 10000, 20000, 30000, 40000, 50000]

    for name, df in [
        ("GridWorld", gridworld),
        ("Battery Collector", battery),
    ]:
        actual_steps = sorted(df["training_step"].unique().tolist())
        if actual_steps != expected_steps:
            raise ValueError(
                f"{name} checkpoint mismatch: {actual_steps}"
            )

    os.makedirs("results", exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    # Read the saved file back and verify it
    check = pd.read_csv(OUTPUT_PATH)

    print("\n=== Combined Dataset Validation ===")
    print("Output:", OUTPUT_PATH)
    print("Shape:", check.shape)
    print("Columns:", check.columns.tolist())
    print("\nRows per environment:")
    print(check["environment"].value_counts().to_string())
    print("\nCheckpoint steps:")
    print(sorted(check["training_step"].unique().tolist()))
    print("Missing values:", int(check.isna().sum().sum()))

    print("\nValidating JSON fields...")
    for column in [
        "state_visitation",
        "action_distribution",
        "extra_metrics",
    ]:
        for value in check[column]:
            json.loads(value)

    print("JSON validation: PASS")
    print("\n[OK] Combined dataset created successfully.")


if __name__ == "__main__":
    main()
