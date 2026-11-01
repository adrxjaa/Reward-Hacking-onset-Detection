"""
train_detector.py

Trains the LSTM-DQN Reward Hacking Onset Detector on the training partition
and validates periodically on unseen validation runs.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rhd.detectors.environment import (
    load_episode_from_run,
    Normalizer,
    DetectorRewardConfig,
)
from rhd.detectors.agent import DQNConfig
from rhd.detectors.trainer import DetectorTrainer, TrainerConfig
from scripts.generate_detector_dataset import prepare_dataset


def train(
    metadata_dir: str = "results/detector",
    save_dir: str = "results/detector",
    num_episodes: int = 350,
    window_size: int = 4,
    learning_rate: float = 1e-3,
    gamma: float = 0.95,
    batch_size: int = 32,
    seed: int = 42,
):
    split_path = os.path.join(metadata_dir, "split_metadata.json")
    norm_path = os.path.join(metadata_dir, "norm_stats.json")

    # If dataset has not been generated yet, generate it
    if not os.path.exists(split_path) or not os.path.exists(norm_path):
        print("Dataset split or normalization statistics not found. Generating...")
        prepare_dataset(metadata_dir=metadata_dir)

    with open(split_path, "r") as f:
        split_meta = json.load(f)

    run_paths = split_meta["run_paths"]
    train_ids = split_meta["train"]
    val_ids = split_meta["validation"]

    print("=" * 60)
    print("  Training LSTM-DQN Reward Hacking Onset Detector")
    print("=" * 60)
    print(f"Train runs ({len(train_ids)}): {train_ids}")
    print(f"Val runs   ({len(val_ids)}): {val_ids}")

    train_episodes = [load_episode_from_run(run_paths[rid]) for rid in train_ids]
    val_episodes = [load_episode_from_run(run_paths[rid]) for rid in val_ids]

    train_episodes = [ep for ep in train_episodes if ep is not None]
    val_episodes = [ep for ep in val_episodes if ep is not None]

    assert len(train_episodes) > 0, "No valid training episodes found!"
    assert len(val_episodes) > 0, "No valid validation episodes found!"

    # Load fitted normalizer (strictly from training runs)
    normalizer = Normalizer.load(norm_path)

    trainer_cfg = TrainerConfig(
        num_episodes=num_episodes,
        warmup_steps=64,
        eval_freq=25,
        window_size=window_size,
        seed=seed,
        save_dir=save_dir,
    )

    dqn_cfg = DQNConfig(
        input_dim=10,
        hidden_dim=64,
        num_layers=1,
        num_actions=2,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        target_update_freq=20,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.985,
    )

    reward_cfg = DetectorRewardConfig()

    trainer = DetectorTrainer(
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        trainer_config=trainer_cfg,
        dqn_config=dqn_cfg,
        reward_config=reward_cfg,
        normalizer=normalizer,
    )

    print("\nStarting training loop...")
    history = trainer.train()
    print(f"\n[OK] Training complete. Best model checkpoint saved to: {os.path.join(save_dir, 'detector_best.pt')}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM-DQN Onset Detector.")
    parser.add_argument("--metadata-dir", type=str, default="results/detector")
    parser.add_argument("--save-dir", type=str, default="results/detector")
    parser.add_argument("--num-episodes", type=int, default=350)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        metadata_dir=args.metadata_dir,
        save_dir=args.save_dir,
        num_episodes=args.num_episodes,
        window_size=args.window_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        seed=args.seed,
    )
