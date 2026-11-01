"""
generate_detector_dataset.py

Generates a multi-seed dataset of generator runs across clean and exploit regimes,
extracts non-leaking behavioral features, computes mathematical onsets,
and creates strict run-level train/validation/test splits with normalization stats.
"""

import os
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rhd.envs import RegionToggleConfig
from rhd.generators.ppo import PPOConfig, run_generator
from rhd.features.extract import FeatureExtractor
from rhd.evaluation.onset import process_run
from rhd.detectors.environment import Normalizer, load_episode_from_run


def _worker_train_run(args):
    run_dir, weight, seed, timesteps = args
    if not os.path.exists(os.path.join(run_dir, "checkpoints")):
        print(f"Generating run: {os.path.basename(run_dir)} (w={weight}, seed={seed})...")
        env_cfg = RegionToggleConfig(region_reward=weight, max_steps=100)
        ppo_cfg = PPOConfig(
            seed=seed,
            total_timesteps=timesteps,
            checkpoint_interval=5000,
            eval_episodes=10,
        )
        run_generator(run_dir, env_cfg, ppo_cfg)
    
    # Feature extraction
    feat_dir = os.path.join(run_dir, "features")
    temp_path = os.path.join(feat_dir, "temporal_features.jsonl")
    if not os.path.exists(temp_path):
        print(f"Extracting features for {os.path.basename(run_dir)}...")
        extractor = FeatureExtractor(run_dir)
        extractor.run_extraction()

    # Onset labeling
    onset_path = os.path.join(feat_dir, "onset.json")
    if not os.path.exists(onset_path):
        print(f"Labeling onset for {os.path.basename(run_dir)}...")
        process_run(run_dir)

    return run_dir


def prepare_dataset(
    output_dir: str = "experiments/detector_dataset",
    metadata_dir: str = "results/detector",
    timesteps: int = 30000,
    max_workers: int = 4,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # 4 seeds x 3 reward weights = 12 total runs
    # w=0.0: clean controls (negative class)
    # w=1.0: standard exploit (positive class)
    # w=5.0: severe exploit (positive class)
    weights = [0.0, 1.0, 5.0]
    seeds = [42, 101, 202, 303]

    tasks = []
    run_lookup = {}

    for w in weights:
        for s in seeds:
            # Check if seed 42 exists in experiments/pilot/run_w{w}
            pilot_dir = os.path.join("experiments", "pilot", f"run_w{w}")
            if s == 42 and os.path.exists(pilot_dir):
                run_dir = pilot_dir
            else:
                run_dir = os.path.join(output_dir, f"run_w{w}_s{s}")
            
            run_id = os.path.basename(os.path.normpath(run_dir))
            run_lookup[(w, s)] = run_dir
            tasks.append((run_dir, w, s, timesteps))

    print(f"Preparing dataset with {len(tasks)} runs across weights {weights} and seeds {seeds}...")
    
    # Execute in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker_train_run, t) for t in tasks]
        for f in as_completed(futures):
            res_dir = f.result()
            print(f"  [DONE] {os.path.basename(res_dir)}")

    # Define run-level splits:
    # Train: seeds 42 and 101 (6 runs: 2 clean, 2 w=1.0, 2 w=5.0)
    # Validation: seed 202 (3 runs: 1 clean, 1 w=1.0, 1 w=5.0)
    # Test: seed 303 (3 runs: 1 clean, 1 w=1.0, 1 w=5.0)
    train_runs = [run_lookup[(w, s)] for w in weights for s in [42, 101]]
    val_runs = [run_lookup[(w, s)] for w in weights for s in [202]]
    test_runs = [run_lookup[(w, s)] for w in weights for s in [303]]

    split_metadata = {
        "train": [os.path.basename(r) for r in train_runs],
        "validation": [os.path.basename(r) for r in val_runs],
        "test": [os.path.basename(r) for r in test_runs],
        "run_paths": {
            os.path.basename(r): r for r in train_runs + val_runs + test_runs
        },
        "description": "Stratified run-level partitioning ensuring zero checkpoint leakage across splits."
    }

    split_path = os.path.join(metadata_dir, "split_metadata.json")
    with open(split_path, "w") as f:
        json.dump(split_metadata, f, indent=2)
    print(f"[OK] Split metadata saved to {split_path}")

    # Compute normalization statistics STRICTLY on training episodes
    train_episodes = []
    for r in train_runs:
        ep = load_episode_from_run(r)
        if ep is not None:
            train_episodes.append(ep)

    normalizer = Normalizer()
    normalizer.fit([ep.features for ep in train_episodes])
    norm_path = os.path.join(metadata_dir, "norm_stats.json")
    normalizer.save(norm_path)
    print(f"[OK] Training-fitted normalization stats saved to {norm_path} (zero leakage)")

    return split_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate detector multi-seed dataset.")
    parser.add_argument("--output-dir", type=str, default="experiments/detector_dataset")
    parser.add_argument("--metadata-dir", type=str, default="results/detector")
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    prepare_dataset(
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        timesteps=args.timesteps,
        max_workers=args.workers,
    )
