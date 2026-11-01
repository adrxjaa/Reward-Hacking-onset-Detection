"""
generate_extended_dataset.py

Generates an independent, large-scale validation dataset of 18 generator runs
across 6 completely unseen random seeds and 3 reward severity conditions.
Extracts behavioral features and computes onsets, preserving all Phase 1 files.
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
from rhd.detectors.ground_truth import get_run_ground_truth_onset


def _worker_run(args):
    run_dir, weight, seed, timesteps = args
    if not os.path.exists(os.path.join(run_dir, "checkpoints")):
        print(f"Generating run: {os.path.basename(run_dir)} (w={weight}, seed={seed})...", flush=True)
        env_cfg = RegionToggleConfig(region_reward=weight, max_steps=100)
        ppo_cfg = PPOConfig(
            seed=seed,
            total_timesteps=timesteps,
            checkpoint_interval=5000,
            eval_episodes=10,
        )
        run_generator(run_dir, env_cfg, ppo_cfg)

    feat_dir = os.path.join(run_dir, "features")
    temp_path = os.path.join(feat_dir, "temporal_features.jsonl")
    if not os.path.exists(temp_path):
        print(f"Extracting features for {os.path.basename(run_dir)}...", flush=True)
        extractor = FeatureExtractor(run_dir)
        extractor.run_extraction()

    # Heuristic onset labeling (for comparison baselines O1 and O2)
    onset_path = os.path.join(feat_dir, "onset.json")
    if not os.path.exists(onset_path):
        process_run(run_dir)

    return run_dir


def generate_validation_runs(
    output_dir: str = "experiments/validation_dataset",
    metadata_dir: str = "results/validation_phase2",
    seeds: list = None,
    weights: list = None,
    timesteps: int = 30000,
    max_workers: int = 6,
):
    if seeds is None:
        seeds = [501, 502, 503, 504, 505, 506]
    if weights is None:
        weights = [0.0, 1.0, 5.0]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    tasks = []
    run_lookup = {}
    for w in weights:
        for s in seeds:
            r_dir = os.path.join(output_dir, f"run_w{w}_s{s}")
            run_lookup[(w, s)] = r_dir
            tasks.append((r_dir, w, s, timesteps))

    print(f"Generating Phase 2 dataset: {len(tasks)} runs across weights {weights} and seeds {seeds}...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker_run, t) for t in tasks]
        for f in as_completed(futures):
            res_dir = f.result()
            print(f"  [DONE] {os.path.basename(res_dir)}", flush=True)

    # Save metadata
    runs_list = [run_lookup[(w, s)] for w in weights for s in seeds]
    ground_truths = {}
    for r in runs_list:
        rid = os.path.basename(r)
        gt = get_run_ground_truth_onset(r)
        ground_truths[rid] = gt

    metadata = {
        "dataset_name": "Phase 2 Validation Corpus (Unseen Seeds)",
        "total_runs": len(runs_list),
        "seeds": seeds,
        "weights": weights,
        "runs": [os.path.basename(r) for r in runs_list],
        "run_paths": {os.path.basename(r): r for r in runs_list},
        "independent_ground_truth_onsets": ground_truths,
    }

    meta_path = os.path.join(metadata_dir, "validation_dataset_metadata.json")
    with open(meta_path, "w") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"\n[OK] Validation dataset metadata saved to {meta_path}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="experiments/validation_dataset")
    parser.add_argument("--metadata-dir", type=str, default="results/validation_phase2")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    generate_validation_runs(
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        max_workers=args.workers,
    )
