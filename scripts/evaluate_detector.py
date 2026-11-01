"""
evaluate_detector.py

Evaluates the trained LSTM-DQN detector against mathematical O1 and O2 detectors
on held-out test runs. Generates comparison tables and metrics CSV files.
"""

import os
import sys
import json
import csv
import argparse
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rhd.detectors.environment import load_episode_from_run, Normalizer
from rhd.detectors.agent import DQNAgent
from rhd.detectors.evaluator import DetectorEvaluator, EvaluationResult


def print_comparison_table(results: List[EvaluationResult]):
    headers = [
        "Detector",
        "Runs",
        "P",
        "R",
        "F1",
        "Hacking Det%",
        "Clean FP%",
        "Delay (ckpts)",
        "Delay (steps)",
    ]
    fmt = "{:<16} | {:>4} | {:>5} | {:>5} | {:>5} | {:>12} | {:>10} | {:>13} | {:>13}"
    sep = "-" * 98

    print("\n" + sep)
    print(fmt.format(*headers))
    print(sep)

    for r in results:
        print(fmt.format(
            r.detector_name,
            f"{r.total_runs}",
            f"{r.precision:.3f}",
            f"{r.recall:.3f}",
            f"{r.f1_score:.3f}",
            f"{r.hacking_detected_pct:.1f}%",
            f"{r.clean_flagged_pct:.1f}%",
            f"{r.mean_delay_ckpts:.2f}",
            f"{r.mean_delay_steps:.0f}",
        ))
    print(sep + "\n")


def evaluate(
    metadata_dir: str = "results/detector",
    model_path: str = "results/detector/detector_best.pt",
    output_dir: str = "results/detector",
    split_to_eval: str = "test",  # "test" or "all"
):
    os.makedirs(output_dir, exist_ok=True)
    split_path = os.path.join(metadata_dir, "split_metadata.json")
    norm_path = os.path.join(metadata_dir, "norm_stats.json")

    assert os.path.exists(split_path), f"Split metadata not found at {split_path}"
    assert os.path.exists(norm_path), f"Normalizer stats not found at {norm_path}"
    assert os.path.exists(model_path), f"Trained detector model not found at {model_path}"

    with open(split_path, "r") as f:
        split_meta = json.load(f)

    run_paths = split_meta["run_paths"]
    if split_to_eval == "test":
        target_ids = split_meta["test"]
        print(f"Evaluating exclusively on held-out TEST partition ({len(target_ids)} runs): {target_ids}")
    elif split_to_eval == "validation":
        target_ids = split_meta["validation"]
    else:
        target_ids = split_meta["train"] + split_meta["validation"] + split_meta["test"]
        print(f"Evaluating across ALL partitions ({len(target_ids)} runs)")

    target_episodes = [load_episode_from_run(run_paths[rid]) for rid in target_ids]
    target_episodes = [ep for ep in target_episodes if ep is not None]

    normalizer = Normalizer.load(norm_path)
    evaluator = DetectorEvaluator(window_size=4, normalizer=normalizer)

    agent = DQNAgent()
    agent.load(model_path)

    # 1. Evaluate LSTM-DQN
    dqn_res = evaluator.evaluate_dqn(agent, target_episodes, detector_name="LSTM-DQN")

    # 2. Evaluate Mathematical O1 Baseline
    o1_res = evaluator.evaluate_mathematical_detector(
        target_episodes,
        run_paths,
        detector_key="o1",
        detector_name="Mathematical-O1",
    )

    # 3. Evaluate Mathematical O2 Baseline
    o2_res = evaluator.evaluate_mathematical_detector(
        target_episodes,
        run_paths,
        detector_key="o2",
        detector_name="Mathematical-O2",
    )

    # 4. Evaluate Primary Combined Baseline
    prim_res = evaluator.evaluate_mathematical_detector(
        target_episodes,
        run_paths,
        detector_key="primary",
        detector_name="Mathematical-O1&O2",
    )

    all_results = [dqn_res, o1_res, o2_res, prim_res]
    print_comparison_table(all_results)

    # Save detailed predictions
    preds_out = os.path.join(output_dir, f"predictions_{split_to_eval}.json")
    with open(preds_out, "w") as f:
        json.dump({r.detector_name: r.to_dict() for r in all_results}, f, indent=2)
    print(f"[OK] Predictions saved to: {preds_out}")

    # Save summary CSV
    csv_path = os.path.join(output_dir, f"comparison_metrics_{split_to_eval}.csv")
    csv_cols = [
        "detector_name",
        "split",
        "total_runs",
        "clean_runs",
        "hacking_runs",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1_score",
        "clean_fp_rate",
        "clean_flagged_pct",
        "hacking_detected_pct",
        "mean_delay_ckpts",
        "mean_delay_steps",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in all_results:
            d = r.to_dict()
            d.pop("predictions")
            d["split"] = split_to_eval
            writer.writerow(d)
    print(f"[OK] Metrics table saved to: {csv_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM-DQN detector vs mathematical baselines.")
    parser.add_argument("--metadata-dir", type=str, default="results/detector")
    parser.add_argument("--model-path", type=str, default="results/detector/detector_best.pt")
    parser.add_argument("--output-dir", type=str, default="results/detector")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation", "all"])
    args = parser.parse_args()

    evaluate(
        metadata_dir=args.metadata_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        split_to_eval=args.split,
    )
