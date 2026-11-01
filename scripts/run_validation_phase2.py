"""
run_validation_phase2.py

Comprehensive Phase 2 Experimental Validation & Ablation Study:
1. Evaluates on 18 independent generator runs across 6 completely unseen random seeds.
2. Uses task-grounded independent ground-truth onsets (NOT derived from O1/O2).
3. Ablation study: Compares LSTM-DQN vs MLP-DQN (non-recurrent baseline) vs Mathematical O1 vs Mathematical O2.
4. Generates run-level checkpoint predictions, confusion matrices, per-condition metrics, and mean +/- std statistics.
5. Preserves all original Phase 1 results in results/detector/ intact.
"""

import os
import sys
import json
import csv
import argparse
from typing import List, Dict, Any, Optional
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rhd.detectors.environment import (
    load_episode_from_run,
    Normalizer,
    DetectorRewardConfig,
    DetectorEpisode,
    DetectorEnv,
)
from rhd.detectors.agent import DQNAgent, DQNConfig
from rhd.detectors.trainer import DetectorTrainer, TrainerConfig
from rhd.detectors.ground_truth import get_run_ground_truth_onset


def evaluate_detector_on_episodes(
    agent: DQNAgent,
    episodes: List[DetectorEpisode],
    normalizer: Normalizer,
    window_size: int = 4,
) -> List[Dict[str, Any]]:
    """
    Evaluates a DQN agent (LSTM or MLP) deterministically on a list of episodes.
    """
    env = DetectorEnv(
        episodes=episodes,
        window_size=window_size,
        normalizer=normalizer,
    )
    records = []

    for i, ep in enumerate(episodes):
        state, info = env.reset(episode_idx=i)
        done = False
        flagged = False
        flag_step = None
        flag_idx = None

        while not done:
            action = agent.select_action(state, deterministic=True)
            step_idx = env.current_step
            env_step = ep.checkpoint_steps[step_idx]

            if action == DetectorEnv.ACTION_FLAG:
                flagged = True
                flag_step = env_step
                flag_idx = step_idx

            state, reward, done, step_info = env.step(action)

        t_star = ep.ground_truth_onset_idx
        t_star_step = ep.ground_truth_onset_step

        if ep.is_clean:
            classification = "FP" if flagged else "TN"
        else:
            if not flagged:
                classification = "FN"
            else:
                if t_star is not None and flag_idx < t_star:
                    classification = "FP_early"  # Premature false alarm
                else:
                    classification = "TP"

        delay_ckpts = None
        delay_steps = None
        if classification == "TP" and t_star is not None and flag_idx is not None:
            delay_ckpts = flag_idx - t_star
            delay_steps = ep.checkpoint_steps[flag_idx] - ep.checkpoint_steps[t_star]

        records.append({
            "run_id": ep.run_id,
            "is_clean": ep.is_clean,
            "true_onset_step": t_star_step,
            "true_onset_idx": t_star,
            "total_checkpoints": ep.total_checkpoints,
            "flagged": flagged,
            "predicted_step": flag_step,
            "predicted_idx": flag_idx,
            "classification": classification,
            "delay_ckpts": delay_ckpts,
            "delay_steps": delay_steps,
        })

    return records


def evaluate_mathematical_on_episodes(
    episodes: List[DetectorEpisode],
    run_paths: Dict[str, str],
    detector_key: str = "o2",
) -> List[Dict[str, Any]]:
    """
    Evaluates mathematical baseline detectors (O1 or O2) against independent ground truth.
    """
    records = []
    for ep in episodes:
        rpath = run_paths.get(ep.run_id)
        onset_file = os.path.join(rpath, "features", "onset.json") if rpath else None

        flagged = False
        flag_step = None
        flag_idx = None

        if onset_file and os.path.exists(onset_file):
            with open(onset_file) as fp:
                data = json.load(fp)
            if data.get(f"{detector_key}_detected"):
                s = data.get(f"{detector_key}_onset_step")
                if s is not None:
                    flagged = True
                    flag_step = int(s)
                    for idx, st in enumerate(ep.checkpoint_steps):
                        if st >= flag_step:
                            flag_idx = idx
                            break

        t_star = ep.ground_truth_onset_idx
        t_star_step = ep.ground_truth_onset_step

        if ep.is_clean:
            classification = "FP" if flagged else "TN"
        else:
            if not flagged:
                classification = "FN"
            else:
                if t_star is not None and flag_idx is not None and flag_idx < t_star:
                    classification = "FP_early"
                else:
                    classification = "TP"

        delay_ckpts = None
        delay_steps = None
        if classification == "TP" and t_star is not None and flag_idx is not None:
            delay_ckpts = flag_idx - t_star
            delay_steps = ep.checkpoint_steps[flag_idx] - ep.checkpoint_steps[t_star]

        records.append({
            "run_id": ep.run_id,
            "is_clean": ep.is_clean,
            "true_onset_step": t_star_step,
            "true_onset_idx": t_star,
            "total_checkpoints": ep.total_checkpoints,
            "flagged": flagged,
            "predicted_step": flag_step,
            "predicted_idx": flag_idx,
            "classification": classification,
            "delay_ckpts": delay_ckpts,
            "delay_steps": delay_steps,
        })

    return records


def compute_metrics_and_confusion(predictions: List[Dict[str, Any]], method_name: str) -> Dict[str, Any]:
    total = len(predictions)
    clean_preds = [p for p in predictions if p["is_clean"]]
    hack_preds = [p for p in predictions if not p["is_clean"]]

    tp = sum(1 for p in hack_preds if p["classification"] == "TP")
    fn = sum(1 for p in hack_preds if p["classification"] == "FN")
    fp_early = sum(1 for p in hack_preds if p["classification"] == "FP_early")
    tn = sum(1 for p in clean_preds if p["classification"] == "TN")
    fp_clean = sum(1 for p in clean_preds if p["classification"] == "FP")

    total_fp = fp_early + fp_clean
    precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    clean_fp_rate = fp_clean / len(clean_preds) if clean_preds else 0.0
    delays_c = [p["delay_ckpts"] for p in predictions if p["delay_ckpts"] is not None]
    delays_s = [p["delay_steps"] for p in predictions if p["delay_steps"] is not None]

    mean_delay_c = float(np.mean(delays_c)) if delays_c else 0.0
    std_delay_c = float(np.std(delays_c)) if len(delays_c) > 1 else 0.0
    mean_delay_s = float(np.mean(delays_s)) if delays_s else 0.0
    std_delay_s = float(np.std(delays_s)) if len(delays_s) > 1 else 0.0

    # Per-condition breakdown
    # Extract condition w from run_id (e.g. run_w1.0_s501 -> w=1.0)
    cond_metrics = {}
    for cond_tag in ["w0.0", "w1.0", "w5.0"]:
        cond_p = [p for p in predictions if cond_tag in p["run_id"]]
        if cond_tag == "w0.0":
            c_fp = sum(1 for p in cond_p if p["flagged"])
            c_tn = len(cond_p) - c_fp
            cond_metrics[cond_tag] = {
                "condition": "Clean Control (w=0.0)",
                "total": len(cond_p),
                "true_negatives": c_tn,
                "false_positives": c_fp,
                "clean_fpr": c_fp / len(cond_p) if cond_p else 0.0,
            }
        else:
            c_tp = sum(1 for p in cond_p if p["classification"] == "TP")
            c_fn = sum(1 for p in cond_p if p["classification"] == "FN")
            c_early = sum(1 for p in cond_p if p["classification"] == "FP_early")
            c_delays = [p["delay_ckpts"] for p in cond_p if p["delay_ckpts"] is not None]
            cond_metrics[cond_tag] = {
                "condition": f"Exploit Severity ({cond_tag})",
                "total": len(cond_p),
                "true_positives": c_tp,
                "false_negatives": c_fn,
                "premature_false_positives": c_early,
                "detection_rate": c_tp / len(cond_p) if cond_p else 0.0,
                "mean_delay_ckpts": float(np.mean(c_delays)) if c_delays else 0.0,
            }

    return {
        "method": method_name,
        "total_runs": total,
        "clean_runs": len(clean_preds),
        "hacking_runs": len(hack_preds),
        "confusion_matrix": {
            "TP": tp,
            "FP_clean": fp_clean,
            "FP_early": fp_early,
            "TN": tn,
            "FN": fn,
        },
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "clean_fp_rate": clean_fp_rate,
        "clean_flagged_pct": clean_fp_rate * 100.0,
        "hacking_detected_pct": (tp / len(hack_preds) * 100.0) if hack_preds else 0.0,
        "mean_delay_ckpts": mean_delay_c,
        "std_delay_ckpts": std_delay_c,
        "mean_delay_steps": mean_delay_s,
        "std_delay_steps": std_delay_s,
        "per_condition": cond_metrics,
        "predictions": predictions,
    }


def run_phase2_validation(
    val_meta_path: str = "results/validation_phase2/validation_dataset_metadata.json",
    train_meta_path: str = "results/detector/split_metadata.json",
    norm_path: str = "results/detector/norm_stats.json",
    output_dir: str = "results/validation_phase2",
):
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(val_meta_path), f"Phase 2 validation dataset metadata not found at {val_meta_path}"
    assert os.path.exists(train_meta_path), f"Phase 1 train split metadata not found at {train_meta_path}"
    assert os.path.exists(norm_path), f"Normalization stats not found at {norm_path}"

    with open(val_meta_path) as fp:
        val_meta = json.load(fp)
    with open(train_meta_path) as fp:
        train_meta = json.load(fp)

    val_paths = val_meta["run_paths"]
    train_paths = train_meta["run_paths"]
    train_ids = train_meta["train"]
    val_ids = train_meta["validation"]

    normalizer = Normalizer.load(norm_path)

    print("=" * 60)
    print("  PHASE 2 EXPERIMENTAL VALIDATION & ABLATION STUDY")
    print("=" * 60)
    print(f"Validation Corpus: {len(val_paths)} unseen generator runs across seeds {val_meta['seeds']}")

    # Load train episodes
    train_episodes = [load_episode_from_run(train_paths[rid], use_independent_ground_truth=True) for rid in train_ids]
    dev_val_episodes = [load_episode_from_run(train_paths[rid], use_independent_ground_truth=True) for rid in val_ids]
    train_episodes = [ep for ep in train_episodes if ep is not None]
    dev_val_episodes = [ep for ep in dev_val_episodes if ep is not None]

    # Load 18 unseen validation episodes
    test_episodes = [load_episode_from_run(p, use_independent_ground_truth=True) for p in val_paths.values()]
    test_episodes = [ep for ep in test_episodes if ep is not None]
    assert len(test_episodes) == len(val_paths), "Failed to load all validation episodes!"

    # 1. Train / Load LSTM-DQN
    lstm_ckpt = os.path.join(output_dir, "lstm_dqn.pt")
    lstm_agent = DQNAgent(config=DQNConfig(model_type="lstm"))
    if not os.path.exists(lstm_ckpt):
        print("\n--- Training LSTM-DQN Detector for Phase 2 ---")
        trainer_lstm = DetectorTrainer(
            train_episodes=train_episodes,
            val_episodes=dev_val_episodes,
            trainer_config=TrainerConfig(num_episodes=350, save_dir=output_dir),
            dqn_config=DQNConfig(model_type="lstm"),
            reward_config=DetectorRewardConfig(),
            normalizer=normalizer,
        )
        trainer_lstm.train()
        lstm_agent.load(os.path.join(output_dir, "detector_best.pt"))
        lstm_agent.save(lstm_ckpt)
    else:
        print(f"\n[OK] Loaded existing LSTM-DQN checkpoint from {lstm_ckpt}")
        lstm_agent.load(lstm_ckpt)

    # 2. Train / Load Ablation: MLP-DQN (Non-Recurrent Feed-Forward Q-Network)
    mlp_ckpt = os.path.join(output_dir, "mlp_dqn.pt")
    mlp_agent = DQNAgent(config=DQNConfig(model_type="mlp"))
    if not os.path.exists(mlp_ckpt):
        print("\n--- Training Ablation: Non-Recurrent MLP-DQN Detector ---")
        trainer_mlp = DetectorTrainer(
            train_episodes=train_episodes,
            val_episodes=dev_val_episodes,
            trainer_config=TrainerConfig(num_episodes=350, save_dir=output_dir),
            dqn_config=DQNConfig(model_type="mlp"),
            reward_config=DetectorRewardConfig(),
            normalizer=normalizer,
        )
        trainer_mlp.train()
        mlp_agent.load(os.path.join(output_dir, "detector_best.pt"))
        mlp_agent.save(mlp_ckpt)
    else:
        print(f"\n[OK] Loaded existing MLP-DQN checkpoint from {mlp_ckpt}")
        mlp_agent.load(mlp_ckpt)

    # 3. Evaluate all 4 methods on 18 unseen validation runs
    print("\nEvaluating all methods on 18 unseen runs against task-grounded ground truth...")
    preds_lstm = evaluate_detector_on_episodes(lstm_agent, test_episodes, normalizer=normalizer)
    preds_mlp = evaluate_detector_on_episodes(mlp_agent, test_episodes, normalizer=normalizer)
    preds_o1 = evaluate_mathematical_on_episodes(test_episodes, val_paths, detector_key="o1")
    preds_o2 = evaluate_mathematical_on_episodes(test_episodes, val_paths, detector_key="o2")

    res_lstm = compute_metrics_and_confusion(preds_lstm, "LSTM-DQN (Sequential)")
    res_mlp = compute_metrics_and_confusion(preds_mlp, "MLP-DQN (Ablation: Non-Recurrent)")
    res_o1 = compute_metrics_and_confusion(preds_o1, "Mathematical-O1")
    res_o2 = compute_metrics_and_confusion(preds_o2, "Mathematical-O2")

    methods_results = [res_lstm, res_mlp, res_o1, res_o2]

    # Print summary table
    fmt = "{:<32} | {:>4} | {:>5} | {:>5} | {:>5} | {:>12} | {:>10} | {:>18}"
    sep = "=" * 105
    print("\n" + sep)
    print(fmt.format("Method", "Runs", "P", "R", "F1", "Hacking Det%", "Clean FP%", "Delay (ckpts ± std)"))
    print(sep)
    for r in methods_results:
        delay_str = f"{r['mean_delay_ckpts']:.2f} ± {r['std_delay_ckpts']:.2f}"
        print(fmt.format(
            r["method"],
            r["total_runs"],
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1_score']:.3f}",
            f"{r['hacking_detected_pct']:.1f}%",
            f"{r['clean_flagged_pct']:.1f}%",
            delay_str,
        ))
    print(sep + "\n")

    # Save summary CSV
    summary_csv = os.path.join(output_dir, "comparative_metrics_extended.csv")
    csv_cols = [
        "method", "total_runs", "clean_runs", "hacking_runs",
        "tp", "fp_clean", "fp_early", "tn", "fn",
        "precision", "recall", "f1_score",
        "clean_fp_rate", "clean_flagged_pct", "hacking_detected_pct",
        "mean_delay_ckpts", "std_delay_ckpts",
        "mean_delay_steps", "std_delay_steps",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=csv_cols)
        writer.writeheader()
        for r in methods_results:
            cm = r["confusion_matrix"]
            writer.writerow({
                "method": r["method"],
                "total_runs": r["total_runs"],
                "clean_runs": r["clean_runs"],
                "hacking_runs": r["hacking_runs"],
                "tp": cm["TP"],
                "fp_clean": cm["FP_clean"],
                "fp_early": cm["FP_early"],
                "tn": cm["TN"],
                "fn": cm["FN"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1_score": r["f1_score"],
                "clean_fp_rate": r["clean_fp_rate"],
                "clean_flagged_pct": r["clean_flagged_pct"],
                "hacking_detected_pct": r["hacking_detected_pct"],
                "mean_delay_ckpts": r["mean_delay_ckpts"],
                "std_delay_ckpts": r["std_delay_ckpts"],
                "mean_delay_steps": r["mean_delay_steps"],
                "std_delay_steps": r["std_delay_steps"],
            })
    print(f"[OK] Summary metrics saved to {summary_csv}")

    # Save exact run-level checkpoint predictions CSV
    preds_csv = os.path.join(output_dir, "run_level_predictions.csv")
    p_cols = [
        "run_id", "condition", "seed", "total_checkpoints", "true_onset_step", "true_onset_idx",
        "lstm_flagged", "lstm_pred_step", "lstm_delay_ckpts", "lstm_class",
        "mlp_flagged", "mlp_pred_step", "mlp_delay_ckpts", "mlp_class",
        "o1_flagged", "o1_pred_step", "o1_delay_ckpts", "o1_class",
        "o2_flagged", "o2_pred_step", "o2_delay_ckpts", "o2_class",
    ]

    p_lstm_map = {p["run_id"]: p for p in preds_lstm}
    p_mlp_map = {p["run_id"]: p for p in preds_mlp}
    p_o1_map = {p["run_id"]: p for p in preds_o1}
    p_o2_map = {p["run_id"]: p for p in preds_o2}

    with open(preds_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=p_cols)
        writer.writeheader()
        for ep in test_episodes:
            rid = ep.run_id
            parts = rid.split("_")
            cond = parts[1] if len(parts) > 1 else "unknown"
            s_str = parts[2] if len(parts) > 2 else "unknown"

            pl = p_lstm_map.get(rid, {})
            pm = p_mlp_map.get(rid, {})
            po1 = p_o1_map.get(rid, {})
            po2 = p_o2_map.get(rid, {})

            writer.writerow({
                "run_id": rid,
                "condition": cond,
                "seed": s_str,
                "total_checkpoints": ep.total_checkpoints,
                "true_onset_step": ep.ground_truth_onset_step,
                "true_onset_idx": ep.ground_truth_onset_idx,
                "lstm_flagged": pl.get("flagged"),
                "lstm_pred_step": pl.get("predicted_step"),
                "lstm_delay_ckpts": pl.get("delay_ckpts"),
                "lstm_class": pl.get("classification"),
                "mlp_flagged": pm.get("flagged"),
                "mlp_pred_step": pm.get("predicted_step"),
                "mlp_delay_ckpts": pm.get("delay_ckpts"),
                "mlp_class": pm.get("classification"),
                "o1_flagged": po1.get("flagged"),
                "o1_pred_step": po1.get("predicted_step"),
                "o1_delay_ckpts": po1.get("delay_ckpts"),
                "o1_class": po1.get("classification"),
                "o2_flagged": po2.get("flagged"),
                "o2_pred_step": po2.get("predicted_step"),
                "o2_delay_ckpts": po2.get("delay_ckpts"),
                "o2_class": po2.get("classification"),
            })
    print(f"[OK] Run-level predictions saved to {preds_csv}")

    # Save complete JSON report
    report_json = os.path.join(output_dir, "phase2_validation_report.json")
    with open(report_json, "w") as fp:
        json.dump({r["method"]: r for r in methods_results}, fp, indent=2)
    print(f"[OK] Full JSON report saved to {report_json}")

    return methods_results


if __name__ == "__main__":
    run_phase2_validation()
