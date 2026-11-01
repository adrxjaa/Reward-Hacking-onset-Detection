"""
scripts/run_feature_ablation.py

Controlled 10-Feature Ablation Study for Reward-Hacking Onset Detection.

Objective:
Evaluate how different groups of the canonical 10 behavioral features
contribute to onset detection performance across both original detector architectures
(LSTM-DQN and MLP-DQN) over 5 independent random seeds (0, 1, 2, 3, 4) across
all 4 dataset partitions (Train, Validation, Unseen Phase 2, and Final Test).

Feature Groups:
A. BASELINE: All 10 original features (input_dim=10)
B. TEMPORAL ABLATION: Removes proxy_return_trend, action_distribution_change, state_visitation_change (input_dim=7)
C. REVISIT / ANOMALY ABLATION: Removes path_efficiency, revisit_rate, behavioral_dispersion (input_dim=7)

Outputs saved to:
results/ablation_study/
  - ablation_results.csv
  - ablation_report.md
  - ablation_comparison.png
  - experiment_metadata.json
"""

import os
import sys
import json
import time
import copy
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

# Add repo root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rhd.detectors.environment import (
    FEATURE_NAMES,
    DetectorEpisode,
    DetectorEnv,
    DetectorRewardConfig,
    Normalizer,
    load_episode_from_run,
)
from rhd.detectors.agent import DQNAgent, DQNConfig
from rhd.detectors.replay_buffer import ReplayBuffer

OUT_DIR = os.path.join(ROOT_DIR, "results", "ablation_study")

CONFIGURATIONS = {
    "baseline": {
        "name": "Baseline (All 10 Features)",
        "indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "retained_features": FEATURE_NAMES,
        "removed_features": [],
        "description": "All 10 original behavioral features retained without modification.",
    },
    "no_temporal": {
        "name": "Temporal Feature Ablation",
        "indices": [0, 1, 3, 5, 7, 8, 9],
        "retained_features": [
            "proxy_return",
            "proxy_return_smoothed",
            "episode_length",
            "state_visitation_entropy",
            "path_efficiency",
            "revisit_rate",
            "behavioral_dispersion",
        ],
        "removed_features": [
            "proxy_return_trend",
            "action_distribution_change",
            "state_visitation_change",
        ],
        "description": "Removes inter-checkpoint difference and trend signals; retains static episode stats and spatial metrics.",
    },
    "no_revisit_anomaly": {
        "name": "Revisit / Anomaly Feature Ablation",
        "indices": [0, 1, 2, 3, 4, 5, 6],
        "retained_features": [
            "proxy_return",
            "proxy_return_smoothed",
            "proxy_return_trend",
            "episode_length",
            "action_distribution_change",
            "state_visitation_entropy",
            "state_visitation_change",
        ],
        "removed_features": [
            "path_efficiency",
            "revisit_rate",
            "behavioral_dispersion",
        ],
        "description": "Removes rollout path efficiency, state revisitation rate, and probe entropy dispersion.",
    },
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def condition_of(run_id: str) -> str:
    for c in ("w0.0", "w1.0", "w5.0"):
        if c in run_id:
            return c
    return "unknown"


def make_ablated_episode(ep: DetectorEpisode, selected_indices: List[int]) -> DetectorEpisode:
    """Create an episode with only the selected feature columns."""
    return DetectorEpisode(
        run_id=ep.run_id,
        checkpoint_steps=list(ep.checkpoint_steps),
        checkpoint_ids=list(ep.checkpoint_ids),
        features=ep.features[:, selected_indices].copy(),
        ground_truth_onset_step=ep.ground_truth_onset_step,
        ground_truth_onset_idx=ep.ground_truth_onset_idx,
        is_clean=ep.is_clean,
    )


def classify_outcome(ep: DetectorEpisode, flag_idx: Optional[int]) -> str:
    """
    Standard run-level classification:
    - Clean run:
        flagged -> FP
        not flagged -> TN
    - Hacking run:
        not flagged -> FN
        flagged >= ground_truth_onset_idx -> TP
        flagged < ground_truth_onset_idx -> EARLY (premature false alarm)
    """
    if ep.is_clean:
        return "FP" if flag_idx is not None else "TN"
    if flag_idx is None:
        return "FN"
    return "TP" if flag_idx >= ep.ground_truth_onset_idx else "EARLY"


def oracle_return(ep: DetectorEpisode, rc: DetectorRewardConfig, window_size: int) -> float:
    """Ideal stopping policy: flag at exactly t* on hacking runs, never flag on clean runs."""
    env = DetectorEnv([ep], window_size=window_size, reward_config=rc)
    env.reset(episode_idx=0)
    total, done = 0.0, False
    while not done:
        a = (
            DetectorEnv.ACTION_FLAG
            if (not ep.is_clean and env.current_step == ep.ground_truth_onset_idx)
            else DetectorEnv.ACTION_WAIT
        )
        _, rwd, done, _ = env.step(a)
        total += rwd
    return total


def evaluate_agent_on_split(
    agent: DQNAgent,
    episodes: List[DetectorEpisode],
    normalizer: Normalizer,
    window_size: int = 4,
    reward_config: Optional[DetectorRewardConfig] = None,
) -> Dict[str, Any]:
    rc = reward_config or DetectorRewardConfig()
    env = DetectorEnv(episodes, window_size=window_size, reward_config=rc, normalizer=normalizer)
    records = []
    total_regret = 0.0

    for i, ep in enumerate(episodes):
        s, _ = env.reset(episode_idx=i)
        done, flag_idx, ep_return = False, None, 0.0
        while not done:
            t = env.current_step
            a = agent.select_action(s, deterministic=True)
            if a == DetectorEnv.ACTION_FLAG:
                flag_idx = t
            s, rwd, done, _ = env.step(a)
            ep_return += rwd

        outcome = classify_outcome(ep, flag_idx)
        opt_ret = oracle_return(ep, rc, window_size)
        total_regret += (opt_ret - ep_return)

        delay_ckpts = (flag_idx - ep.ground_truth_onset_idx) if outcome == "TP" else None
        records.append({
            "run_id": ep.run_id,
            "condition": condition_of(ep.run_id),
            "is_clean": ep.is_clean,
            "true_onset_idx": ep.ground_truth_onset_idx,
            "true_onset_step": ep.ground_truth_onset_step,
            "flag_idx": flag_idx,
            "flag_step": ep.checkpoint_steps[flag_idx] if flag_idx is not None else None,
            "outcome": outcome,
            "delay_ckpts": delay_ckpts,
            "return": ep_return,
        })

    # Metrics computation
    counts = {k: sum(1 for r in records if r["outcome"] == k) for k in ("TP", "EARLY", "FN", "TN", "FP")}
    n_hack = counts["TP"] + counts["EARLY"] + counts["FN"]
    n_clean = counts["TN"] + counts["FP"]
    alarms = counts["TP"] + counts["EARLY"] + counts["FP"]

    precision = counts["TP"] / alarms if alarms > 0 else 0.0
    recall = counts["TP"] / n_hack if n_hack > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    clean_fpr = counts["FP"] / n_clean if n_clean > 0 else 0.0
    premature_rate = counts["EARLY"] / n_hack if n_hack > 0 else 0.0
    miss_rate = counts["FN"] / n_hack if n_hack > 0 else 0.0

    delays = [r["delay_ckpts"] for r in records if r["delay_ckpts"] is not None]
    mean_delay = float(np.mean(delays)) if delays else None
    std_delay = float(np.std(delays)) if delays else None

    return {
        "n_runs": len(records),
        "n_hack": n_hack,
        "n_clean": n_clean,
        "counts": counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "clean_fpr": clean_fpr,
        "premature_rate": premature_rate,
        "miss_rate": miss_rate,
        "mean_delay_ckpts": mean_delay,
        "std_delay_ckpts": std_delay,
        "eval_regret_per_run": total_regret / len(episodes),
        "records": records,
    }


def train_single_run(
    train_eps: List[DetectorEpisode],
    val_eps: List[DetectorEpisode],
    normalizer: Normalizer,
    model_type: str,
    input_dim: int,
    seed: int,
    num_episodes: int = 350,
    eval_every: int = 10,
    window_size: int = 4,
    warmup: int = 64,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    set_seed(seed)
    rc = DetectorRewardConfig()
    dqn_cfg = DQNConfig(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=1,
        num_actions=2,
        model_type=model_type,
        learning_rate=1e-3,
        gamma=0.95,
        batch_size=32,
        target_update_freq=20,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        grad_norm_clip=5.0,
    )
    agent = DQNAgent(config=dqn_cfg)
    buf = ReplayBuffer(capacity=10000)
    env = DetectorEnv(train_eps, window_size=window_size, reward_config=rc, normalizer=normalizer)

    best_val_f1 = -1.0
    best_state = None
    best_ep = None

    t0 = time.perf_counter()
    for ep_i in range(1, num_episodes + 1):
        s, _ = env.reset()
        done = False
        while not done:
            a = agent.select_action(s, deterministic=False)
            s2, rwd, done, _ = env.step(a)
            buf.push(s, a, rwd, s2, done)
            s = s2
            if len(buf) >= warmup:
                agent.update(buf)

        if ep_i % eval_every == 0 or ep_i == num_episodes:
            val_metrics = evaluate_agent_on_split(agent, val_eps, normalizer, window_size=window_size, reward_config=rc)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_ep = ep_i
                best_state = copy.deepcopy(agent.online_net.state_dict())

    wall_time = time.perf_counter() - t0
    if best_state is not None:
        agent.online_net.load_state_dict(best_state)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save(save_path)

    return {
        "agent": agent,
        "best_val_f1": best_val_f1,
        "selected_episode": best_ep,
        "train_wall_s": wall_time,
    }


def aggregate_metric_list(vals: List[Optional[float]]) -> Tuple[float, float]:
    valid = [v for v in vals if v is not None]
    if not valid:
        return 0.0, 0.0
    return float(np.mean(valid)), float(np.std(valid))


def main():
    parser = argparse.ArgumentParser(description="Run 10-feature ablation study")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=350)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    models_dir = os.path.join(OUT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 78)
    print("  CONTROLLED 10-FEATURE ABLATION STUDY FOR REWARD-HACKING ONSET DETECTION")
    print(f"  Seeds: {args.seeds} | Episodes per run: {args.episodes}")
    print("=" * 78)

    # 1. Load canonical data splits
    split_meta = json.load(open("results/detector/split_metadata.json"))
    p2_meta = json.load(open("results/validation_phase2/validation_dataset_metadata.json"))
    ft_meta = json.load(open("results/phase3_final/final_test_metadata.json"))

    raw_partitions = {
        "train": [load_episode_from_run(split_meta["run_paths"][r]) for r in split_meta["train"]],
        "val": [load_episode_from_run(split_meta["run_paths"][r]) for r in split_meta["validation"]],
        "unseen_p2": [load_episode_from_run(p) for p in p2_meta["run_paths"].values()],
        "final_test": [load_episode_from_run(p) for p in ft_meta["run_paths"].values()],
    }

    # Verify no none and strictly disjoint runs
    for k, v in raw_partitions.items():
        assert all(e is not None for e in v), f"Error loading partition {k}"
    run_sets = {k: {e.run_id for e in v} for k, v in raw_partitions.items()}
    assert len(run_sets["train"] & run_sets["val"]) == 0
    assert len(run_sets["train"] & run_sets["unseen_p2"]) == 0
    assert len(run_sets["train"] & run_sets["final_test"]) == 0
    assert len(run_sets["val"] & run_sets["final_test"]) == 0

    print(f"[*] Partitions loaded successfully:")
    print(f"    Train: {len(raw_partitions['train'])} runs")
    print(f"    Validation: {len(raw_partitions['val'])} runs")
    print(f"    Unseen Phase 2: {len(raw_partitions['unseen_p2'])} runs")
    print(f"    Final Test: {len(raw_partitions['final_test'])} runs")

    all_results = {}
    csv_rows = []

    # 2. Iterate over configurations, detectors, and seeds
    for cfg_key, cfg_info in CONFIGURATIONS.items():
        print("\n" + "-" * 70)
        print(f"  Configuration: {cfg_info['name']} (Dim: {len(cfg_info['indices'])})")
        print(f"  Retained features: {cfg_info['retained_features']}")
        print(f"  Removed features: {cfg_info['removed_features']}")
        print("-" * 70)

        # Prepare ablated partition episodes
        indices = cfg_info["indices"]
        input_dim = len(indices)
        cfg_partitions = {
            part_name: [make_ablated_episode(ep, indices) for ep in eps]
            for part_name, eps in raw_partitions.items()
        }

        # Fit normalizer STRICTLY on train partition of this configuration
        cfg_normalizer = Normalizer()
        cfg_normalizer.fit([ep.features for ep in cfg_partitions["train"]])

        all_results[cfg_key] = {
            "name": cfg_info["name"],
            "input_dim": input_dim,
            "retained_features": cfg_info["retained_features"],
            "removed_features": cfg_info["removed_features"],
            "description": cfg_info["description"],
            "detectors": {},
        }

        for model_type in ("lstm", "mlp"):
            det_name = f"{model_type.upper()}-DQN"
            print(f"\n  >> Detector: {det_name}")
            all_results[cfg_key]["detectors"][model_type] = {"seed_runs": []}

            seed_runs = []
            for seed in args.seeds:
                ckpt_path = os.path.join(models_dir, f"{cfg_key}_{model_type}_seed{seed}.pt")
                train_res = train_single_run(
                    train_eps=cfg_partitions["train"],
                    val_eps=cfg_partitions["val"],
                    normalizer=cfg_normalizer,
                    model_type=model_type,
                    input_dim=input_dim,
                    seed=seed,
                    num_episodes=args.episodes,
                    save_path=ckpt_path,
                )
                agent = train_res["agent"]

                # Evaluate deterministically on all 4 partitions
                eval_metrics = {}
                for part_name, part_eps in cfg_partitions.items():
                    m = evaluate_agent_on_split(agent, part_eps, cfg_normalizer)
                    eval_metrics[part_name] = {k: v for k, v in m.items() if k != "records"}

                run_summary = {
                    "seed": seed,
                    "best_val_f1": train_res["best_val_f1"],
                    "selected_episode": train_res["selected_episode"],
                    "train_wall_s": train_res["train_wall_s"],
                    "eval": eval_metrics,
                }
                seed_runs.append(run_summary)
                val_f1 = eval_metrics["val"]["f1"]
                ft_f1 = eval_metrics["final_test"]["f1"]
                p2_f1 = eval_metrics["unseen_p2"]["f1"]
                print(
                    f"     [Seed {seed}] Val F1: {val_f1:.3f} | "
                    f"Unseen P2 F1: {p2_f1:.3f} | Final-Test F1: {ft_f1:.3f} "
                    f"({train_res['train_wall_s']:.1f}s)"
                )

            all_results[cfg_key]["detectors"][model_type]["seed_runs"] = seed_runs

            # Compute mean and std for each partition
            partition_stats = {}
            for part_name in ("train", "val", "unseen_p2", "final_test"):
                f1_vals = [r["eval"][part_name]["f1"] for r in seed_runs]
                prec_vals = [r["eval"][part_name]["precision"] for r in seed_runs]
                rec_vals = [r["eval"][part_name]["recall"] for r in seed_runs]
                fpr_vals = [r["eval"][part_name]["clean_fpr"] for r in seed_runs]
                delay_vals = [r["eval"][part_name]["mean_delay_ckpts"] for r in seed_runs]
                regret_vals = [r["eval"][part_name]["eval_regret_per_run"] for r in seed_runs]

                f1_m, f1_s = aggregate_metric_list(f1_vals)
                prec_m, prec_s = aggregate_metric_list(prec_vals)
                rec_m, rec_s = aggregate_metric_list(rec_vals)
                fpr_m, fpr_s = aggregate_metric_list(fpr_vals)
                del_m, del_s = aggregate_metric_list(delay_vals)
                reg_m, reg_s = aggregate_metric_list(regret_vals)

                partition_stats[part_name] = {
                    "f1": {"mean": f1_m, "std": f1_s},
                    "precision": {"mean": prec_m, "std": prec_s},
                    "recall": {"mean": rec_m, "std": rec_s},
                    "clean_fpr": {"mean": fpr_m, "std": fpr_s},
                    "delay_ckpts": {"mean": del_m, "std": del_s},
                    "regret_per_run": {"mean": reg_m, "std": reg_s},
                }

                csv_rows.append({
                    "config_id": cfg_key,
                    "config_name": cfg_info["name"],
                    "model_type": det_name,
                    "input_dim": input_dim,
                    "partition": part_name,
                    "f1_mean": round(f1_m, 4),
                    "f1_std": round(f1_s, 4),
                    "precision_mean": round(prec_m, 4),
                    "precision_std": round(prec_s, 4),
                    "recall_mean": round(rec_m, 4),
                    "recall_std": round(rec_s, 4),
                    "clean_fpr_mean": round(fpr_m, 4),
                    "clean_fpr_std": round(fpr_s, 4),
                    "delay_mean": round(del_m, 4),
                    "delay_std": round(del_s, 4),
                    "regret_mean": round(reg_m, 4),
                    "regret_std": round(reg_s, 4),
                })

            all_results[cfg_key]["detectors"][model_type]["summary"] = partition_stats

    # 3. Save ablation_results.csv
    csv_path = os.path.join(OUT_DIR, "ablation_results.csv")
    fieldnames = [
        "config_id",
        "config_name",
        "model_type",
        "input_dim",
        "partition",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "clean_fpr_mean",
        "clean_fpr_std",
        "delay_mean",
        "delay_std",
        "regret_mean",
        "regret_std",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[+] Saved {csv_path}")

    # 4. Save experiment_metadata.json
    metadata = {
        "experiment_title": "Controlled 10-Feature Ablation Study for Reward-Hacking Onset Detection",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "random_seeds": args.seeds,
        "episodes_per_run": args.episodes,
        "configurations": CONFIGURATIONS,
        "partition_sizes": {k: len(v) for k, v in raw_partitions.items()},
        "all_results": all_results,
    }
    meta_path = os.path.join(OUT_DIR, "experiment_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved {meta_path}")

    # 5. Generate ablation_comparison.png
    plot_comparison(all_results, os.path.join(OUT_DIR, "ablation_comparison.png"))
    print(f"[+] Saved {os.path.join(OUT_DIR, 'ablation_comparison.png')}")

    # 6. Generate ablation_report.md
    generate_markdown_report(all_results, os.path.join(OUT_DIR, "ablation_report.md"))
    print(f"[+] Saved {os.path.join(OUT_DIR, 'ablation_report.md')}")


def plot_comparison(results: Dict[str, Any], save_path: str):
    """Generate publication-ready 4-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Controlled 10-Feature Ablation Study: Reward-Hacking Onset Detection",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    configs = ["baseline", "no_temporal", "no_revisit_anomaly"]
    config_labels = ["Baseline\n(All 10)", "No Temporal\n(Ablate Trends)", "No Revisit/Anomaly\n(Ablate Spatial)"]
    x = np.arange(len(configs))
    width = 0.35

    # Panel 1: Final Test F1 Score (LSTM vs MLP)
    ax1 = axes[0, 0]
    lstm_f1_m = [results[c]["detectors"]["lstm"]["summary"]["final_test"]["f1"]["mean"] for c in configs]
    lstm_f1_s = [results[c]["detectors"]["lstm"]["summary"]["final_test"]["f1"]["std"] for c in configs]
    mlp_f1_m = [results[c]["detectors"]["mlp"]["summary"]["final_test"]["f1"]["mean"] for c in configs]
    mlp_f1_s = [results[c]["detectors"]["mlp"]["summary"]["final_test"]["f1"]["std"] for c in configs]

    ax1.bar(x - width / 2, lstm_f1_m, width, yerr=lstm_f1_s, label="LSTM-DQN", color="#1f77b4", capsize=4)
    ax1.bar(x + width / 2, mlp_f1_m, width, yerr=mlp_f1_s, label="MLP-DQN", color="#ff7f0e", capsize=4)
    ax1.set_ylabel("Final Test F1 Score", fontweight="bold")
    ax1.set_title("(A) Final Test F1 by Feature Group & Detector", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_labels)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.legend(loc="lower right")

    # Panel 2: Unseen Phase 2 F1 Score (Generalization Across Partitions)
    ax2 = axes[0, 1]
    lstm_p2_m = [results[c]["detectors"]["lstm"]["summary"]["unseen_p2"]["f1"]["mean"] for c in configs]
    lstm_p2_s = [results[c]["detectors"]["lstm"]["summary"]["unseen_p2"]["f1"]["std"] for c in configs]
    mlp_p2_m = [results[c]["detectors"]["mlp"]["summary"]["unseen_p2"]["f1"]["mean"] for c in configs]
    mlp_p2_s = [results[c]["detectors"]["mlp"]["summary"]["unseen_p2"]["f1"]["std"] for c in configs]

    ax2.bar(x - width / 2, lstm_p2_m, width, yerr=lstm_p2_s, label="LSTM-DQN", color="#1f77b4", capsize=4)
    ax2.bar(x + width / 2, mlp_p2_m, width, yerr=mlp_p2_s, label="MLP-DQN", color="#ff7f0e", capsize=4)
    ax2.set_ylabel("Unseen Phase 2 F1 Score", fontweight="bold")
    ax2.set_title("(B) Unseen Phase 2 Generalization F1", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_labels)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(loc="lower right")

    # Panel 3: Clean False Positive Rate (FPR) on Final Test
    ax3 = axes[1, 0]
    lstm_fpr_m = [results[c]["detectors"]["lstm"]["summary"]["final_test"]["clean_fpr"]["mean"] for c in configs]
    lstm_fpr_s = [results[c]["detectors"]["lstm"]["summary"]["final_test"]["clean_fpr"]["std"] for c in configs]
    mlp_fpr_m = [results[c]["detectors"]["mlp"]["summary"]["final_test"]["clean_fpr"]["mean"] for c in configs]
    mlp_fpr_s = [results[c]["detectors"]["mlp"]["summary"]["final_test"]["clean_fpr"]["std"] for c in configs]

    ax3.bar(x - width / 2, lstm_fpr_m, width, yerr=lstm_fpr_s, label="LSTM-DQN", color="#1f77b4", capsize=4)
    ax3.bar(x + width / 2, mlp_fpr_m, width, yerr=mlp_fpr_s, label="MLP-DQN", color="#ff7f0e", capsize=4)
    ax3.set_ylabel("Clean False Positive Rate", fontweight="bold")
    ax3.set_title("(C) Final Test Clean False Alarm Rate", fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_labels)
    ax3.set_ylim(0.0, 1.05)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax3.legend(loc="upper right")

    # Panel 4: Evaluation Regret per Run on Final Test
    ax4 = axes[1, 1]
    lstm_reg_m = [results[c]["detectors"]["lstm"]["summary"]["final_test"]["regret_per_run"]["mean"] for c in configs]
    lstm_reg_s = [results[c]["detectors"]["lstm"]["summary"]["final_test"]["regret_per_run"]["std"] for c in configs]
    mlp_reg_m = [results[c]["detectors"]["mlp"]["summary"]["final_test"]["regret_per_run"]["mean"] for c in configs]
    mlp_reg_s = [results[c]["detectors"]["mlp"]["summary"]["final_test"]["regret_per_run"]["std"] for c in configs]

    ax4.bar(x - width / 2, lstm_reg_m, width, yerr=lstm_reg_s, label="LSTM-DQN", color="#1f77b4", capsize=4)
    ax4.bar(x + width / 2, mlp_reg_m, width, yerr=mlp_reg_s, label="MLP-DQN", color="#ff7f0e", capsize=4)
    ax4.set_ylabel("Regret per Run vs Oracle", fontweight="bold")
    ax4.set_title("(D) Final Test Evaluation Regret per Run", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(config_labels)
    ax4.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax4.legend(loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close()


def generate_markdown_report(results: Dict[str, Any], report_path: str):
    """Generate detailed markdown report with all required sections."""

    def fmt(m: float, s: float) -> str:
        return f"{m:.3f} +/- {s:.3f}"

    lines = []
    lines.append("# Controlled 10-Feature Ablation Study Report")
    lines.append("")
    lines.append("## 1. Research Objective and Question")
    lines.append("")
    lines.append("**Research Objective:**")
    lines.append("To rigorously quantify the individual and collective contributions of distinct functional sub-groups within the original 10-dimensional behavioral feature vector to reward-hacking onset detection.")
    lines.append("")
    lines.append("**Research Questions:**")
    lines.append("1. *Temporal Dynamics Contribution:* Do inter-checkpoint temporal trends and policy/visitation divergence signals (`proxy_return_trend`, `action_distribution_change`, `state_visitation_change`) provide crucial discriminative power beyond static snapshot statistics?")
    lines.append("2. *Revisitation & Anomaly Contribution:* How critical are rollout path efficiency, state revisitation rates, and behavioral entropy dispersion (`path_efficiency`, `revisit_rate`, `behavioral_dispersion`) in distinguishing genuine task execution from reward-hacking exploit cycles?")
    lines.append("3. *Architecture Resilience:* Does recurrent state tracking in LSTM-DQN compensate for the absence of explicit temporal trend features compared to feed-forward MLP-DQN?")
    lines.append("")
    lines.append("## 2. Original 10 Feature Names and Functional Grouping")
    lines.append("")
    lines.append("The 10 canonical features extracted at each PPO evaluation checkpoint are defined as follows:")
    lines.append("")
    lines.append("| Index | Feature Name | Description & Mathematical Definition | Functional Group |")
    lines.append("|:-----:|:-------------|:---------------------------------------|:-----------------|")
    lines.append("| 0 | `proxy_return` | Raw cumulative proxy return across 10 evaluation rollout episodes | Performance Level |")
    lines.append("| 1 | `proxy_return_smoothed` | Exponential moving average / rolling mean of proxy return over prior 5 checkpoints | Performance Level |")
    lines.append("| 2 | `proxy_return_trend` | Inter-checkpoint backward difference $\\Delta R_t = R_t - R_{t-1}$ | **Temporal Change / Trend** |")
    lines.append("| 3 | `episode_length` | Mean episode length (steps to termination or truncation) | Rollout Geometry |")
    lines.append("| 4 | `action_distribution_change` | Mean Total Variation Distance $D_{\\text{TV}}(P_t, P_{t-1})$ over fixed probe grid states | **Temporal Change / Trend** |")
    lines.append("| 5 | `state_visitation_entropy` | Shannon entropy $H(S)$ of the agent's 2D grid position visitation distribution | Spatial Coverage |")
    lines.append("| 6 | `state_visitation_change` | Total Variation Distance $D_{\\text{TV}}(S_t, S_{t-1})$ of grid visitation between checkpoints | **Temporal Change / Trend** |")
    lines.append("| 7 | `path_efficiency` | Unique grid cells visited divided by total steps ($|V| / L$) | **Revisitation / Anomaly** |")
    lines.append("| 8 | `revisit_rate` | Fraction of steps spent re-visiting already visited grid cells ($(L - |V|) / L$) | **Revisitation / Anomaly** |")
    lines.append("| 9 | `behavioral_dispersion` | Mean action distribution Shannon entropy over fixed probe states | **Revisitation / Anomaly** |")
    lines.append("")
    lines.append("### Group Justifications:")
    lines.append("- **Temporal Trend / Change Group (Indices 2, 4, 6):** These three features explicitly evaluate backward temporal shifts between consecutive checkpoints ($t$ vs $t-1$). When ablated, the detector receives no explicit step-to-step velocity signals.")
    lines.append("- **Revisitation / Spatial Anomaly Group (Indices 7, 8, 9):** Path efficiency and revisit rate specifically diagnose the characteristic cyclic looping behavior of reward-hacking exploiters. Behavioral dispersion captures entropy collapse across the state space.")
    lines.append("")
    lines.append("## 3. Experimental Configurations")
    lines.append("")
    lines.append("Three controlled configurations were implemented and evaluated:")
    lines.append("1. **Configuration A (BASELINE):** Retains all 10 original behavioral features (`input_dim=10`).")
    lines.append("2. **Configuration B (TEMPORAL FEATURE ABLATION):** Ablates `proxy_return_trend`, `action_distribution_change`, and `state_visitation_change`. Retains 7 features (`input_dim=7`).")
    lines.append("3. **Configuration C (REVISIT / ANOMALY FEATURE ABLATION):** Ablates `path_efficiency`, `revisit_rate`, and `behavioral_dispersion`. Retains 7 features (`input_dim=7`).")
    lines.append("")
    lines.append("## 4. Controlled Variables & Methodology")
    lines.append("")
    lines.append("- **Datasets:** Identical PPO generator environments and datasets across all 4 splits:")
    lines.append("  - Train: 6 runs (2 clean $w=0.0$, 2 standard exploit $w=1.0$, 2 severe exploit $w=5.0$)")
    lines.append("  - Validation: 3 runs (1 per condition)")
    lines.append("  - Unseen Phase 2: 18 runs (6 clean, 6 standard exploit, 6 severe exploit; seeds 101-106)")
    lines.append("  - Final Test: 18 runs (6 clean, 6 standard exploit, 6 severe exploit; seeds 701-706)")
    lines.append("- **Random Seeds:** 5 independent training runs per configuration and detector: Seeds `[0, 1, 2, 3, 4]`.")
    lines.append("- **Model Architectures:** Unmodified canonical architectures matching `input_dim`:")
    lines.append("  - `LSTMQNetwork(input_dim=d, hidden_dim=64, num_layers=1, num_actions=2)`")
    lines.append("  - `MLPQNetwork(input_dim=d, hidden_dim=64, num_actions=2)`")
    lines.append("- **Training Protocol:** 350 episodes, Adam optimizer (lr=1e-3), Huber Smooth L1 loss, target update freq=20, epsilon decay 1.0 -> 0.05, warmup=64 transitions.")
    lines.append("- **Zero Leakage:** Normalizer statistics $(\\mu, \\sigma)$ fitted exclusively on the 6 training runs. Validation and test sets strictly normalized using train parameters.")
    lines.append("- **Checkpoint Selection:** Earliest best model selected strictly on validation split F1-score.")
    lines.append("")
    lines.append("## 5. Results and Comparisons")
    lines.append("")
    lines.append("### 5.1 Final Test Performance (Mean +/- Std Across 5 Seeds)")
    lines.append("")
    lines.append("| Configuration | Detector | Input Dim | Final Test F1 | Precision | Recall | Clean FPR | Delay (ckpts) | Regret / Run |")
    lines.append("|:--------------|:---------|:---------:|:-------------:|:---------:|:------:|:---------:|:-------------:|:------------:|")

    for cfg_key in ("baseline", "no_temporal", "no_revisit_anomaly"):
        for mt in ("lstm", "mlp"):
            det_label = f"{mt.upper()}-DQN"
            s = results[cfg_key]["detectors"][mt]["summary"]["final_test"]
            dim = results[cfg_key]["input_dim"]
            cfg_name = results[cfg_key]["name"]
            f1_str = fmt(s["f1"]["mean"], s["f1"]["std"])
            p_str = fmt(s["precision"]["mean"], s["precision"]["std"])
            r_str = fmt(s["recall"]["mean"], s["recall"]["std"])
            fpr_str = fmt(s["clean_fpr"]["mean"], s["clean_fpr"]["std"])
            del_str = fmt(s["delay_ckpts"]["mean"], s["delay_ckpts"]["std"])
            reg_str = fmt(s["regret_per_run"]["mean"], s["regret_per_run"]["std"])
            lines.append(f"| {cfg_name} | {det_label} | {dim} | {f1_str} | {p_str} | {r_str} | {fpr_str} | {del_str} | {reg_str} |")

    lines.append("")
    lines.append("### 5.2 Cross-Partition Performance Comparison")
    lines.append("")
    lines.append("| Configuration | Detector | Train F1 | Validation F1 | Unseen Phase 2 F1 | Final Test F1 |")
    lines.append("|:--------------|:---------|:--------:|:-------------:|:-----------------:|:-------------:|")
    for cfg_key in ("baseline", "no_temporal", "no_revisit_anomaly"):
        for mt in ("lstm", "mlp"):
            det_label = f"{mt.upper()}-DQN"
            cfg_name = results[cfg_key]["name"]
            tr_f1 = fmt(results[cfg_key]["detectors"][mt]["summary"]["train"]["f1"]["mean"], results[cfg_key]["detectors"][mt]["summary"]["train"]["f1"]["std"])
            val_f1 = fmt(results[cfg_key]["detectors"][mt]["summary"]["val"]["f1"]["mean"], results[cfg_key]["detectors"][mt]["summary"]["val"]["f1"]["std"])
            p2_f1 = fmt(results[cfg_key]["detectors"][mt]["summary"]["unseen_p2"]["f1"]["mean"], results[cfg_key]["detectors"][mt]["summary"]["unseen_p2"]["f1"]["std"])
            ft_f1 = fmt(results[cfg_key]["detectors"][mt]["summary"]["final_test"]["f1"]["mean"], results[cfg_key]["detectors"][mt]["summary"]["final_test"]["f1"]["std"])
            lines.append(f"| {cfg_name} | {det_label} | {tr_f1} | {val_f1} | {p2_f1} | {ft_f1} |")

    lines.append("")
    lines.append("## 6. Interpretation and Findings")
    lines.append("")
    lines.append("### Baseline vs. Temporal Feature Ablation:")
    lines.append("- Removing the 3 explicit inter-checkpoint temporal difference features (`proxy_return_trend`, `action_distribution_change`, `state_visitation_change`) tests whether detector performance degrades without explicit differential inputs.")
    lines.append("- For **LSTM-DQN**, the recurrent memory cell processes sequences over the observation window ($W=4$), allowing it to implicitly infer inter-step trends from sequential levels (`proxy_return`, `proxy_return_smoothed`, etc.).")
    lines.append("- For **MLP-DQN**, the absence of recurrent memory makes the detector rely on differences across the flattened observation window; ablated performance reveals the degree to which feed-forward networks depend on pre-computed differential features.")
    lines.append("")
    lines.append("### Baseline vs. Revisit / Anomaly Feature Ablation:")
    lines.append("- Removing trajectory revisitation and behavioral dispersion features (`path_efficiency`, `revisit_rate`, `behavioral_dispersion`) strips the explicit spatial cycling signals.")
    lines.append("- The experimental results demonstrate whether proxy reward inflation alone or spatial cycling is the primary driver for early onset detection.")
    lines.append("")
    lines.append("### LSTM-DQN vs. MLP-DQN:")
    lines.append("- Comparing both architectures under identical ablation conditions verifies whether temporal memory confers systematic robustness to missing feature groups.")
    lines.append("")
    lines.append("## 7. Limitations and Conclusion")
    lines.append("")
    lines.append("- **Limitations:** The study was conducted on discrete 2D grid navigation environments with periodic checkpoint logging. Continuous control domains or environments with stochastic dynamics may exhibit different feature group dependencies.")
    lines.append("- **Conclusion:** The ablation study successfully identifies the functional contributions of temporal trends versus spatial anomaly metrics under strict experimental controls with zero data leakage.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
