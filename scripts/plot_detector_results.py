"""
plot_detector_results.py

Generates publication-quality scientific plots from actual experiment outputs.
All plots are derived strictly from experimental training logs and evaluation predictions.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def plot_all(
    results_dir: str = "results/detector",
    plots_dir: str = "results/detector_plots",
):
    os.makedirs(plots_dir, exist_ok=True)
    training_log_path = os.path.join(results_dir, "training_log.json")
    preds_test_path = os.path.join(results_dir, "predictions_test.json")
    preds_all_path = os.path.join(results_dir, "predictions_all.json")

    # Use all predictions if available, else test
    preds_path = preds_all_path if os.path.exists(preds_all_path) else preds_test_path

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12})

    # -------------------------------------------------------------
    # Plot 1: Training Dynamics (Reward, Loss, Epsilon, Val F1)
    # -------------------------------------------------------------
    if os.path.exists(training_log_path):
        with open(training_log_path, "r") as f:
            log = json.load(f)

        episodes = log["episode"]
        rewards = log["reward"]
        losses = log["loss"]
        epsilons = log["epsilon"]
        val_f1 = log.get("val_f1", [])

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Reward
        axs[0, 0].plot(episodes, rewards, color="#1f77b4", alpha=0.3, label="Raw Return")
        if len(rewards) >= 10:
            window = 10
            rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
            axs[0, 0].plot(episodes[window - 1 :], rolling, color="#1f77b4", lw=2, label="10-Ep Rolling Mean")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Episode Return")
        axs[0, 0].set_title("Detector Episode Return over Training")
        axs[0, 0].legend()

        # Loss
        valid_losses = [l for l in losses if l > 0]
        valid_eps = [episodes[i] for i, l in enumerate(losses) if l > 0]
        if valid_losses:
            axs[0, 1].plot(valid_eps, valid_losses, color="#d62728", lw=1.5)
            axs[0, 1].set_xlabel("Episode")
            axs[0, 1].set_ylabel("Huber Loss")
            axs[0, 1].set_title("DQN Bellman Loss over Training")

        # Epsilon
        axs[1, 0].plot(episodes, epsilons, color="#2ca02c", lw=2)
        axs[1, 0].set_xlabel("Episode")
        axs[1, 0].set_ylabel("Epsilon (Exploration Rate)")
        axs[1, 0].set_title("Exploration Schedule Decay")

        # Validation F1
        val_points = [(episodes[i], f1) for i, f1 in enumerate(val_f1) if f1 > 0]
        if val_points:
            v_ep, v_f1 = zip(*val_points)
            axs[1, 1].plot(v_ep, v_f1, marker="o", color="#9467bd", lw=2, label="Validation F1")
            axs[1, 1].set_ylim(-0.05, 1.05)
            axs[1, 1].set_xlabel("Episode")
            axs[1, 1].set_ylabel("F1 Score")
            axs[1, 1].set_title("Validation F1 Score Tracking")
            axs[1, 1].legend()

        plt.tight_layout()
        out_p1 = os.path.join(plots_dir, "01_detector_training_dynamics.png")
        plt.savefig(out_p1, dpi=300)
        plt.close()
        print(f"[Plot] Saved: {out_p1}")

    # -------------------------------------------------------------
    # Plot 2: Comparative Metrics (Precision, Recall, F1, Detection%)
    # -------------------------------------------------------------
    if os.path.exists(preds_path):
        with open(preds_path, "r") as f:
            data = json.load(f)

        detector_names = list(data.keys())
        p_vals = [data[k]["precision"] for k in detector_names]
        r_vals = [data[k]["recall"] for k in detector_names]
        f1_vals = [data[k]["f1_score"] for k in detector_names]
        det_pct = [data[k]["hacking_detected_pct"] / 100.0 for k in detector_names]

        x = np.arange(len(detector_names))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - 1.5 * width, p_vals, width, label="Precision", color="#1f77b4")
        ax.bar(x - 0.5 * width, r_vals, width, label="Recall", color="#2ca02c")
        ax.bar(x + 0.5 * width, f1_vals, width, label="F1-Score", color="#ff7f0e")
        ax.bar(x + 1.5 * width, det_pct, width, label="Hacking Detected Rate", color="#d62728")

        ax.set_ylabel("Score / Rate")
        ax.set_title("Comparative Performance: LSTM-DQN vs Mathematical Onset Baselines")
        ax.set_xticks(x)
        ax.set_xticklabels(detector_names, rotation=15)
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right")

        plt.tight_layout()
        out_p2 = os.path.join(plots_dir, "02_detector_comparison_metrics.png")
        plt.savefig(out_p2, dpi=300)
        plt.close()
        print(f"[Plot] Saved: {out_p2}")

        # -------------------------------------------------------------
        # Plot 3: Detection Delay Distribution
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        delay_data = {}
        for k in detector_names:
            delays = [p["delay_ckpts"] for p in data[k]["predictions"] if p.get("delay_ckpts") is not None]
            delay_data[k] = delays

        positions = range(len(detector_names))
        mean_delays = [np.mean(delay_data[k]) if delay_data[k] else 0.0 for k in detector_names]
        bars = ax.bar(positions, mean_delays, color=["#1f77b4", "#7f7f7f", "#2ca02c", "#d62728"], width=0.5)

        for bar, k in zip(bars, detector_names):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f"{yval:.2f} ckpts", ha="center", va="bottom", fontweight="bold")

        ax.set_xticks(positions)
        ax.set_xticklabels(detector_names, rotation=15)
        ax.set_ylabel("Mean Detection Delay (Checkpoints)")
        ax.set_title("Mean Onset Detection Delay by Method (Lower is Better)")
        ax.set_ylim(0, max(mean_delays, default=1.0) * 1.35)

        plt.tight_layout()
        out_p3 = os.path.join(plots_dir, "03_detection_delay_distribution.png")
        plt.savefig(out_p3, dpi=300)
        plt.close()
        print(f"[Plot] Saved: {out_p3}")

        # -------------------------------------------------------------
        # Plot 4: Clean Control False Alarm Analysis
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        fp_pcts = [data[k]["clean_flagged_pct"] for k in detector_names]
        bars = ax.bar(positions, fp_pcts, color=["#2ca02c" if p == 0 else "#d62728" for p in fp_pcts], width=0.5)

        for bar, p in zip(bars, fp_pcts):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{p:.1f}%", ha="center", va="bottom", fontweight="bold")

        ax.set_xticks(positions)
        ax.set_xticklabels(detector_names, rotation=15)
        ax.set_ylabel("Clean Control False Alarm Rate (%)")
        ax.set_title("False Alarm Rate on Clean Negative Control Runs (w=0.0)")
        ax.set_ylim(0, max(max(fp_pcts, default=0.0) * 1.4, 20.0))

        plt.tight_layout()
        out_p4 = os.path.join(plots_dir, "04_clean_control_false_alarm_analysis.png")
        plt.savefig(out_p4, dpi=300)
        plt.close()
        print(f"[Plot] Saved: {out_p4}")

    # -------------------------------------------------------------
    # Plot 5: True Onset vs Predicted Onset Across Example Runs
    # -------------------------------------------------------------
    split_meta_path = os.path.join(results_dir, "split_metadata.json")
    if os.path.exists(split_meta_path) and os.path.exists(preds_path):
        with open(split_meta_path, "r") as f:
            split_meta = json.load(f)
        with open(preds_path, "r") as f:
            preds_data = json.load(f)

        dqn_preds = {p["run_id"]: p for p in preds_data.get("LSTM-DQN", {}).get("predictions", [])}
        o2_preds = {p["run_id"]: p for p in preds_data.get("Mathematical-O2", {}).get("predictions", [])}

        # Select up to 4 illustrative runs (1 clean, 1 w=1.0, 2 w=5.0)
        test_ids = split_meta["test"]
        sample_ids = test_ids[:4] if len(test_ids) >= 4 else test_ids

        fig, axs = plt.subplots(len(sample_ids), 1, figsize=(10, 3 * len(sample_ids)), sharex=True)
        if len(sample_ids) == 1:
            axs = [axs]

        run_paths = split_meta["run_paths"]
        for idx, rid in enumerate(sample_ids):
            r_path = run_paths[rid]
            temp_path = os.path.join(r_path, "features", "temporal_features.jsonl")
            off_path = os.path.join(r_path, "features", "offline_eval.jsonl")

            if not os.path.exists(temp_path) or not os.path.exists(off_path):
                continue

            steps, proxy, true_ret = [], [], []
            with open(temp_path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        steps.append(int(r["environment_steps"]))
                        proxy.append(float(r["proxy_return"]))
            with open(off_path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        true_ret.append(float(r.get("true_return", 0.0)))

            ax = axs[idx]
            ax.plot(steps, proxy, label="Proxy Return (Observed)", color="#1f77b4", lw=2)
            ax.plot(steps, true_ret, label="True Objective Return (Hidden)", color="#2ca02c", lw=2, linestyle=":")

            dqn_p = dqn_preds.get(rid)
            o2_p = o2_preds.get(rid)

            # Ground truth onset vertical line
            if dqn_p and dqn_p.get("true_onset_step") is not None:
                ax.axvline(dqn_p["true_onset_step"], color="black", linestyle="--", lw=1.5, label="True Onset t*")

            # LSTM-DQN predicted flag
            if dqn_p and dqn_p.get("predicted_step") is not None:
                ax.axvline(dqn_p["predicted_step"], color="#d62728", lw=2, label=f"LSTM-DQN Flag ({dqn_p['predicted_step']})")

            # O2 predicted flag
            if o2_p and o2_p.get("predicted_step") is not None:
                ax.axvline(o2_p["predicted_step"], color="#ff7f0e", linestyle="-.", lw=1.5, label=f"O2 Flag ({o2_p['predicted_step']})")

            ax.set_title(f"Run: {rid} ({'Clean Control' if (dqn_p and dqn_p['is_clean']) else 'Reward Hacking Exploit'})")
            ax.set_ylabel("Return")
            if idx == 0:
                ax.legend(loc="upper left")

        axs[-1].set_xlabel("Environment Steps")
        plt.tight_layout()
        out_p5 = os.path.join(plots_dir, "05_true_vs_predicted_onset_trajectory.png")
        plt.savefig(out_p5, dpi=300)
        plt.close()
        print(f"[Plot] Saved: {out_p5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scientific plots for detector results.")
    parser.add_argument("--results-dir", type=str, default="results/detector")
    parser.add_argument("--plots-dir", type=str, default="results/detector_plots")
    args = parser.parse_args()

    plot_all(results_dir=args.results_dir, plots_dir=args.plots_dir)
