"""
plot_validation_phase2.py

Generates publication-quality scientific plots for Phase 2 Experimental Validation:
1. Ablation comparison bar chart (LSTM-DQN vs MLP-DQN vs O1 vs O2)
2. Per-condition detection performance breakdown (Clean w=0.0, Standard w=1.0, Severe w=5.0)
3. Detection latency / delay distribution (mean +/- std checkpoints)
4. Multi-panel confusion matrices for all 4 detection systems
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_style():
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

def plot_ablation_f1_comparison(report: dict, out_dir: str):
    methods = ["LSTM-DQN (Sequential)", "MLP-DQN (Ablation: Non-Recurrent)", "Mathematical-O1", "Mathematical-O2"]
    short_labels = ["LSTM-DQN\n(Sequential)", "MLP-DQN\n(Ablation)", "Math O1\n(Baseline)", "Math O2\n(Baseline)"]
    
    precisions = [report[m]["precision"] for m in methods]
    recalls = [report[m]["recall"] for m in methods]
    f1s = [report[m]["f1_score"] for m in methods]
    clean_fprs = [report[m]["clean_fp_rate"] for m in methods]

    x = np.arange(len(methods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - 1.5 * width, precisions, width, label="Precision", color="#2b5c8f")
    rects2 = ax.bar(x - 0.5 * width, recalls, width, label="Recall", color="#2a9d8f")
    rects3 = ax.bar(x + 0.5 * width, f1s, width, label="F1-Score", color="#e76f51")
    rects4 = ax.bar(x + 1.5 * width, clean_fprs, width, label="Clean FPR (False Alarm)", color="#e63946", hatch="//")

    ax.set_ylabel("Score / Rate")
    ax.set_title("Detector Benchmark & Recurrence Ablation (18 Unseen Runs, Seeds 501–506)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "ablation_f1_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path}")

def plot_per_condition_breakdown(report: dict, out_dir: str):
    methods = ["LSTM-DQN (Sequential)", "MLP-DQN (Ablation: Non-Recurrent)", "Mathematical-O1", "Mathematical-O2"]
    short_labels = ["LSTM-DQN", "MLP-DQN", "Math O1", "Math O2"]
    
    clean_tn = [report[m]["per_condition"]["w0.0"]["true_negatives"] / report[m]["per_condition"]["w0.0"]["total"] * 100 for m in methods]
    w1_tp = [report[m]["per_condition"]["w1.0"]["detection_rate"] * 100 for m in methods]
    w5_tp = [report[m]["per_condition"]["w5.0"]["detection_rate"] * 100 for m in methods]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width, clean_tn, width, label="Clean (w=0.0) Specificity (TN%)", color="#2a9d8f")
    rects2 = ax.bar(x, w1_tp, width, label="Standard Exploit (w=1.0) Det% (TP%)", color="#f4a261")
    rects3 = ax.bar(x + width, w5_tp, width, label="Severe Exploit (w=5.0) Det% (TP%)", color="#e76f51")

    ax.set_ylabel("Accuracy / Detection Rate (%)")
    ax.set_title("Per-Condition Performance Breakdown Across Exploit Severities")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 115)
    ax.axhline(100.0, color="gray", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "per_condition_performance.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path}")

def plot_detection_latency(report: dict, out_dir: str):
    methods = ["LSTM-DQN (Sequential)", "MLP-DQN (Ablation: Non-Recurrent)", "Mathematical-O1", "Mathematical-O2"]
    short_labels = ["LSTM-DQN", "MLP-DQN", "Math O1", "Math O2"]
    
    means = [report[m]["mean_delay_ckpts"] for m in methods]
    stds = [report[m]["std_delay_ckpts"] for m in methods]
    colors = ["#2b5c8f", "#457b9d", "#e63946", "#e76f51"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(short_labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Detection Delay (Checkpoints after True Onset)")
    ax.set_title(r"Detection Delay ($\mu \pm \sigma$ Checkpoints) for Detected True Positives")
    ax.set_ylim(0, max(max(means) + max(stds) + 1.0, 3.5))

    for bar, m, s in zip(bars, means, stds):
        ax.annotate(f"{m:.2f} ± {s:.2f} ckpts",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + s),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "detection_latency_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path}")

def plot_confusion_matrices(report: dict, out_dir: str):
    methods = [
        ("LSTM-DQN (Sequential)", "LSTM-DQN (Sequential)"),
        ("MLP-DQN (Ablation: Non-Recurrent)", "MLP-DQN (Ablation)"),
        ("Mathematical-O1", "Mathematical-O1"),
        ("Mathematical-O2", "Mathematical-O2"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (m_key, title) in enumerate(methods):
        ax = axes[idx]
        cm = report[m_key]["confusion_matrix"]
        
        # 2x2 matrix: [ [TN, FP], [FN, TP] ]
        # Clean runs: TN, FP_clean
        # Hack runs: FN, TP (+ FP_early noted)
        matrix = np.array([
            [cm["TN"], cm["FP_clean"]],
            [cm["FN"], cm["TP"]]
        ], dtype=float)
        
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Clean", "Predicted Hacking"])
        ax.set_yticklabels(["Actual Clean", "Actual Hacking"])

        annot_texts = [
            [f"TN\n{cm['TN']}", f"FP\n{cm['FP_clean']}"],
            [f"FN\n{cm['FN']}", f"TP\n{cm['TP']}"]
        ]
        if cm["FP_early"] > 0:
            annot_texts[1][1] += f"\n(+{cm['FP_early']} early)"

        for r_i in range(2):
            for c_i in range(2):
                val = matrix[r_i, c_i]
                txt_color = "white" if val > 6 else "black"
                ax.text(c_i, r_i, annot_texts[r_i][c_i], ha="center", va="center",
                        fontsize=11, fontweight="bold", color=txt_color)

        ax.set_title(f"{title}\nF1 = {report[m_key]['f1_score']:.3f} | Delay = {report[m_key]['mean_delay_ckpts']:.2f} ckpts", fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved {out_path}")

def generate_all_validation_plots(report_json_path: str = "results/validation_phase2/phase2_validation_report.json"):
    assert os.path.exists(report_json_path), f"Report not found at {report_json_path}"
    with open(report_json_path) as fp:
        report = json.load(fp)

    out_dir = os.path.join(os.path.dirname(report_json_path), "plots")
    os.makedirs(out_dir, exist_ok=True)
    set_style()

    plot_ablation_f1_comparison(report, out_dir)
    plot_per_condition_breakdown(report, out_dir)
    plot_detection_latency(report, out_dir)
    plot_confusion_matrices(report, out_dir)
    print(f"\n[ALL DONE] All validation plots generated in {out_dir}")

if __name__ == "__main__":
    generate_all_validation_plots()
