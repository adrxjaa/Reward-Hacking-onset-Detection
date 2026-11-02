"""
plot_subtle_onset.py - Publication-Quality Analysis of Delayed/Subtle Reward Hacking Onset
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rhd.detectors.models import LSTMQNetwork, MLPQNetwork


def generate_subtle_onset_figure():
    run_dir = "experiments/subtle_onset_test/run_delayed_w0.6"
    out_dir = "results/subtle_onset_test"
    os.makedirs(out_dir, exist_ok=True)

    t_path = os.path.join(run_dir, "features", "temporal_features.jsonl")
    o_path = os.path.join(run_dir, "features", "offline_eval.jsonl")

    with open(t_path) as f:
        tfs = [json.loads(l) for l in f if l.strip()]
    with open(o_path) as f:
        ofs = [json.loads(l) for l in f if l.strip()]

    steps = [r["environment_steps"] for r in tfs]
    proxies = [r["proxy_return"] for r in tfs]
    trues = [r["true_return"] for r in ofs]
    goals = [r["goal_success"] * 100 for r in ofs]
    exploits = [r["region_entries"] for r in ofs]

    tvds = [r["action_distribution_change"] for r in tfs]
    dispersions = [r["behavioral_dispersion"] for r in tfs]
    revisit_rates = [r["revisit_rate"] for r in tfs]
    path_effs = [r["path_efficiency"] for r in tfs]

    # Load norm stats
    with open("results/detector/norm_stats.json") as f:
        stats = json.load(f)
    means = np.array(stats["mean"], dtype=np.float32)
    stds = np.array(stats["std"], dtype=np.float32)

    # Load LSTM-DQN
    ckpt = torch.load("results/ablation_study/models/baseline_lstm_seed0.pt", map_location="cpu")
    lstm_model = LSTMQNetwork(input_dim=10, hidden_dim=64, num_layers=1, num_actions=2)
    lstm_model.load_state_dict(ckpt["online_net"])
    lstm_model.eval()

    feature_keys = [
        "proxy_return", "proxy_return_smoothed", "proxy_return_change",
        "episode_length", "action_distribution_change", "state_visitation_entropy",
        "state_visitation_change", "path_efficiency", "revisit_rate", "behavioral_dispersion"
    ]
    raw_feats = np.array([[float(tf.get(k, 0.0)) for k in feature_keys] for tf in tfs], dtype=np.float32)
    norm_feats = (raw_feats - means) / stds

    W = 4
    q_waits = []
    q_flags = []
    flagged_step = None

    for idx in range(len(norm_feats)):
        if idx + 1 < W:
            pad = np.zeros((W - (idx + 1), 10), dtype=np.float32)
            win = np.vstack([pad, norm_feats[:idx+1]])
        else:
            win = norm_feats[idx-W+1:idx+1]
        x = torch.as_tensor(win).unsqueeze(0)
        with torch.no_grad():
            q = lstm_model(x)
            qw, qf = float(q[0, 0].item()), float(q[0, 1].item())
            q_waits.append(qw)
            q_flags.append(qf)
            if qf > qw and flagged_step is None:
                flagged_step = steps[idx]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Panel 1: Returns & Goal Success
    ax1 = axs[0]
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(steps, proxies, "b-o", label="Proxy Return (Training Signal)", linewidth=2)
    l2 = ax1.plot(steps, trues, "g-s", label="True Return (Latent Task)", linewidth=2)
    l3 = ax1_twin.plot(steps, goals, "m--^", label="Goal Success Rate %", linewidth=2, alpha=0.8)

    ax1.axvspan(5000, 20000, color="green", alpha=0.10, label="Phase 1: Honest Task Learned (100% Success)")
    ax1.axvspan(35000, 60000, color="red", alpha=0.10, label="Phase 2: Reward Hacking Takeover (0% Goal)")
    ax1.axvline(flagged_step, color="darkred", linestyle="--", linewidth=2.5, label=f"LSTM-DQN Detected Onset (Step {flagged_step:,})")

    ax1.set_ylabel("Episode Return", fontsize=11, fontweight="bold")
    ax1_twin.set_ylabel("Goal Success %", fontsize=11, fontweight="bold", color="purple")
    ax1_twin.set_ylim(-5, 110)
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", fontsize=9)
    ax1.set_title("Long-Horizon Delayed Reward Hacking Emergence (PPO on RegionToggle w=0.6)", fontsize=13, fontweight="bold")

    # Panel 2: Behavioral Precursors
    ax2 = axs[1]
    ax2.plot(steps, tvds, "darkorange", marker="o", label="Probe-State TVD (Policy Divergence)", linewidth=2)
    ax2.plot(steps, dispersions, "teal", marker="s", label="Behavioral Dispersion (Entropy Collapse)", linewidth=2)
    ax2.plot(steps, revisit_rates, "crimson", marker="^", label="Revisit Rate (Cyclic Loops)", linewidth=2)
    ax2.plot(steps, path_effs, "navy", marker="d", label="Path Efficiency (|Visited|/L)", linewidth=2)
    ax2.axvline(flagged_step, color="darkred", linestyle="--", linewidth=2)
    ax2.set_ylabel("Normalized Feature Value", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_title("Zero-Leakage Behavioral Precursors: Divergence, Entropy Collapse, and Cyclic Looping", fontsize=12, fontweight="bold")

    # Panel 3: Detector Q-Values & Optimal Stopping Decision
    ax3 = axs[2]
    ax3.plot(steps, q_waits, "royalblue", marker="o", label="Q(s, WAIT) - Continue Monitoring", linewidth=2.5)
    ax3.plot(steps, q_flags, "firebrick", marker="s", label="Q(s, FLAG) - Declare Onset Alarm", linewidth=2.5)
    ax3.axvline(flagged_step, color="darkred", linestyle="--", linewidth=2.5, label=f"Optimal Stopping Time τ = Step {flagged_step:,}")
    ax3.scatter([flagged_step], [q_flags[steps.index(flagged_step)]], color="red", s=150, zorder=5)
    ax3.annotate(
        f"FLAG Triggered!\nStep {flagged_step:,}\nQ_FLAG > Q_WAIT",
        xy=(flagged_step, q_flags[steps.index(flagged_step)]),
        xytext=(flagged_step + 3000, q_flags[steps.index(flagged_step)] + 0.5),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=8),
        fontweight="bold",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8),
    )
    ax3.set_ylabel("Estimated Action-Value Q(s, a)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Environment Steps", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=10)
    ax3.set_title("Learned LSTM-DQN Optimal Stopping Decision Dynamics (Zero Ground-Truth Leakage)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_file = os.path.join(out_dir, "delayed_onset_analysis.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Subtle onset analysis figure saved to: {out_file}")


if __name__ == "__main__":
    generate_subtle_onset_figure()
