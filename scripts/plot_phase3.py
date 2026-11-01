"""plot_phase3.py - figures for Phase 3 from results/phase3_final/phase3_results.json (read-only)."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/phase3_final"
R = json.load(open(os.path.join(OUT, "phase3_results.json")))
P = os.path.join(OUT, "plots")
os.makedirs(P, exist_ok=True)
COL = {"lstm": "#2b5c8f", "mlp": "#e76f51"}


def curves(mt, key, xkey):
    runs = R["training_runs"][mt]
    xs = np.array([c[xkey] for c in runs[0]["curve"]], dtype=float)
    ys = np.array([[c[key] for c in r["curve"]] for r in runs], dtype=float)
    if xkey == "interactions":  # interactions differ per seed -> average x as well
        xs = np.array([[c[xkey] for c in r["curve"]] for r in runs], dtype=float).mean(0)
    return xs, ys.mean(0), ys.std(0)


# Sample efficiency
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, xkey, xl in [(axes[0], "episode", "Training episodes (generator runs replayed)"),
                     (axes[1], "interactions", "Detector environment interactions (checkpoint transitions)")]:
    for mt in ("lstm", "mlp"):
        for key, ls in (("val_f1", "-"), ("train_f1", ":")):
            x, m, s = curves(mt, key, xkey)
            ax.plot(x, m, ls, color=COL[mt], label=f"{mt.upper()} {key.split('_')[0]}")
            if key == "val_f1":
                ax.fill_between(x, m - s, m + s, color=COL[mt], alpha=0.15)
    ax.axhline(R["f1_convergence_threshold"], color="gray", ls="--", lw=1, label="threshold")
    ax.set_xlabel(xl); ax.set_ylabel("F1 (mean ± std over training seeds)"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
fig.suptitle(f"Sample efficiency (n={len(R['train_seeds'])} training seeds; val = 3 runs)")
plt.tight_layout(); plt.savefig(os.path.join(P, "sample_efficiency.png"), dpi=200); plt.close()

# Cumulative training regret
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for mt in ("lstm", "mlp"):
    x, m, s = curves(mt, "cum_train_regret", "episode")
    ax.plot(x, m, color=COL[mt], label=mt.upper()); ax.fill_between(x, m - s, m + s, color=COL[mt], alpha=0.15)
ax.set_xlabel("Training episodes"); ax.set_ylabel("Cumulative regret vs. oracle stopping policy")
ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(P, "cumulative_regret.png"), dpi=200); plt.close()

# Generalization gap
parts = ["train", "val", "unseen_p2", "final_test"]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
xs = np.arange(len(parts))
for i, mt in enumerate(("lstm", "mlp")):
    s = R["summary_deployable"][mt]["partitions"]
    ax.bar(xs + (i - 0.5) * 0.35, [s[p]["f1"]["mean"] for p in parts], 0.35,
           yerr=[s[p]["f1"]["std"] for p in parts], capsize=4, color=COL[mt], label=mt.upper())
for j, key in enumerate(("o1", "o2")):
    ax.plot(xs, [R["oracle_baselines"][key][p]["f1"] for p in parts], "o--" if key == "o1" else "s--",
            color="gray" if key == "o1" else "black", label=f"{key.upper()} (oracle, uses true reward)")
ax.set_xticks(xs); ax.set_xticklabels(parts); ax.set_ylabel("F1"); ax.set_ylim(0, 1.1); ax.legend(fontsize=8)
ax.set_title("Generalization across partitions (mean ± std over training seeds)")
plt.tight_layout(); plt.savefig(os.path.join(P, "generalization_gap.png"), dpi=200); plt.close()

# Sensitivity
names = list(R["sensitivity"])
m = [R["sensitivity"][n]["unseen_p2_f1"]["mean"] for n in names]
s = [R["sensitivity"][n]["unseen_p2_f1"]["std"] for n in names]
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(range(len(names)), m, yerr=s, capsize=3, color=["#2a9d8f" if n == "baseline" else "#8ab17d" for n in names])
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Unseen (Phase 2) F1"); ax.set_ylim(0, 1.1)
ax.set_title(f"LSTM-DQN sensitivity (n={len(R['sensitivity']['baseline']['per_seed'])} training seeds each)")
plt.tight_layout(); plt.savefig(os.path.join(P, "sensitivity.png"), dpi=200); plt.close()
print("[OK] plots in", P)
