import json
import os
import matplotlib.pyplot as plt
from rhd.evaluation.onset import process_run

def plot_run_onset(run_dir: str):
    feat_dir = os.path.join(run_dir, "features")
    temp_path = os.path.join(feat_dir, "temporal_features.jsonl")
    off_path = os.path.join(feat_dir, "offline_eval.jsonl")
    
    if not os.path.exists(temp_path) or not os.path.exists(off_path):
        print(f"Skipping {run_dir}, missing features")
        return
        
    res = process_run(run_dir)
    
    steps = []
    proxy_returns = []
    true_returns = []
    goal_success = []
    region_entries = []
    
    with open(temp_path) as f:
        tfs = [json.loads(line) for line in f]
    with open(off_path) as f:
        oms = [json.loads(line) for line in f]
        
    for tf, om in zip(tfs, oms):
        steps.append(tf["environment_steps"])
        proxy_returns.append(tf["proxy_return"])
        true_returns.append(om["true_return"])
        goal_success.append(om["goal_success"])
        region_entries.append(om["region_entries"])
        
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    axs[0].plot(steps, proxy_returns, label="Proxy Return", color="blue")
    axs[0].plot(steps, true_returns, label="True Return", color="green")
    
    # Mark primary onset
    if res["primary_detected"]:
        axs[0].axvline(res["primary_onset_step"], color='red', linestyle='--', label='Primary Onset')
    elif res["o1_detected"]:
        axs[0].axvline(res["o1_onset_step"], color='orange', linestyle='--', label='O1 Onset Only')
    elif res["o2_detected"]:
        axs[0].axvline(res["o2_onset_step"], color='orange', linestyle='--', label='O2 Onset Only')
        
    axs[0].legend()
    axs[0].set_title(f"Returns - {os.path.basename(run_dir)}")
    
    gap = [p - t for p, t in zip(proxy_returns, true_returns)]
    axs[1].plot(steps, gap, label="Proxy-True Gap", color="purple")
    axs[1].legend()
    axs[1].set_title("Proxy/True Gap")
    
    axs[2].plot(steps, goal_success, label="Goal Success", color="green")
    axs[2].legend()
    axs[2].set_title("Goal Success")
    
    axs[3].plot(steps, region_entries, label="Region Entries (Exploit)", color="red")
    axs[3].legend()
    axs[3].set_title("Exploit Indicator")
    axs[3].set_xlabel("Environment Steps")
    
    plt.tight_layout()
    out_path = os.path.join(feat_dir, "onset_plot.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    plot_run_onset("experiments/pilot/run_w0.0")
    plot_run_onset("experiments/pilot/run_w1.0")
