import json
import os
import matplotlib.pyplot as plt

def plot_smoke_experiment(run_dir: str):
    features_file = os.path.join(run_dir, "features", "temporal_features.jsonl")
    if not os.path.exists(features_file):
        print("Features file not found!")
        return
        
    steps = []
    proxy_returns = []
    action_dist_changes = []
    visitation_entropies = []
    revisit_rates = []
    behavioral_dispersions = []
    
    with open(features_file, "r") as f:
        for line in f:
            feat = json.loads(line)
            steps.append(feat["environment_steps"])
            proxy_returns.append(feat["proxy_return"])
            action_dist_changes.append(feat["action_distribution_change"])
            visitation_entropies.append(feat["state_visitation_entropy"])
            revisit_rates.append(feat["revisit_rate"])
            behavioral_dispersions.append(feat["behavioral_dispersion"])
            
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    
    axs[0].plot(steps, proxy_returns, marker='o')
    axs[0].set_title("Proxy Return")
    
    axs[1].plot(steps, action_dist_changes, marker='o', color='orange')
    axs[1].set_title("Action Distribution Change (JSD)")
    
    axs[2].plot(steps, visitation_entropies, marker='o', color='green')
    axs[2].set_title("State Visitation Entropy")
    
    axs[3].plot(steps, revisit_rates, marker='o', color='red')
    axs[3].set_title("Revisit Rate")
    
    axs[4].plot(steps, behavioral_dispersions, marker='o', color='purple')
    axs[4].set_title("Behavioral Dispersion")
    axs[4].set_xlabel("Environment Steps")
    
    plt.tight_layout()
    out_path = os.path.join(run_dir, "features", "smoke_plot.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="experiments/pilot/run_w1.0")
    args = parser.parse_args()
    plot_smoke_experiment(args.run_dir)
