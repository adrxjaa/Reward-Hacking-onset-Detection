import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rhd.envs import RegionToggleConfig
from rhd.generators.ppo import PPOConfig, run_generator

def run_pilot():
    base_dir = "experiments/pilot"
    
    # Let's test three reward weights: 0.0 (control), 1.0 (some exploit), 5.0 (high exploit)
    weights = [0.0, 1.0, 5.0]
    
    for w in weights:
        print(f"--- Running Pilot w/ region_reward = {w} ---")
        run_dir = os.path.join(base_dir, f"run_w{w}")
        env_cfg = RegionToggleConfig(
            region_reward=w,
            max_steps=100
        )
        # 30k timesteps might be enough for this tiny environment
        ppo_cfg = PPOConfig(
            seed=42,
            total_timesteps=30000,
            checkpoint_interval=5000,
            eval_episodes=10,
        )
        run_generator(run_dir, env_cfg, ppo_cfg)
        
        # Print results of the last evaluation
        metrics_file = os.path.join(run_dir, "metrics", "eval_metrics.jsonl")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                lines = f.readlines()
                if lines:
                    last_eval = json.loads(lines[-1])
                    print(f"Final step={last_eval['step']} metrics: Proxy={last_eval['mean_proxy_return']:.2f}, "
                          f"True={last_eval['mean_true_return']:.2f}, Goal%={last_eval['mean_goal_success']:.2f}, "
                          f"Entries={last_eval['mean_region_entries']:.2f}")
        print("\n")

if __name__ == "__main__":
    run_pilot()
