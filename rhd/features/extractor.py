import json
import os
import glob
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO

from rhd.envs.region_toggle import RegionToggleEnv, RegionToggleConfig

@dataclass
class ProbeStateConfig:
    states: List[Tuple[int, int]]

def generate_probe_states(env_cfg: RegionToggleConfig) -> ProbeStateConfig:
    states = []
    for r in range(env_cfg.size):
        for c in range(env_cfg.size):
            states.append((r, c))
    return ProbeStateConfig(states=states)

class PolicyAdapter:
    def __init__(self, model):
        self.model = model

    def get_action_distribution(self, obs: np.ndarray) -> np.ndarray:
        import torch
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.squeeze(0).cpu().numpy()
        return probs

def compute_tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * np.sum(np.abs(p - q))

def extract_features_for_run(run_dir: str):
    features_dir = os.path.join(run_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "r") as f:
        config_data = json.load(f)
    env_cfg = RegionToggleConfig(**config_data["env"])
    
    probe_config = generate_probe_states(env_cfg)
    with open(os.path.join(features_dir, "probe_states.json"), "w") as f:
        json.dump(asdict(probe_config), f, indent=2)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_*.zip")))
    
    eval_env = RegionToggleEnv(env_cfg, expose_eval_info=True)
    
    schema = {
        "run_id": "string",
        "checkpoint_id": "integer",
        "environment_steps": "integer",
        "proxy_return": "float",
        "proxy_return_smoothed": "float",
        "proxy_return_change": "float",
        "episode_length": "float",
        "action_distribution_change": "float (Total Variation Distance)",
        "state_visitation_entropy": "float",
        "state_visitation_change": "float (Total Variation Distance)",
        "path_efficiency": "float",
        "revisit_rate": "float",
        "behavioral_dispersion": "float",
        "diagnostic_ppo_entropy": "float"
    }
    with open(os.path.join(features_dir, "feature_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    temporal_out = open(os.path.join(features_dir, "temporal_features.jsonl"), "w")
    offline_out = open(os.path.join(features_dir, "offline_eval.jsonl"), "w")

    prev_proxy_return = None
    prev_action_dists = None
    prev_state_visitation = None
    proxy_history = []

    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        step_str = ckpt_name.replace("checkpoint_", "").replace(".zip", "")
        step = int(step_str)

        model = PPO.load(ckpt_path)
        adapter = PolicyAdapter(model)

        dists = []
        for r, c in probe_config.states:
            d = max(env_cfg.size - 1, 1)
            obs = np.array([r / d, c / d], dtype=np.float32)
            dist = adapter.get_action_distribution(obs)
            dists.append(dist)
        
        current_action_dists = np.array(dists)
        
        if prev_action_dists is not None:
            tvds = [compute_tvd(p, q) for p, q in zip(current_action_dists, prev_action_dists)]
            action_dist_change = float(np.mean(tvds))
        else:
            action_dist_change = 0.0
            
        entropies = [-np.sum(p * np.log(p + 1e-10)) for p in current_action_dists]
        behavioral_dispersion = float(np.mean(entropies))

        num_episodes = 10
        returns_proxy = []
        lengths = []
        state_counts = np.zeros((env_cfg.size, env_cfg.size))
        
        path_efficiencies = []
        revisit_rates = []
        
        true_returns = []
        goal_successes = []
        region_entries = []

        for ep in range(num_episodes):
            obs, info = eval_env.reset(seed=42 + step + ep)
            done = False
            ep_proxy = 0.0
            ep_len = 0
            visited = set()
            trajectory = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                
                e_info = info.get("eval", {})
                pos = e_info.get("position", eval_env._pos)
                state_counts[pos[0], pos[1]] += 1
                visited.add(pos)
                trajectory.append(pos)
                
                obs, r, term, trunc, info = eval_env.step(int(action))
                ep_proxy += r
                ep_len += 1
                done = term or trunc
                
            returns_proxy.append(ep_proxy)
            lengths.append(ep_len)
            
            if ep_len > 0:
                path_efficiencies.append(len(visited) / ep_len)
                revisit_rates.append((ep_len - len(visited)) / ep_len)
            else:
                path_efficiencies.append(0.0)
                revisit_rates.append(0.0)
                
            e_info = info.get("eval", {})
            true_returns.append(e_info.get("true_return", 0.0))
            goal_successes.append(int(e_info.get("goal_reached", False)))
            region_entries.append(e_info.get("region_entries", 0))

        mean_proxy = float(np.mean(returns_proxy))
        proxy_history.append(mean_proxy)
        proxy_smoothed = float(np.mean(proxy_history[-5:]))
        
        proxy_change = 0.0
        if prev_proxy_return is not None:
            proxy_change = mean_proxy - prev_proxy_return

        total_visits = np.sum(state_counts)
        if total_visits > 0:
            state_dist = state_counts / total_visits
        else:
            state_dist = np.ones_like(state_counts) / state_counts.size
            
        visitation_entropy = float(-np.sum(state_dist * np.log(state_dist + 1e-10)))
        
        if prev_state_visitation is not None:
            visitation_change = float(compute_tvd(state_dist.flatten(), prev_state_visitation.flatten()))
        else:
            visitation_change = 0.0

        run_id = os.path.basename(os.path.normpath(run_dir))
        feature_vec = {
            "run_id": run_id,
            "checkpoint_id": step,
            "environment_steps": step,
            "proxy_return": mean_proxy,
            "proxy_return_smoothed": proxy_smoothed,
            "proxy_return_change": proxy_change,
            "episode_length": float(np.mean(lengths)),
            "action_distribution_change": action_dist_change,
            "state_visitation_entropy": visitation_entropy,
            "state_visitation_change": visitation_change,
            "path_efficiency": float(np.mean(path_efficiencies)),
            "revisit_rate": float(np.mean(revisit_rates)),
            "behavioral_dispersion": behavioral_dispersion,
            "diagnostic_ppo_entropy": behavioral_dispersion
        }
        
        temporal_out.write(json.dumps(feature_vec) + "\n")
        
        offline_eval = {
            "run_id": run_id,
            "checkpoint_id": step,
            "true_return": float(np.mean(true_returns)),
            "goal_success": float(np.mean(goal_successes)),
            "region_entries": float(np.mean(region_entries))
        }
        offline_out.write(json.dumps(offline_eval) + "\n")

        prev_proxy_return = mean_proxy
        prev_action_dists = current_action_dists
        prev_state_visitation = state_dist

    temporal_out.close()
    offline_out.close()
    eval_env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    extract_features_for_run(args.run_dir)
