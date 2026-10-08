import json
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon
from stable_baselines3 import PPO

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # Jensen-Shannon divergence
    # Ensure they sum to 1 and have no negatives
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(jensenshannon(p, q, base=2))

def calculate_state_visitation_entropy(visitation: Dict[str, float]) -> float:
    counts = np.array(list(visitation.values()), dtype=np.float32)
    if len(counts) == 0:
        return 0.0
    p = counts / np.sum(counts)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

class FeatureExtractor:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        
        # Load metadata
        with open(os.path.join(run_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
            
        self.seed = self.metadata["seed"]
        self.env_config = self.metadata["env_config"]
        
        # We need a stable random number generator for probe states
        self.rng = np.random.default_rng(self.seed)
        
        self.probe_states = self._generate_probe_states()
        
    def _generate_probe_states(self, n_states: int = 20) -> np.ndarray:
        # Generate random valid states in the environment.
        # Since obs is [row/d, col/d], where d is max(size-1, 1), we can sample grid coords.
        size = self.env_config.get("size", 6)
        d = max(size - 1, 1)
        states = []
        for _ in range(n_states):
            r = self.rng.integers(0, size)
            c = self.rng.integers(0, size)
            states.append([r / d, c / d])
        return np.array(states, dtype=np.float32)

    def extract_checkpoint(self, ckpt_path: str, offline_metrics: Dict[str, Any], prev_action_dist: Optional[np.ndarray], prev_state_visitation: Optional[np.ndarray]) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        # Load the model
        model = PPO.load(ckpt_path, device="cpu")
        
        # Action distribution at probe states
        n_samples = 30
        n_probes = len(self.probe_states)
        n_actions = 4
        
        action_counts = np.zeros((n_probes, n_actions), dtype=np.float32)
        for _ in range(n_samples):
            actions, _ = model.predict(self.probe_states, deterministic=False)
            for i, a in enumerate(actions):
                action_counts[i, int(a)] += 1
                
        action_dist = action_counts / n_samples
        
        # Behavioral dispersion
        dispersion_per_state = []
        for i in range(n_probes):
            p = np.clip(action_dist[i], 1e-10, 1.0)
            p = p / np.sum(p)
            dispersion_per_state.append(-np.sum(p * np.log2(p)))
        behavioral_dispersion = float(np.mean(dispersion_per_state))
        
        flat_action_dist = action_dist.flatten()
        action_distribution_change = 0.0
        if prev_action_dist is not None:
            action_distribution_change = js_divergence(flat_action_dist, prev_action_dist)
            
        # State visitation through rollouts
        # Size of grid = 6
        from rhd.envs import RegionToggleEnv, RegionToggleConfig
        eval_env = RegionToggleEnv(RegionToggleConfig(**self.env_config), expose_eval_info=True)
        size = self.env_config.get("size", 6)
        state_counts = np.zeros((size, size), dtype=np.float32)
        
        for ep in range(10):
            obs, _ = eval_env.reset(seed=self.seed + ep)
            done = False
            while not done:
                # Discretize obs
                r = int(round(obs[0] * max(size - 1, 1)))
                c = int(round(obs[1] * max(size - 1, 1)))
                state_counts[r, c] += 1
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = eval_env.step(int(action))
                done = term or trunc
                
        state_visitation = state_counts.flatten()
        total_visits = np.sum(state_visitation)
        state_visitation_dist = state_visitation / total_visits if total_visits > 0 else state_visitation
        
        # Visitation entropy
        p = np.clip(state_visitation_dist, 1e-10, 1.0)
        p = p[p > 1e-10]
        visitation_entropy = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0
        
        visitation_change = 0.0
        if prev_state_visitation is not None:
            visitation_change = js_divergence(state_visitation_dist, prev_state_visitation)
            
        # Offline metrics
        proxy_return = offline_metrics.get("mean_proxy_return", 0.0)
        ep_length = offline_metrics.get("mean_episode_length", 0.0)
        region_entries = offline_metrics.get("mean_region_entries", 0.0)
        revisit_rate = region_entries / max(ep_length, 1.0)
        path_efficiency = 10.0 / ep_length if ep_length > 0 else 0.0
            
        features = {
            "checkpoint_id": os.path.basename(ckpt_path),
            "environment_steps": offline_metrics.get("step", 0),
            "proxy_return": float(proxy_return),
            "episode_length": float(ep_length),
            "action_distribution_change": action_distribution_change,
            "state_visitation_entropy": float(visitation_entropy),
            "state_visitation_change": float(visitation_change),
            "path_efficiency": float(path_efficiency),
            "revisit_rate": float(revisit_rate),
            "behavioral_dispersion": float(behavioral_dispersion),
        }
        
        return features, flat_action_dist, state_visitation_dist

    def run_extraction(self):
        # Read eval metrics
        metrics_file = os.path.join(self.run_dir, "metrics", "eval_metrics.jsonl")
        metrics_by_step = {}
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                for line in f:
                    m = json.loads(line)
                    metrics_by_step[m["step"]] = m
                    
        ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            return
            
        ckpts = sorted([c for c in os.listdir(ckpt_dir) if c.endswith(".zip")])
        
        out_features = []
        out_offline = []
        
        prev_action_dist = None
        prev_state_vis = None
        
        for ckpt in ckpts:
            step_str = ckpt.replace("checkpoint_", "").replace(".zip", "")
            step = int(step_str)
            
            ckpt_path = os.path.join(ckpt_dir, ckpt)
            m = metrics_by_step.get(step, {})
            
            features, action_dist, state_vis = self.extract_checkpoint(ckpt_path, m, prev_action_dist, prev_state_vis)
            prev_action_dist = action_dist
            prev_state_vis = state_vis
            
            # Smooth proxy return
            if len(out_features) > 0:
                features["proxy_return_smoothed"] = 0.8 * out_features[-1]["proxy_return_smoothed"] + 0.2 * features["proxy_return"]
            else:
                features["proxy_return_smoothed"] = features["proxy_return"]
                
            features["proxy_return_trend"] = features["proxy_return_smoothed"] - (out_features[-1]["proxy_return_smoothed"] if out_features else features["proxy_return_smoothed"])
            
            out_features.append(features)
            
            # Offline eval info (NEVER IN FEATURES)
            out_offline.append({
                "checkpoint_id": ckpt,
                "environment_steps": step,
                "true_return": m.get("mean_true_return", 0.0),
                "goal_success": m.get("mean_goal_success", 0.0),
                "region_entries": m.get("mean_region_entries", 0.0),
            })
            
        # Write outputs
        feat_dir = os.path.join(self.run_dir, "features")
        os.makedirs(feat_dir, exist_ok=True)
        
        with open(os.path.join(feat_dir, "temporal_features.jsonl"), "w") as f:
            for feat in out_features:
                f.write(json.dumps(feat) + "\n")
                
        with open(os.path.join(feat_dir, "offline_eval.jsonl"), "w") as f:
            for off in out_offline:
                f.write(json.dumps(off) + "\n")
                
        with open(os.path.join(feat_dir, "probe_states.json"), "w") as f:
            json.dump({"states": self.probe_states.tolist()}, f, indent=2)
            
        schema = {k: type(v).__name__ for k, v in out_features[0].items()}
        with open(os.path.join(feat_dir, "feature_schema.json"), "w") as f:
            json.dump(schema, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    
    extractor = FeatureExtractor(args.run_dir)
    extractor.run_extraction()
