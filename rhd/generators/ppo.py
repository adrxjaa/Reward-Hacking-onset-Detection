import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rhd.envs import RegionToggleEnv, RegionToggleConfig

@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 50000
    checkpoint_interval: int = 5000
    eval_episodes: int = 10
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2


class EvalAndCheckpointCallback(BaseCallback):
    def __init__(self, eval_env: RegionToggleEnv, config: PPOConfig, env_config: RegionToggleConfig, run_dir: str):
        super().__init__()
        self.eval_env = eval_env
        self.config = config
        self.env_config = env_config
        self.run_dir = run_dir
        self.metrics_file = os.path.join(run_dir, "metrics", "eval_metrics.jsonl")
        
        self.next_checkpoint = 0
        self.interval = config.checkpoint_interval
        
    def _on_step(self) -> bool:
        # Save checkpoint exactly at multiple of interval (and at 0)
        # We also need to evaluate here.
        # SB3 n_steps is per env, so num_timesteps increments by n_envs every step.
        # We might not land exactly on the interval if interval is not a multiple of n_envs.
        # But we only have 1 env here, so we land exactly on it.
        if self.num_timesteps >= self.next_checkpoint:
            self._do_checkpoint_and_eval(self.next_checkpoint)
            # Advance next_checkpoint
            while self.next_checkpoint <= self.num_timesteps:
                self.next_checkpoint += self.interval
        return True

    def _on_training_start(self) -> None:
        if self.next_checkpoint == 0:
            self._do_checkpoint_and_eval(0)
            self.next_checkpoint += self.interval

    def _do_checkpoint_and_eval(self, step: int):
        # Save checkpoint
        ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        ckpt_name = f"checkpoint_{step:08d}"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        self.model.save(ckpt_path)
        
        # Evaluate
        eval_metrics = self._evaluate(step)
        
        # Write to JSONL
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(eval_metrics) + "\n")

    def _evaluate(self, step: int) -> Dict[str, Any]:
        proxy_returns = []
        true_returns = []
        goal_success = []
        region_entries = []
        ep_lengths = []
        
        for ep in range(self.config.eval_episodes):
            obs, _ = self.eval_env.reset(seed=self.config.seed + step + ep)
            done = False
            ep_proxy = 0.0
            ep_len = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = self.eval_env.step(int(action))
                ep_proxy += r
                ep_len += 1
                done = term or trunc
                
            proxy_returns.append(ep_proxy)
            ep_lengths.append(ep_len)
            e_info = info.get("eval", {})
            true_returns.append(e_info.get("true_return", 0.0))
            goal_success.append(int(e_info.get("goal_reached", False)))
            region_entries.append(e_info.get("region_entries", 0))

        return {
            "step": step,
            "mean_proxy_return": float(np.mean(proxy_returns)),
            "mean_true_return": float(np.mean(true_returns)),
            "mean_goal_success": float(np.mean(goal_success)),
            "mean_region_entries": float(np.mean(region_entries)),
            "mean_episode_length": float(np.mean(ep_lengths)),
        }


def run_generator(run_dir: str, env_config: RegionToggleConfig, ppo_config: PPOConfig):
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    
    # Save metadata
    import stable_baselines3
    import torch
    metadata = {
        "seed": ppo_config.seed,
        "env_config": env_config.to_dict(),
        "ppo_config": asdict(ppo_config),
        "versions": {
            "python": "3.14.7", # roughly...
            "gymnasium": gym.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        }
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"env": env_config.to_dict(), "ppo": asdict(ppo_config)}, f, indent=2)

    # Train env: no eval info
    def make_env():
        e = RegionToggleEnv(env_config, expose_eval_info=False)
        e.reset(seed=ppo_config.seed)
        return e
    
    from stable_baselines3.common.vec_env import DummyVecEnv
    train_env = DummyVecEnv([make_env])
    
    # Eval env: expose eval info
    eval_env = RegionToggleEnv(env_config, expose_eval_info=True)
    
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=ppo_config.learning_rate,
        n_steps=ppo_config.n_steps,
        batch_size=ppo_config.batch_size,
        n_epochs=ppo_config.n_epochs,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        clip_range=ppo_config.clip_range,
        seed=ppo_config.seed,
        verbose=0
    )
    
    cb = EvalAndCheckpointCallback(eval_env, ppo_config, env_config, run_dir)
    
    model.learn(total_timesteps=ppo_config.total_timesteps, callback=cb)
    
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--region-reward", type=float, default=1.0)
    args = parser.parse_args()
    
    env_cfg = RegionToggleConfig(region_reward=args.region_reward)
    ppo_cfg = PPOConfig(seed=args.seed)
    
    run_generator(args.run_dir, env_cfg, ppo_cfg)
