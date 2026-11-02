"""
unified.py - Unified multi-algorithm generator supporting PPO, A2C, and DQN
across RegionToggle, CoinCollector, and BatteryCollector environments.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Union

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rhd.envs import (
    create_environment,
    create_env_config,
    RegionToggleEnv,
    RegionToggleConfig,
    CoinCollectorEnv,
    CoinCollectorConfig,
    BatteryCollectorEnv,
    BatteryCollectorConfig,
)


@dataclass
class GeneratorConfig:
    algo: str = "ppo"                # "ppo", "a2c", or "dqn"
    seed: int = 42
    total_timesteps: int = 30000
    checkpoint_interval: int = 5000
    eval_episodes: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99

    # PPO specific
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2

    # DQN specific
    buffer_size: int = 10000
    learning_starts: int = 250
    target_update_interval: int = 250
    exploration_fraction: float = 0.3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05

    # A2C specific
    a2c_n_steps: int = 32
    ent_coef: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)


class UnifiedEvalAndCheckpointCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        config: GeneratorConfig,
        env_config: Any,
        run_dir: str,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.config = config
        self.env_config = env_config
        self.run_dir = run_dir
        self.metrics_file = os.path.join(run_dir, "metrics", "eval_metrics.jsonl")
        self.next_checkpoint = 0
        self.interval = config.checkpoint_interval

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_checkpoint:
            self._do_checkpoint_and_eval(self.next_checkpoint)
            while self.next_checkpoint <= self.num_timesteps:
                self.next_checkpoint += self.interval
        return True

    def _on_training_start(self) -> None:
        if self.next_checkpoint == 0:
            self._do_checkpoint_and_eval(0)
            self.next_checkpoint += self.interval

    def _do_checkpoint_and_eval(self, step: int):
        ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        ckpt_name = f"checkpoint_{step:08d}"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        self.model.save(ckpt_path)

        eval_metrics = self._evaluate(step)

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(eval_metrics) + "\n")

        print(
            f"  [{self.config.algo.upper()}] {step:>6,} steps | "
            f"Proxy: {eval_metrics['mean_proxy_return']:>6.2f} | "
            f"True: {eval_metrics['mean_true_return']:>4.1f} | "
            f"Goal%: {eval_metrics['mean_goal_success']*100:>5.1f}% | "
            f"Exploits: {eval_metrics['mean_region_entries']:>4.1f}",
            flush=True,
        )

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
            # Generalized exploit metric
            exploits = e_info.get("region_entries", e_info.get("coin_visits", e_info.get("battery_visits", 0)))
            region_entries.append(exploits)

        return {
            "step": step,
            "mean_proxy_return": float(np.mean(proxy_returns)),
            "mean_true_return": float(np.mean(true_returns)),
            "mean_goal_success": float(np.mean(goal_success)),
            "mean_region_entries": float(np.mean(region_entries)),
            "mean_episode_length": float(np.mean(ep_lengths)),
        }


def build_rl_model(algo: str, env: gym.Env, cfg: GeneratorConfig):
    algo_clean = algo.lower()
    if algo_clean == "dqn":
        return DQN(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            batch_size=cfg.batch_size,
            target_update_interval=cfg.target_update_interval,
            exploration_fraction=cfg.exploration_fraction,
            exploration_initial_eps=cfg.exploration_initial_eps,
            exploration_final_eps=cfg.exploration_final_eps,
            gamma=cfg.gamma,
            seed=cfg.seed,
            verbose=0,
        )
    elif algo_clean == "a2c":
        return A2C(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.a2c_n_steps,
            gamma=cfg.gamma,
            ent_coef=cfg.ent_coef,
            seed=cfg.seed,
            verbose=0,
        )
    elif algo_clean == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            seed=cfg.seed,
            verbose=0,
        )
    else:
        raise ValueError(f"Unknown RL algorithm: {algo}. Must be 'ppo', 'a2c', or 'dqn'.")


def run_unified_generator(
    run_dir: str,
    env_config: Any,
    gen_config: Optional[GeneratorConfig] = None,
):
    gen_config = gen_config or GeneratorConfig()
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    metadata = {
        "seed": gen_config.seed,
        "algo": gen_config.algo,
        "env_config": env_config.to_dict(),
        "generator_config": gen_config.to_dict(),
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(
            {
                "env": env_config.to_dict(),
                "algo": gen_config.algo,
                "generator": gen_config.to_dict(),
                "ppo": gen_config.to_dict(),  # for backward compatibility
            },
            f,
            indent=2,
        )

    # Training environment (strictly quarantined)
    def make_train_env():
        e = create_environment(env_config, expose_eval_info=False)
        e.reset(seed=gen_config.seed)
        return e

    train_env = DummyVecEnv([make_train_env])

    # Evaluation environment (exposes eval info)
    eval_env = create_environment(env_config, expose_eval_info=True)

    model = build_rl_model(gen_config.algo, train_env, gen_config)
    cb = UnifiedEvalAndCheckpointCallback(eval_env, gen_config, env_config, run_dir)

    model.learn(total_timesteps=gen_config.total_timesteps, callback=cb)

    train_env.close()
    eval_env.close()
