import os
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

from rhd.detectors.environment import (
    DetectorEpisode,
    DetectorEnv,
    DetectorRewardConfig,
    Normalizer,
)
from rhd.detectors.agent import DQNAgent, DQNConfig
from rhd.detectors.replay_buffer import ReplayBuffer
from rhd.detectors.evaluator import DetectorEvaluator


@dataclass
class TrainerConfig:
    num_episodes: int = 300
    warmup_steps: int = 64
    eval_freq: int = 25
    window_size: int = 4
    seed: int = 42
    save_dir: str = "results/detector"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DetectorTrainer:
    """
    Orchestrates the training of the LSTM-DQN Onset Detector.
    """

    def __init__(
        self,
        train_episodes: List[DetectorEpisode],
        val_episodes: Optional[List[DetectorEpisode]] = None,
        trainer_config: Optional[TrainerConfig] = None,
        dqn_config: Optional[DQNConfig] = None,
        reward_config: Optional[DetectorRewardConfig] = None,
        normalizer: Optional[Normalizer] = None,
    ):
        self.train_episodes = train_episodes
        self.val_episodes = val_episodes or []
        self.config = trainer_config or TrainerConfig()
        self.dqn_config = dqn_config or DQNConfig()
        self.reward_config = reward_config or DetectorRewardConfig()
        self.normalizer = normalizer

        # Set seeds
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.agent = DQNAgent(config=self.dqn_config)
        self.replay_buffer = ReplayBuffer(capacity=10000)

        self.train_env = DetectorEnv(
            episodes=self.train_episodes,
            window_size=self.config.window_size,
            reward_config=self.reward_config,
            normalizer=self.normalizer,
        )

        self.evaluator = DetectorEvaluator(
            window_size=self.config.window_size,
            normalizer=self.normalizer,
        )

    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop across generator run episodes.
        """
        os.makedirs(self.config.save_dir, exist_ok=True)
        history = {
            "episode": [],
            "reward": [],
            "loss": [],
            "epsilon": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
        }

        best_val_f1 = -1.0
        best_ckpt_path = os.path.join(self.config.save_dir, "detector_best.pt")
        latest_ckpt_path = os.path.join(self.config.save_dir, "detector_latest.pt")

        for ep_idx in range(1, self.config.num_episodes + 1):
            state, info = self.train_env.reset()
            done = False
            ep_reward = 0.0
            losses = []

            while not done:
                action = self.agent.select_action(state, deterministic=False)
                next_state, reward, done, step_info = self.train_env.step(action)

                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward

                # Perform Bellman gradient update
                if len(self.replay_buffer) >= self.config.warmup_steps:
                    loss_val = self.agent.update(self.replay_buffer)
                    if loss_val is not None:
                        losses.append(loss_val)

            avg_loss = float(np.mean(losses)) if losses else 0.0
            history["episode"].append(ep_idx)
            history["reward"].append(float(ep_reward))
            history["loss"].append(avg_loss)
            history["epsilon"].append(float(self.agent.epsilon))

            val_f1 = 0.0
            val_p = 0.0
            val_r = 0.0

            # Periodic validation
            if self.val_episodes and (ep_idx % self.config.eval_freq == 0 or ep_idx == self.config.num_episodes):
                val_res = self.evaluator.evaluate_dqn(self.agent, self.val_episodes, "val")
                val_f1 = val_res.f1_score
                val_p = val_res.precision
                val_r = val_res.recall

                print(
                    f"Ep {ep_idx:>4}/{self.config.num_episodes} | "
                    f"Reward: {ep_reward:>6.2f} | Loss: {avg_loss:>7.4f} | "
                    f"Eps: {self.agent.epsilon:.3f} | "
                    f"Val F1: {val_f1:.3f} (P={val_p:.2f}, R={val_r:.2f})"
                )

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self.agent.save(best_ckpt_path)

            history["val_f1"].append(float(val_f1))
            history["val_precision"].append(float(val_p))
            history["val_recall"].append(float(val_r))

        # Always save latest
        self.agent.save(latest_ckpt_path)

        # If best checkpoint wasn't saved (e.g. no val improvement), copy latest to best
        if not os.path.exists(best_ckpt_path):
            self.agent.save(best_ckpt_path)

        # Save training history
        log_path = os.path.join(self.config.save_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        return history
