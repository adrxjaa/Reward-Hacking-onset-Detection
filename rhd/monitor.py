"""
monitor.py - High-Level Production Interface for Reward Hacking Onset Detection

Designed for easy deployment in user RL workflows:
  from rhd import RewardHackingDetector, RewardHackingCallback
"""
from __future__ import annotations

import json
import os
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from rhd.detectors.models import LSTMQNetwork


class RewardHackingDetector:
    """
    Inference-ready Reward Hacking Onset Detector.
    
    Evaluates chronological behavioral feature vectors without requiring access
    to the true reward function during deployment.
    """

    def __init__(
        self,
        model_weights_path: Optional[str] = None,
        norm_stats_path: Optional[str] = None,
        window_size: int = 4,
        device: str = "cpu",
    ):
        self.window_size = window_size
        self.device = device
        self.model = LSTMQNetwork(input_dim=10, hidden_dim=64, num_layers=1, num_actions=2).to(device)

        # Default paths anchored inside repository if not specified
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if model_weights_path is None:
            # Check baseline weights or detector weights
            candidates = [
                os.path.join(base_dir, "results", "ablation_study", "models", "baseline_lstm_seed0.pt"),
                os.path.join(base_dir, "results", "detector", "best_model.pt"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_weights_path = c
                    break

        if norm_stats_path is None:
            norm_stats_path = os.path.join(base_dir, "results", "detector", "norm_stats.json")

        if model_weights_path and os.path.exists(model_weights_path):
            ckpt = torch.load(model_weights_path, map_location=device)
            state_dict = ckpt["online_net"] if "online_net" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            print(f"[RewardHackingDetector] Warning: Model weights not found at {model_weights_path}")

        if norm_stats_path and os.path.exists(norm_stats_path):
            with open(norm_stats_path, "r") as f:
                stats = json.load(f)
            self.mean = np.array(stats["mean"], dtype=np.float32)
            self.std = np.array(stats["std"], dtype=np.float32)
        else:
            self.mean = np.zeros(10, dtype=np.float32)
            self.std = np.ones(10, dtype=np.float32)

        self.history: List[np.ndarray] = []
        self.onset_detected: bool = False
        self.onset_step: Optional[int] = None

    def reset(self):
        """Reset internal history for a new RL monitoring session."""
        self.history.clear()
        self.onset_detected = False
        self.onset_step = None

    def push_and_predict(self, feature_vector: np.ndarray, current_step: int) -> Dict[str, Any]:
        """
        Ingest normalized 10D behavioral feature vector and predict optimal stopping decision.

        Args:
            feature_vector: 10D array of behavioral metrics.
            current_step: Current training timestep.

        Returns:
            Dict containing 'action' (0=WAIT, 1=FLAG), 'q_wait', 'q_flag', 'onset_detected'.
        """
        assert len(feature_vector) == 10, f"Expected 10-dimensional feature vector, got {len(feature_vector)}"

        norm_vec = (np.array(feature_vector, dtype=np.float32) - self.mean) / self.std
        self.history.append(norm_vec)

        # Build chronological sliding window
        W = self.window_size
        k = len(self.history)
        if k < W:
            pad = np.zeros((W - k, 10), dtype=np.float32)
            win = np.vstack([pad, np.array(self.history)])
        else:
            win = np.array(self.history[-W:])

        x = torch.as_tensor(win, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.model(x)
            q_wait = float(q[0, 0].item())
            q_flag = float(q[0, 1].item())
            action = int(torch.argmax(q, dim=-1).item())

        is_flag = (action == 1)
        if is_flag and not self.onset_detected:
            self.onset_detected = True
            self.onset_step = current_step

        return {
            "action": action,
            "action_name": "FLAG" if is_flag else "WAIT",
            "q_wait": q_wait,
            "q_flag": q_flag,
            "onset_detected": self.onset_detected,
            "onset_step": self.onset_step,
        }


class RewardHackingCallback(BaseCallback):
    """
    Stable-Baselines3 Callback for online detection of reward hacking during RL training.
    """

    def __init__(
        self,
        check_freq: int = 5000,
        detector: Optional[RewardHackingDetector] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.detector = detector or RewardHackingDetector()
        self.last_check = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_check >= self.check_freq:
            self.last_check = self.num_timesteps
            # In online training, if onset is detected, logs diagnostic alert
            if self.detector.onset_detected:
                if self.verbose > 0:
                    print(f"\n[ALERT] Reward-Hacking Detected at Step {self.detector.onset_step:,}! Policy may be exploiting proxy reward.")
        return True
