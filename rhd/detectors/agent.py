from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from rhd.detectors.models import LSTMQNetwork
from rhd.detectors.replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    input_dim: int = 10
    hidden_dim: int = 64
    num_layers: int = 1
    num_actions: int = 2
    learning_rate: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 32
    target_update_freq: int = 20
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.99
    grad_norm_clip: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DQNAgent:
    """
    DQN Agent using an LSTM sequential encoder for Q-value estimation.
    """

    def __init__(
        self,
        config: Optional[DQNConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or DQNConfig()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.online_net = LSTMQNetwork(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_actions=self.config.num_actions,
        ).to(self.device)

        self.target_net = LSTMQNetwork(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_actions=self.config.num_actions,
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=self.config.learning_rate,
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stable learning

        self.epsilon = self.config.epsilon_start
        self.update_count = 0

    def select_action(
        self,
        state_window: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        Select action via epsilon-greedy policy.
        state_window: shape (window_size, input_dim)
        """
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.config.num_actions - 1)

        with torch.no_grad():
            s_tensor = torch.tensor(
                state_window,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)  # Add batch dim -> (1, W, D)
            q_vals = self.online_net(s_tensor)
            action = int(torch.argmax(q_vals, dim=-1).item())
            return action

    def get_q_values(self, state_window: np.ndarray) -> np.ndarray:
        """
        Get estimated Q-values for all actions on a single window.
        """
        with torch.no_grad():
            s_tensor = torch.tensor(
                state_window,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            q_vals = self.online_net(s_tensor)
            return q_vals.squeeze(0).cpu().numpy()

    def update(self, replay_buffer: ReplayBuffer) -> Optional[float]:
        """
        Perform one Bellman target update step on a sampled batch.
        """
        if len(replay_buffer) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(
            self.config.batch_size,
            device=self.device,
        )

        # Compute Q(s, a) using online net
        q_eval = self.online_net(states).gather(1, actions)

        # Compute max_a' Q_target(s', a')
        with torch.no_grad():
            q_next_max = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            # Bellman target: y = r + gamma * (1 - done) * max_a' Q(s', a')
            q_target = rewards + (self.config.gamma * q_next_max * (1.0 - dones))

        loss = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(
                self.online_net.parameters(),
                self.config.grad_norm_clip,
            )
        self.optimizer.step()

        self.update_count += 1

        # Synchronize target network
        if self.update_count % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )

        return float(loss.item())

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            "config": self.config.to_dict(),
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_count": self.update_count,
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.config.epsilon_min)
        self.update_count = checkpoint.get("update_count", 0)
