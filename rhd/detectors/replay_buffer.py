import random
from collections import deque
from typing import Tuple, List
import numpy as np
import torch


class ReplayBuffer:
    """
    Experience replay buffer for storing windowed transitions.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Store a single transition.
        state, next_state: array of shape (window_size, input_dim)
        """
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of transitions.
        """
        batch = random.sample(self.buffer, batch_size)
        return self._tensorize_batch(batch, device)

    def sample_balanced(
        self,
        batch_size: int,
        flag_ratio: float = 0.25,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch with balanced representation of FLAG actions to prevent
        class-imbalance starvation in sequential stopping problems.
        """
        flag_transitions = [t for t in self.buffer if t[1] == 1]
        wait_transitions = [t for t in self.buffer if t[1] == 0]

        n_flag = min(len(flag_transitions), max(1, int(batch_size * flag_ratio))) if flag_transitions else 0
        n_wait = batch_size - n_flag

        if n_flag > 0 and len(wait_transitions) >= n_wait:
            batch = random.sample(flag_transitions, n_flag) + random.sample(wait_transitions, n_wait)
            random.shuffle(batch)
        else:
            batch = random.sample(self.buffer, batch_size)

        return self._tensorize_batch(batch, device)

    def _tensorize_batch(
        self,
        batch: List[Tuple],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)
