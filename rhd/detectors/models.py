import torch
import torch.nn as nn


class LSTMQNetwork(nn.Module):
    """
    Recurrent Q-Network combining an LSTM sequential encoder with a Q-value head.
    
    Processes rolling window observations of shape (batch_size, window_size, input_dim)
    and predicts action-values Q(s, WAIT) and Q(s, FLAG).
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_actions: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_actions = num_actions

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, window_size, input_dim)
               or (window_size, input_dim)
               
        Returns:
            Tensor of shape (batch_size, num_actions)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension

        # out shape: (batch_size, window_size, hidden_dim)
        out, (hn, cn) = self.lstm(x)

        # Extract the representation from the final step of the window
        last_hidden = out[:, -1, :]

        # Compute Q-values for all actions
        q_values = self.q_head(last_hidden)
        return q_values


class MLPQNetwork(nn.Module):
    """
    Ablation baseline: Non-recurrent feed-forward Q-Network without an LSTM sequence encoder.
    Operates strictly on the current checkpoint feature vector, ablating temporal recurrence.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        num_actions: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            if x.shape[-1] == self.input_dim and x.shape[0] > 1:
                # (W, D) single window -> take last frame
                x = x[-1:, :]
            elif x.shape[-1] == self.input_dim and x.shape[0] == 1:
                pass
            else:
                x = x.unsqueeze(0)
        elif x.dim() == 3:
            # (B, W, D) batched windows -> extract current (last) frame
            x = x[:, -1, :]

        return self.net(x)


class ImprovedLSTMQNetwork(nn.Module):
    """
    Improved Recurrent Dueling Q-Network for sequential reward hacking onset detection.
    Features:
    - Dual pooling sequence aggregation: concatenates last step h_W and temporal mean pool h_mean.
    - Dueling DQN architecture: separate state value V(s) and advantage A(s, a) heads.
    - Optional multi-layer recurrent stack with dropout.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_actions: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_actions = num_actions

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dual pooling produces [last_hidden, mean_hidden] of size 2 * hidden_dim
        agg_dim = hidden_dim * 2

        # State value stream V(s)
        self.v_head = nn.Sequential(
            nn.Linear(agg_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage stream A(s, a)
        self.a_head = nn.Sequential(
            nn.Linear(agg_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, W, D)

        out, _ = self.lstm(x)  # (B, W, H)
        last_hidden = out[:, -1, :]  # (B, H)
        mean_hidden = out.mean(dim=1)  # (B, H)
        agg = torch.cat([last_hidden, mean_hidden], dim=-1)  # (B, 2*H)

        v = self.v_head(agg)  # (B, 1)
        a = self.a_head(agg)  # (B, num_actions)

        # Dueling formula: Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))
        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q


class ImprovedMLPQNetwork(nn.Module):
    """
    Improved Non-Recurrent Dueling Q-Network baseline.
    Flattens the temporal window (W * D) with residual LayerNorm blocks and dueling heads.
    """

    def __init__(
        self,
        input_dim: int = 16,
        window_size: int = 4,
        hidden_dim: int = 64,
        num_actions: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        total_input_dim = input_dim * window_size

        self.encoder = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # State value stream V(s)
        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Advantage stream A(s, a)
        self.a_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, W, D)

        B, W, D = x.shape
        if W != self.window_size:
            if W < self.window_size:
                pad = torch.zeros(B, self.window_size - W, D, device=x.device, dtype=x.dtype)
                x = torch.cat([pad, x], dim=1)
            else:
                x = x[:, -self.window_size:, :]
        flat = x.reshape(B, -1)
        feat = self.encoder(flat)

        v = self.v_head(feat)
        a = self.a_head(feat)

        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q


