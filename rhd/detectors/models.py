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



