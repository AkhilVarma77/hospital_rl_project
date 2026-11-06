"""Message encoder: turns an agent's LOCAL observation into a fixed-size message vector.

    obs_i  [B, obs_dim_i]  ->  h_i  [B, d_model]

Each agent has its own encoder because the three observation sizes differ. The encoded vector is
used twice: as the agent's own state representation and as the message other agents can read.
Only observations available at decision time go in, so nothing from the future is shared.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MessageEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden: int, d_model: int) -> None:
        super().__init__()
        l1, l2 = nn.Linear(obs_dim, hidden), nn.Linear(hidden, d_model)
        for lin in (l1, l2):
            nn.init.orthogonal_(lin.weight, np.sqrt(2))
            nn.init.zeros_(lin.bias)
        self.net = nn.Sequential(l1, nn.Tanh(), l2, nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
