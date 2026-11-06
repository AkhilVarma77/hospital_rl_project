"""Communication between agents: scaled dot-product attention over the OTHER agents' messages.

Notation: B = batch, N = number of agents (3, order bed/doctor/or), d = d_model.

    h        [B, N, d]     encoded observation (= message) of every agent
    Q_i      [B, N, d_k]   query of receiver i : Wq_i h_i      (one projection PER receiver)
    K_j      [B, N, d_k]   key of sender j     : Wk h_j        (shared)
    V_j      [B, N, d_v]   value of sender j   : Wv h_j        (shared)
    scores   [B, N, N]     s_ij = Q_i . K_j / sqrt(d_k)  (diagonal masked: no self-attention)
    weights  [B, N, N]     alpha_ij = softmax over senders j != i   (each row sums to 1)
    context  [B, N, d_v]   c_i = sum_j alpha_ij V_j

The receiving agent then uses z_i = concat(h_i, c_i) [d + d_v] in its policy and value heads.

The weights are NOT hard-coded: Wq, Wk and Wv are trained by the RL loss. Attention weights
decide how much information each agent takes from the others; reward coefficients decide how
important outcomes are - they are unrelated.

Modes (for the ablation study)
    "none"      no communication; the module is a no-op (independent agents)
    "mean"      uniform weights 1/(N-1): communication WITHOUT attention
    "attention" learned weights
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

MODES = ("none", "mean", "attention")


class CommunicationModule(nn.Module):
    def __init__(self, n_agents: int, d_model: int, d_k: int, d_v: int, mode: str) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown communication mode '{mode}', expected one of {MODES}")
        self.mode, self.n_agents, self.d_k, self.d_v = mode, n_agents, d_k, d_v
        if mode != "none":
            self.v_proj = nn.Linear(d_model, d_v)
        if mode == "attention":
            self.q_proj = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(n_agents)])
            self.k_proj = nn.Linear(d_model, d_k)
        self.register_buffer("_eye", torch.eye(n_agents, dtype=torch.bool), persistent=False)

    def forward(self, h: torch.Tensor) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """h [B, N, d] -> (context [B, N, d_v], weights [B, N, N]); (None, None) when mode == 'none'."""
        if self.mode == "none":
            return None, None
        values = self.v_proj(h)                                           # [B, N, d_v]
        if self.mode == "mean":
            others = (~self._eye).to(h.dtype)
            weights = (others / (self.n_agents - 1)).unsqueeze(0).expand(h.shape[0], -1, -1)
        else:
            queries = torch.stack([self.q_proj[i](h[:, i]) for i in range(self.n_agents)], dim=1)
            keys = self.k_proj(h)                                          # [B, N, d_k]
            scores = queries @ keys.transpose(1, 2) / math.sqrt(self.d_k)  # [B, N, N]
            scores = scores.masked_fill(self._eye, float("-inf"))          # never attend to itself
            weights = torch.softmax(scores, dim=-1)
        return weights @ values, weights                                   # [B, N, d_v], [B, N, N]
