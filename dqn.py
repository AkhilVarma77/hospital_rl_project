"""Deep Q-Network (DQN) for one agent, with action masking.

Comparable to PPO on purpose: same observations, same K+1 discrete actions, same action masks,
same shared reward, same environment and the same interaction budget. Differences are only
algorithmic: DQN is off-policy (replay buffer + target network + epsilon-greedy exploration)
while PPO is on-policy (fresh rollouts + clipped policy-gradient).

Pieces
* Q-network  Q(o, a) -> one value per action (MLP, ReLU).
* Masking    infeasible actions get Q = -1e8, both when ACTING and inside the TARGET max, so the
             agent can never choose, nor bootstrap from, an infeasible action.
* Target     y = r + gamma * (1 - done) * max_{a' feasible in o'} Q_target(o', a')   (Bellman)
* Loss       Huber(Q(o, a), y)   (smooth L1, more robust than squared error to reward spikes)
* Target net hard-copied from the online net every `target_update_every` rounds.
* Exploration  epsilon-greedy over the FEASIBLE actions; epsilon decays linearly.
This is plain DQN (no double-Q, no prioritised replay, no dueling head).
Independent learners + a replay buffer holding older data from other agents' older policies
make the problem non-stationary; this is a known weakness of independent DQN.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_VALUE = -1e8


@dataclass
class DQNConfig:
    hidden_sizes: tuple[int, ...] = (128, 128)
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_capacity: int = 50_000
    learning_starts: int = 1000
    train_every: int = 4
    target_update_every: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_rounds: int = 20_000
    max_grad_norm: float = 10.0
    steps_per_iteration: int = 1024

    @classmethod
    def from_config(cls, cfg: dict[str, Any], overrides: Optional[dict[str, Any]] = None) -> "DQNConfig":
        values = dict(cfg.get("dqn", {}))
        values.update(overrides or {})
        if "hidden_sizes" in values:
            values["hidden_sizes"] = tuple(values["hidden_sizes"])
        return cls(**values)


def epsilon_at(hp: DQNConfig, round_index: int) -> float:
    """Linear decay from epsilon_start to epsilon_end over epsilon_decay_rounds rounds."""
    frac = min(max(round_index / max(hp.epsilon_decay_rounds, 1), 0.0), 1.0)
    return hp.epsilon_start + frac * (hp.epsilon_end - hp.epsilon_start)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def masked_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, q, torch.full_like(q, MASK_VALUE))


def bellman_targets(target_net: nn.Module, rewards: torch.Tensor, next_obs: torch.Tensor,
                    next_masks: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """y = r + gamma * (1 - done) * max over FEASIBLE next actions of Q_target."""
    with torch.no_grad():
        next_q = masked_q(target_net(next_obs), next_masks).max(dim=1).values
        return rewards + gamma * (1.0 - dones) * next_q


class DQNLearner:
    def __init__(self, obs_dim: int, n_actions: int, hp: DQNConfig, seed: int = 0) -> None:
        self.obs_dim, self.n_actions, self.hp = obs_dim, n_actions, hp
        self.online = QNetwork(obs_dim, n_actions, hp.hidden_sizes)
        self.target = QNetwork(obs_dim, n_actions, hp.hidden_sizes)
        self.sync_target()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=hp.lr)
        self.rng = np.random.default_rng(seed)        # exploration and replay sampling

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    # ---- acting ---------------------------------------------------------------------------
    @torch.no_grad()
    def q_values(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        return masked_q(self.online(o), m).squeeze(0).numpy()

    def act(self, obs: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy over FEASIBLE actions. epsilon=0 is the greedy (evaluation) policy."""
        mask = np.asarray(mask, dtype=bool)
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        return int(np.argmax(self.q_values(obs, mask)))

    # ---- learning ------------------------------------------------------------------------------
    def update(self, batch: dict[str, np.ndarray]) -> tuple[float, float]:
        """One gradient step on a sampled batch. Returns (loss, mean Q of the taken actions)."""
        obs = torch.as_tensor(batch["obs"])
        actions = torch.as_tensor(batch["actions"]).unsqueeze(1)
        targets = bellman_targets(self.target, torch.as_tensor(batch["rewards"]),
                                  torch.as_tensor(batch["next_obs"]),
                                  torch.as_tensor(batch["next_masks"]),
                                  torch.as_tensor(batch["dones"]), self.hp.gamma)
        q = self.online(obs).gather(1, actions).squeeze(1)
        loss = F.smooth_l1_loss(q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.hp.max_grad_norm)
        self.optimizer.step()
        return float(loss.item()), float(q.mean().item())

    # ---- persistence -----------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {"model": self.online.state_dict(), "obs_dim": self.obs_dim,
                "n_actions": self.n_actions, "hp": asdict(self.hp)}

    def load_model_state(self, model_state: dict[str, Any]) -> None:
        self.online.load_state_dict(model_state)
        self.sync_target()
