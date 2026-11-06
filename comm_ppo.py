"""PPO with inter-agent communication (jointly trained actor-critic for the three agents).

Architecture (per decision round, all agents act simultaneously):
    obs_i --MessageEncoder_i--> h_i --CommunicationModule--> c_i
    z_i = [h_i, c_i]  (just h_i when mode == "none")
    z_i --actor head_i--> action logits (masked)       z_i --critic head_i--> V_i

Execution is therefore not fully decentralised: at decision time each agent reads the others'
messages (encoded LOCAL observations). Learning is end-to-end: the PPO losses of all agents flow
through the shared attention module into every encoder.

The PPO objective, GAE, masking and hyperparameters are exactly those of src/algorithms/ppo.py;
only the network changes, so "none" / "mean" / "attention" differ by communication alone.
The reward is the same shared reward for every agent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.algorithms.ppo import MASK_VALUE, PPOConfig, _mlp, ppo_loss
from src.algorithms.rollout_buffer import RolloutBuffer
from src.communication.attention_module import MODES, CommunicationModule
from src.communication.message_encoder import MessageEncoder
from src.environment.hospital_state import AGENTS


@dataclass
class CommConfig:
    mode: str = "attention"
    d_model: int = 64
    d_k: int = 32
    d_v: int = 32
    encoder_hidden: int = 128
    head_hidden: int = 128

    @classmethod
    def from_config(cls, cfg: dict[str, Any], overrides: Optional[dict[str, Any]] = None) -> "CommConfig":
        values = dict(cfg.get("comm", {}))
        values.update(overrides or {})
        if values.get("mode", "attention") not in MODES:
            raise ValueError(f"unknown communication mode '{values['mode']}'")
        return cls(**values)


class CommActorCritic(nn.Module):
    def __init__(self, obs_dims: dict[str, int], n_actions: dict[str, int], cc: CommConfig) -> None:
        super().__init__()
        self.cc = cc
        self.agents = list(AGENTS)
        self.encoders = nn.ModuleDict({a: MessageEncoder(obs_dims[a], cc.encoder_hidden, cc.d_model)
                                       for a in self.agents})
        self.comm = CommunicationModule(len(self.agents), cc.d_model, cc.d_k, cc.d_v, cc.mode)
        z_dim = cc.d_model + (0 if cc.mode == "none" else cc.d_v)
        self.actors = nn.ModuleDict({a: _mlp(z_dim, (cc.head_hidden,), n_actions[a], 0.01)
                                     for a in self.agents})
        self.critics = nn.ModuleDict({a: _mlp(z_dim, (cc.head_hidden,), 1, 1.0) for a in self.agents})

    def forward_all(self, obs: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]):
        """-> masked logits {agent: [B, A_i]}, values {agent: [B]}, attention weights [B, N, N] or None"""
        h = torch.stack([self.encoders[a](obs[a]) for a in self.agents], dim=1)     # [B, N, d]
        context, weights = self.comm(h)
        logits, values = {}, {}
        for i, a in enumerate(self.agents):
            z = h[:, i] if context is None else torch.cat([h[:, i], context[:, i]], dim=-1)
            raw = self.actors[a](z)
            logits[a] = torch.where(masks[a], raw, torch.full_like(raw, MASK_VALUE))
            values[a] = self.critics[a](z).squeeze(-1)
        return logits, values, weights

    def evaluate(self, obs, masks, actions):
        logits, values, weights = self.forward_all(obs, masks)
        out = {}
        for a in self.agents:
            dist = Categorical(logits=logits[a])
            out[a] = (dist.log_prob(actions[a]), dist.entropy(), values[a])
        return out, weights


def _to_batch(arrs: dict[str, np.ndarray], dtype) -> dict[str, torch.Tensor]:
    return {a: torch.as_tensor(np.asarray(v), dtype=dtype).unsqueeze(0) for a, v in arrs.items()}


class CommPPOLearner:
    """The joint model, its optimiser and the PPO update."""

    def __init__(self, obs_dims: dict[str, int], n_actions: dict[str, int], hp: PPOConfig,
                 cc: CommConfig, seed: int = 0) -> None:
        self.obs_dims, self.n_actions, self.hp, self.cc = obs_dims, n_actions, hp, cc
        self.model = CommActorCritic(obs_dims, n_actions, cc)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hp.lr, eps=1e-5)
        self._rng = np.random.default_rng(seed)

    # ---- acting ----------------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray], masks: dict[str, np.ndarray], deterministic: bool = False):
        """-> ({agent: (action, logp, value)}, attention weights [N, N] numpy or None)"""
        logits, values, weights = self.model.forward_all(_to_batch(obs, torch.float32),
                                                         _to_batch(masks, torch.bool))
        out = {}
        for a in AGENTS:
            dist = Categorical(logits=logits[a])
            action = logits[a].argmax(dim=-1) if deterministic else dist.sample()
            out[a] = (int(action.item()), float(dist.log_prob(action).item()), float(values[a].item()))
        return out, (None if weights is None else weights[0].numpy())

    @torch.no_grad()
    def values(self, obs: dict[str, np.ndarray]) -> dict[str, float]:
        masks = {a: np.ones(self.n_actions[a], bool) for a in AGENTS}
        _, values, _ = self.model.forward_all(_to_batch(obs, torch.float32), _to_batch(masks, torch.bool))
        return {a: float(values[a].item()) for a in AGENTS}

    # ---- learning --------------------------------------------------------------------------------
    def update(self, buffers: dict[str, RolloutBuffer], last_values: dict[str, float]) -> dict[str, float]:
        hp = self.hp
        T = buffers[AGENTS[0]].size
        adv, active, stats = {}, {}, {}
        for a in AGENTS:
            buf = buffers[a]
            buf.compute_gae(last_values[a], hp.gamma, hp.gae_lambda)
            idx = buf.active_indices()
            adv[a] = buf.advantages.copy()
            if hp.normalize_advantages and len(idx) > 1:
                x = adv[a][idx]
                adv[a][idx] = (x - x.mean()) / (x.std() + 1e-8)
            active[a] = buf.active.copy()
            var_ret = float(np.var(buf.returns[idx])) if len(idx) else 0.0
            stats[f"{a}_explained_variance"] = (
                1.0 - float(np.var(buf.returns[idx] - buf.values[idx])) / var_ret if var_ret > 1e-12 else 0.0)
            stats[f"{a}_active_steps"] = float(len(idx))

        rounds = np.flatnonzero(np.any([active[a] for a in AGENTS], axis=0))
        acc = {f"{a}_{k}": 0.0 for a in AGENTS
               for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}
        counts = {a: 0 for a in AGENTS}
        if len(rounds) < 2:
            return {**stats, **acc}

        obs_t = {a: torch.as_tensor(buffers[a].obs) for a in AGENTS}
        mask_t = {a: torch.as_tensor(buffers[a].masks) for a in AGENTS}
        act_t = {a: torch.as_tensor(buffers[a].actions) for a in AGENTS}
        old_t = {a: torch.as_tensor(buffers[a].logps) for a in AGENTS}
        adv_t = {a: torch.as_tensor(adv[a]) for a in AGENTS}
        ret_t = {a: torch.as_tensor(buffers[a].returns) for a in AGENTS}
        act_flag = {a: torch.as_tensor(active[a]) for a in AGENTS}

        for _ in range(hp.update_epochs):
            order = rounds.copy()
            self._rng.shuffle(order)
            for start in range(0, len(order), hp.minibatch_size):
                b = torch.as_tensor(order[start:start + hp.minibatch_size])
                out, _ = self.model.evaluate({a: obs_t[a][b] for a in AGENTS},
                                             {a: mask_t[a][b] for a in AGENTS},
                                             {a: act_t[a][b] for a in AGENTS})
                total, any_loss = 0.0, False
                for a in AGENTS:
                    sel = act_flag[a][b]
                    if not bool(sel.any()):
                        continue
                    logp, ent, val = out[a]
                    loss, info = ppo_loss(logp[sel], old_t[a][b][sel], adv_t[a][b][sel], val[sel],
                                          ret_t[a][b][sel], ent[sel], hp.clip_eps, hp.vf_coef, hp.ent_coef)
                    total = total + loss
                    any_loss = True
                    counts[a] += 1
                    for k, v in info.items():
                        acc[f"{a}_{k}"] += v
                if not any_loss:
                    continue
                self.optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), hp.max_grad_norm)
                self.optimizer.step()
        for key in acc:
            acc[key] /= max(counts[key.split("_")[0]], 1)
        return {**stats, **acc}

    # ---- persistence -----------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {"model": self.model.state_dict(), "obs_dims": dict(self.obs_dims),
                "n_actions": dict(self.n_actions), "hp": asdict(self.hp), "comm": asdict(self.cc)}

    def load_model_state(self, model_state: dict[str, Any]) -> None:
        self.model.load_state_dict(model_state)
