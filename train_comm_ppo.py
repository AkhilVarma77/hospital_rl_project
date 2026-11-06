"""PPO training with inter-agent communication (none / mean / attention).

Same loop, seeds, logs, budget and PPO hyperparameters as train_ppo.py (independent PPO); the only
difference is the joint network with the communication module. Training episodes use seeds >=
100000 (never the evaluation seeds 1000+). For modes "mean" and "attention", train_log.csv also
records the average attention weight of every receiver on every sender (`attn_<receiver>_from_<sender>`).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import yaml

from src.algorithms.comm_ppo import CommConfig, CommPPOLearner
from src.algorithms.ppo import PPOConfig
from src.algorithms.rollout_buffer import RolloutBuffer
from src.environment.hospital_state import AGENTS
from src.environment.scenario_mix_env import make_training_env
from src.forecasting.demand_forecaster import make_forecast_provider
from src.evaluation.evaluate import flatten_summary
from src.training.train_ppo import TRAIN_SEED_OFFSET
from src.utils.logging_utils import CSVLogger
from src.utils.seeding import set_global_seed


class CommPolicySet:
    """Communicating policies for all three agents. `act` never updates anything."""

    def __init__(self, learner: CommPPOLearner, deterministic: bool = True,
                 forecast_spec: Optional[dict[str, Any]] = None) -> None:
        self.learner = learner
        self.deterministic = deterministic
        self.forecast_spec = forecast_spec

    def act(self, obs: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, int]:
        out, _ = self.learner.act(obs, masks, self.deterministic)
        return {a: out[a][0] for a in AGENTS}

    def attention_weights(self, obs: dict[str, np.ndarray], masks: dict[str, np.ndarray]):
        """[N, N] weights (rows = receivers, columns = senders, order bed/doctor/or) or None."""
        return self.learner.act(obs, masks, True)[1]


def save_comm_checkpoint(path: Path, learner: CommPPOLearner, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"learner": learner.state(), "meta": meta}, path)


def load_comm_policy_set(path: str | Path, deterministic: bool = True) -> CommPolicySet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    st = ckpt["learner"]
    hp = PPOConfig(**{**st["hp"], "hidden_sizes": tuple(st["hp"]["hidden_sizes"])})
    learner = CommPPOLearner(st["obs_dims"], st["n_actions"], hp, CommConfig(**st["comm"]))
    learner.load_model_state(st["model"])
    return CommPolicySet(learner, deterministic, forecast_spec=ckpt.get("meta", {}).get("forecast"))


def train_comm_ppo(config: dict[str, Any], scenario: str | Sequence[str], out_dir: str | Path, iterations: int,
                   seed: int = 0, mode: Optional[str] = None,
                   ppo_overrides: Optional[dict[str, Any]] = None,
                   comm_overrides: Optional[dict[str, Any]] = None,
                   checkpoint_every: int = 10, verbose: bool = True,
                   forecast_spec: Optional[dict[str, Any]] = None) -> CommPolicySet:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    torch.set_num_threads(1)

    base_seed = TRAIN_SEED_OFFSET + seed * 10_000
    ma = make_training_env(config, scenario, seed=base_seed,
                       forecast_provider=make_forecast_provider(forecast_spec))
    hp = PPOConfig.from_config(ma.env.cfg, ppo_overrides)
    overrides = dict(comm_overrides or {})
    if mode is not None:
        overrides["mode"] = mode
    cc = CommConfig.from_config(ma.env.cfg, overrides)
    learner = CommPPOLearner(ma.obs_dims, ma.n_actions, hp, cc, seed=seed)

    with open(out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"scenario": scenario, "seed": seed, "iterations": iterations,
                        "forecast": forecast_spec, "ppo": {**hp.__dict__, "hidden_sizes": list(hp.hidden_sizes)},
                        "comm": cc.__dict__, "env_config": ma.env.cfg}, f, sort_keys=False)
    train_log, episode_log = CSVLogger(out / "train_log.csv"), CSVLogger(out / "episodes.csv")

    episode_idx, episode_return, total_rounds = 0, 0.0, 0
    obs, info = ma.reset(seed=base_seed + episode_idx)
    start = time.perf_counter()

    for it in range(1, iterations + 1):
        buffers = {a: RolloutBuffer(hp.rollout_steps, ma.obs_dims[a], ma.n_actions[a]) for a in AGENTS}
        finished_returns: list[float] = []
        attn_sum, attn_n = np.zeros((len(AGENTS), len(AGENTS))), 0

        for _ in range(hp.rollout_steps):
            masks = info["masks"]
            chosen, weights = learner.act(obs, masks)
            next_obs, rewards, terminated, next_info = ma.step({a: chosen[a][0] for a in AGENTS})
            for a in AGENTS:
                act, logp, val = chosen[a]
                buffers[a].add(obs[a], masks[a], act, logp, val, rewards[a], terminated)
            if weights is not None:
                attn_sum += weights
                attn_n += 1
            episode_return += rewards[AGENTS[0]]
            total_rounds += 1
            if terminated:
                row = {"iteration": it, "episode": episode_idx, "scenario": ma.env.scenario,
                       "env_seed": ma.env.seed,
                       "episode_return": episode_return, "rounds": ma.rounds_taken}
                row.update(flatten_summary(ma.episode_summary()))
                episode_log.log(row)
                finished_returns.append(episode_return)
                episode_idx += 1
                episode_return = 0.0
                next_obs, next_info = ma.reset(seed=base_seed + episode_idx)
            obs, info = next_obs, next_info

        row = {"iteration": it, "rounds_total": total_rounds, "episodes_total": episode_idx,
               "mean_episode_return": float(np.mean(finished_returns)) if finished_returns else float("nan"),
               "episodes_in_iteration": len(finished_returns),
               "elapsed_seconds": time.perf_counter() - start}
        if attn_n:
            mean_w = attn_sum / attn_n
            for i, recv in enumerate(AGENTS):
                for j, send in enumerate(AGENTS):
                    if i != j:
                        row[f"attn_{recv}_from_{send}"] = float(mean_w[i, j])
        row.update(learner.update(buffers, learner.values(obs)))
        train_log.log(row)

        if verbose:
            print(f"iter {it:4d}/{iterations} | rounds {total_rounds:7d} | episodes {episode_idx:4d} | "
                  f"mean return {row['mean_episode_return']:9.2f} | {row['elapsed_seconds']:6.0f}s", flush=True)
        if checkpoint_every and it % checkpoint_every == 0:
            save_comm_checkpoint(out / "checkpoints" / f"iter_{it}.pt", learner,
                                 {"scenario": scenario, "seed": seed, "iteration": it, "mode": cc.mode,
                                  "forecast": forecast_spec})

    save_comm_checkpoint(out / "checkpoints" / "final.pt", learner,
                         {"scenario": scenario, "seed": seed, "iteration": iterations, "mode": cc.mode,
                          "forecast": forecast_spec})
    return CommPolicySet(learner, forecast_spec=forecast_spec)