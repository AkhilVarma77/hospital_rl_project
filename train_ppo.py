"""Independent-PPO training loop on the multi-agent hospital environment.

Every iteration:
  1. collect `rollout_steps` decision rounds with the CURRENT policies (sampling actions),
  2. for each agent compute GAE advantages and run several PPO epochs on its own data,
  3. append one row to train_log.csv; every finished episode appends a row to episodes.csv.

Simplifications (documented on purpose):
* An episode is a fixed 3-day horizon. Its end is treated as terminal (no bootstrapping).
* Discounting is per decision round, not per clock minute.
* Training episodes use seeds >= 100000, so they never overlap the evaluation seeds (1000+).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import yaml

from src.algorithms.ppo import PPOConfig, PPOLearner
from src.algorithms.rollout_buffer import RolloutBuffer
from src.environment.hospital_state import AGENTS
from src.environment.scenario_mix_env import make_training_env
from src.forecasting.demand_forecaster import make_forecast_provider
from src.evaluation.evaluate import flatten_summary
from src.utils.logging_utils import CSVLogger
from src.utils.seeding import set_global_seed

TRAIN_SEED_OFFSET = 100_000


class PPOPolicySet:
    """The three agents' PPO policies bundled together. `act` never updates anything."""

    def __init__(self, learners: dict[str, PPOLearner], deterministic: bool = True,
                 forecast_spec: Optional[dict[str, Any]] = None) -> None:
        self.learners = learners
        self.deterministic = deterministic
        self.forecast_spec = forecast_spec      # forecaster the policies were trained with (or None)

    def act(self, obs: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, int]:
        return {a: self.learners[a].act(obs[a], masks[a], self.deterministic)[0] for a in AGENTS}


def save_checkpoint(path: Path, learners: dict[str, PPOLearner], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"learners": {a: l.state() for a, l in learners.items()}, "meta": meta}, path)


def load_policy_set(path: str | Path, deterministic: bool = True) -> PPOPolicySet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    learners = {}
    for agent, st in ckpt["learners"].items():
        hp = PPOConfig(**{**st["hp"], "hidden_sizes": tuple(st["hp"]["hidden_sizes"])})
        learner = PPOLearner(st["obs_dim"], st["n_actions"], hp)
        learner.load_model_state(st["model"])
        learners[agent] = learner
    return PPOPolicySet(learners, deterministic, forecast_spec=ckpt.get("meta", {}).get("forecast"))


def train_ppo(config: dict[str, Any], scenario: str | Sequence[str], out_dir: str | Path, iterations: int,
              seed: int = 0, ppo_overrides: Optional[dict[str, Any]] = None,
              checkpoint_every: int = 10, verbose: bool = True,
              forecast_spec: Optional[dict[str, Any]] = None) -> PPOPolicySet:
    """Train three independent PPO agents; returns the final policies."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    torch.set_num_threads(1)                      # tiny networks: one thread is fastest

    ma = make_training_env(config, scenario, seed=TRAIN_SEED_OFFSET + seed * 10_000,
                       forecast_provider=make_forecast_provider(forecast_spec))
    hp = PPOConfig.from_config(ma.env.cfg, ppo_overrides)
    learners = {a: PPOLearner(ma.obs_dims[a], ma.n_actions[a], hp, seed=seed + i)
                for i, a in enumerate(AGENTS)}

    with open(out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"scenario": scenario, "seed": seed, "iterations": iterations,
                        "forecast": forecast_spec, "ppo": {**hp.__dict__, "hidden_sizes": list(hp.hidden_sizes)},
                        "env_config": ma.env.cfg}, f, sort_keys=False)
    train_log, episode_log = CSVLogger(out / "train_log.csv"), CSVLogger(out / "episodes.csv")

    episode_idx = 0
    obs, info = ma.reset(seed=TRAIN_SEED_OFFSET + seed * 10_000 + episode_idx)
    episode_return = 0.0
    total_rounds, start = 0, time.perf_counter()

    for it in range(1, iterations + 1):
        buffers = {a: RolloutBuffer(hp.rollout_steps, ma.obs_dims[a], ma.n_actions[a]) for a in AGENTS}
        finished_returns: list[float] = []

        for _ in range(hp.rollout_steps):
            masks = info["masks"]
            chosen = {a: learners[a].act(obs[a], masks[a]) for a in AGENTS}
            next_obs, rewards, terminated, next_info = ma.step({a: chosen[a][0] for a in AGENTS})
            for a in AGENTS:
                act, logp, val = chosen[a]
                buffers[a].add(obs[a], masks[a], act, logp, val, rewards[a], terminated)
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
                next_obs, next_info = ma.reset(seed=TRAIN_SEED_OFFSET + seed * 10_000 + episode_idx)
            obs, info = next_obs, next_info

        row = {"iteration": it, "rounds_total": total_rounds, "episodes_total": episode_idx,
               "mean_episode_return": float(np.mean(finished_returns)) if finished_returns else float("nan"),
               "episodes_in_iteration": len(finished_returns),
               "elapsed_seconds": time.perf_counter() - start}
        for a in AGENTS:
            stats = learners[a].update(buffers[a], learners[a].value(obs[a]))
            row.update({f"{a}_{k}": v for k, v in stats.items()})
        train_log.log(row)

        if verbose:
            r = row["mean_episode_return"]
            print(f"iter {it:4d}/{iterations} | rounds {total_rounds:7d} | episodes {episode_idx:4d} | "
                  f"mean return {r:9.2f} | {row['elapsed_seconds']:6.0f}s", flush=True)
        if checkpoint_every and it % checkpoint_every == 0:
            save_checkpoint(out / "checkpoints" / f"iter_{it}.pt", learners,
                            {"scenario": scenario, "seed": seed, "iteration": it, "forecast": forecast_spec})

    save_checkpoint(out / "checkpoints" / "final.pt", learners,
                    {"scenario": scenario, "seed": seed, "iteration": iterations, "forecast": forecast_spec})
    return PPOPolicySet(learners, forecast_spec=forecast_spec)