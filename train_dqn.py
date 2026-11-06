"""DQN training loop on the multi-agent hospital environment (independent learners).

Mirrors train_ppo.py so the two are comparable: same environment, reward, observations, action
masks, training-seed scheme (>= 100000, never overlapping evaluation seeds 1000+), same logging
files, and the same interaction budget (`iterations` x `steps_per_iteration` decision rounds).

Every decision round: each agent picks an epsilon-greedy feasible action, the transition
(obs, mask, action, reward, next_obs, next_mask, done) is stored in that agent's replay buffer,
and every `train_every` rounds each agent takes one gradient step on a sampled batch.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import yaml

from src.algorithms.dqn import DQNConfig, DQNLearner, epsilon_at
from src.algorithms.replay_buffer import ReplayBuffer
from src.environment.hospital_state import AGENTS
from src.environment.scenario_mix_env import make_training_env
from src.forecasting.demand_forecaster import make_forecast_provider
from src.evaluation.evaluate import flatten_summary
from src.training.train_ppo import TRAIN_SEED_OFFSET
from src.utils.logging_utils import CSVLogger
from src.utils.seeding import set_global_seed


class DQNPolicySet:
    """The three agents' DQN policies. `act` is greedy by default and never updates anything."""

    def __init__(self, learners: dict[str, DQNLearner], epsilon: float = 0.0,
                 forecast_spec: Optional[dict[str, Any]] = None) -> None:
        self.learners = learners
        self.epsilon = epsilon
        self.forecast_spec = forecast_spec

    def act(self, obs: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, int]:
        return {a: self.learners[a].act(obs[a], masks[a], self.epsilon) for a in AGENTS}


def save_dqn_checkpoint(path: Path, learners: dict[str, DQNLearner], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"learners": {a: l.state() for a, l in learners.items()}, "meta": meta}, path)


def load_dqn_policy_set(path: str | Path, epsilon: float = 0.0) -> DQNPolicySet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    learners = {}
    for agent, st in ckpt["learners"].items():
        hp = DQNConfig(**{**st["hp"], "hidden_sizes": tuple(st["hp"]["hidden_sizes"])})
        learner = DQNLearner(st["obs_dim"], st["n_actions"], hp)
        learner.load_model_state(st["model"])
        learners[agent] = learner
    return DQNPolicySet(learners, epsilon, forecast_spec=ckpt.get("meta", {}).get("forecast"))


def train_dqn(config: dict[str, Any], scenario: str | Sequence[str], out_dir: str | Path, iterations: int,
              seed: int = 0, dqn_overrides: Optional[dict[str, Any]] = None,
              checkpoint_every: int = 10, verbose: bool = True,
              forecast_spec: Optional[dict[str, Any]] = None) -> DQNPolicySet:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    torch.set_num_threads(1)

    base_seed = TRAIN_SEED_OFFSET + seed * 10_000
    ma = make_training_env(config, scenario, seed=base_seed,
                       forecast_provider=make_forecast_provider(forecast_spec))
    hp = DQNConfig.from_config(ma.env.cfg, dqn_overrides)
    learners = {a: DQNLearner(ma.obs_dims[a], ma.n_actions[a], hp, seed=seed + i)
                for i, a in enumerate(AGENTS)}
    buffers = {a: ReplayBuffer(hp.buffer_capacity, ma.obs_dims[a], ma.n_actions[a]) for a in AGENTS}

    with open(out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"scenario": scenario, "seed": seed, "iterations": iterations,
                        "forecast": forecast_spec, "dqn": {**hp.__dict__, "hidden_sizes": list(hp.hidden_sizes)},
                        "env_config": ma.env.cfg}, f, sort_keys=False)
    train_log, episode_log = CSVLogger(out / "train_log.csv"), CSVLogger(out / "episodes.csv")

    episode_idx, episode_return, total_rounds = 0, 0.0, 0
    obs, info = ma.reset(seed=base_seed + episode_idx)
    start = time.perf_counter()

    for it in range(1, iterations + 1):
        finished_returns: list[float] = []
        losses = {a: [] for a in AGENTS}
        qvals = {a: [] for a in AGENTS}

        for _ in range(hp.steps_per_iteration):
            eps = epsilon_at(hp, total_rounds)
            masks = info["masks"]
            actions = {a: learners[a].act(obs[a], masks[a], eps) for a in AGENTS}
            next_obs, rewards, terminated, next_info = ma.step(actions)
            for a in AGENTS:
                buffers[a].add(obs[a], masks[a], actions[a], rewards[a],
                               next_obs[a], next_info["masks"][a], terminated)
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
                obs, info = ma.reset(seed=base_seed + episode_idx)
            else:
                obs, info = next_obs, next_info

            if total_rounds >= hp.learning_starts and total_rounds % hp.train_every == 0:
                for a in AGENTS:
                    if len(buffers[a]) >= hp.batch_size:
                        loss, q = learners[a].update(buffers[a].sample(hp.batch_size, learners[a].rng))
                        losses[a].append(loss)
                        qvals[a].append(q)
            if total_rounds % hp.target_update_every == 0:
                for a in AGENTS:
                    learners[a].sync_target()

        row = {"iteration": it, "rounds_total": total_rounds, "episodes_total": episode_idx,
               "mean_episode_return": float(np.mean(finished_returns)) if finished_returns else float("nan"),
               "episodes_in_iteration": len(finished_returns), "epsilon": epsilon_at(hp, total_rounds),
               "elapsed_seconds": time.perf_counter() - start}
        for a in AGENTS:
            row[f"{a}_loss"] = float(np.mean(losses[a])) if losses[a] else float("nan")
            row[f"{a}_mean_q"] = float(np.mean(qvals[a])) if qvals[a] else float("nan")
        train_log.log(row)

        if verbose:
            print(f"iter {it:4d}/{iterations} | rounds {total_rounds:7d} | episodes {episode_idx:4d} | "
                  f"mean return {row['mean_episode_return']:9.2f} | eps {row['epsilon']:.2f} | "
                  f"{row['elapsed_seconds']:6.0f}s", flush=True)
        if checkpoint_every and it % checkpoint_every == 0:
            save_dqn_checkpoint(out / "checkpoints" / f"iter_{it}.pt", learners,
                                {"scenario": scenario, "seed": seed, "iteration": it,
                                 "forecast": forecast_spec})

    save_dqn_checkpoint(out / "checkpoints" / "final.pt", learners,
                        {"scenario": scenario, "seed": seed, "iteration": iterations,
                         "forecast": forecast_spec})
    return DQNPolicySet(learners, forecast_spec=forecast_spec)