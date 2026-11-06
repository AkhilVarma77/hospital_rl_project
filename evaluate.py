"""Run a policy on a scenario for several seeds and collect episode metrics.

A "policy" is any object with a `run_step(env) -> StepResult` method that makes the decisions
for the current step and advances the clock. Rule-based baselines implement it now; RL policies
will implement it later, so every method is evaluated by exactly the same code.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Iterable

import pandas as pd

from src.environment.hospital_env import HospitalEnv
from src.environment.multi_agent_env import HospitalMAEnv
from src.forecasting.demand_forecaster import make_forecast_provider

KEY_METRICS = [
    "patients_arrived", "throughput", "mean_wait_to_treatment", "emergency_response_delay",
    "emergency_surgery_wait", "unserved_patients", "bed_utilization", "doctor_utilization",
    "or_utilization", "invalid_actions_total",
]


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Turn the nested per-agent dictionaries of an episode summary into flat columns."""
    flat: dict[str, Any] = {}
    for key, value in summary.items():
        if key == "invalid_reasons":
            continue
        if isinstance(value, dict):
            for agent, v in value.items():
                flat[f"{key}_{agent}"] = v
        else:
            flat[key] = value
    return flat


def run_policy_episode(env: HospitalEnv, policy: Any) -> dict[str, Any]:
    """Play one full episode and return the raw (nested) episode summary."""
    while not env.done:
        policy.run_step(env)
    return env.get_episode_summary()


def evaluate_policy(config: dict[str, Any], scenario: str, policy_factory: Callable[[], Any],
                    seeds: Iterable[int], policy_name: str) -> pd.DataFrame:
    """One row per seed. A fresh policy object is created for every episode."""
    rows = []
    for seed in seeds:
        env = HospitalEnv(config, scenario, seed=seed)
        policy = policy_factory()
        start = time.perf_counter()
        summary = run_policy_episode(env, policy)
        row = {"policy": policy_name, "scenario": scenario, "seed": seed,
               "eval_seconds": time.perf_counter() - start}
        row.update(flatten_summary(summary))
        rows.append(row)
    return pd.DataFrame(rows)


def run_ma_episode(ma_env: HospitalMAEnv, actor: Any) -> tuple[dict[str, Any], float]:
    """Play one episode with a learned policy set (`actor.act(obs, masks) -> {agent: index}`).
    Nothing is updated. Returns (episode summary, shared episode return)."""
    obs, info = ma_env.reset()
    total = 0.0
    while not ma_env.done:
        obs, rewards, _, info = ma_env.step(actor.act(obs, info["masks"]))
        total += rewards["bed"]
    return ma_env.episode_summary(), total


def evaluate_ma_policy(config: dict[str, Any], scenario: str, actor: Any,
                       seeds: Iterable[int], policy_name: str,
                       forecast_spec: Any = None) -> pd.DataFrame:
    """Evaluate a learned policy on the same simulator, metrics and seeds as the baselines.

    The forecaster is taken from `forecast_spec`, else from the policy itself (policies remember
    the forecaster they were trained with), else no forecast (zeros) is used."""
    spec = forecast_spec if forecast_spec is not None else getattr(actor, "forecast_spec", None)
    rows = []
    for seed in seeds:
        ma = HospitalMAEnv(config, scenario, seed=seed, forecast_provider=make_forecast_provider(spec))
        start = time.perf_counter()
        summary, episode_return = run_ma_episode(ma, actor)
        row = {"policy": policy_name, "scenario": scenario, "seed": seed,
               "eval_seconds": time.perf_counter() - start, "episode_return": episode_return}
        row.update(flatten_summary(summary))
        rows.append(row)
    return pd.DataFrame(rows)