"""Training environment that mixes several scenarios.

One policy is trained on episodes drawn from many scenarios (normal, high demand, surge,
shortages, ...) and then evaluated on each scenario separately. The scenario of an episode is
chosen deterministically from its seed (`seed % number_of_scenarios`), so consecutive training
episodes cycle through the scenarios evenly and runs are reproducible.

Observation and action sizes do not depend on how many beds/doctors/ORs a scenario has, so the
same networks work in every scenario. The object exposes the same attributes as `HospitalMAEnv`
that the training loops use (obs_dims, n_actions, env, rounds_taken, reset, step, done,
episode_summary).
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from src.environment.multi_agent_env import HospitalMAEnv


class ScenarioMixEnv:
    def __init__(self, config: dict[str, Any], scenarios: Sequence[str], seed: Optional[int] = None,
                 forecast_provider: Optional[Callable[[Any], Sequence[float]]] = None) -> None:
        if not scenarios:
            raise ValueError("need at least one scenario")
        self.scenarios = list(scenarios)
        self.envs = {s: HospitalMAEnv(config, s, seed=seed, forecast_provider=forecast_provider)
                     for s in self.scenarios}
        self._active = self.envs[self.scenarios[0]]

    # ---- attributes the training loops read (delegated to the active scenario) -------------
    @property
    def active(self) -> HospitalMAEnv:
        return self._active

    @property
    def env(self):
        return self._active.env

    @property
    def obs_dims(self) -> dict[str, int]:
        return self._active.obs_dims

    @property
    def n_actions(self) -> dict[str, int]:
        return self._active.n_actions

    @property
    def agent_objs(self):
        return self._active.agent_objs

    @property
    def rounds_taken(self) -> int:
        return self._active.rounds_taken

    @property
    def done(self) -> bool:
        return self._active.done

    # ---- API -----------------------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._active = self.envs[self.scenarios[seed % len(self.scenarios)]]
        return self._active.reset(seed)

    def step(self, actions):
        return self._active.step(actions)

    def episode_summary(self) -> dict[str, Any]:
        return self._active.episode_summary()


def make_training_env(config: dict[str, Any], scenario, seed: Optional[int] = None,
                      forecast_provider=None):
    """`scenario` is one name (single-scenario training) or a list of names (mixed training)."""
    if isinstance(scenario, str):
        return HospitalMAEnv(config, scenario, seed=seed, forecast_provider=forecast_provider)
    return ScenarioMixEnv(config, scenario, seed=seed, forecast_provider=forecast_provider)
