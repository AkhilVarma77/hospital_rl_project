"""Multi-agent wrapper: the interface the PPO / DQN code will train against.

    obs, info = env.reset(seed)
    obs, rewards, terminated, info = env.step({"bed": a_bed, "doctor": a_doc, "or": a_or})

* `obs[agent]`        float32 vector (see src/agents/base_agent.py)
* `info["masks"]`     boolean action masks per agent (K+1 actions, defer always allowed)
* `rewards[agent]`    the SAME shared reward for every agent (cooperative)
* `terminated`        True once the episode horizon is reached

One call to `step` is one DECISION ROUND: all three agents choose simultaneously from the same
state, the actions are validated/executed by the environment (urgent patient wins conflicts).
The CLOCK ONLY ADVANCES when every agent is "done for this step", i.e. it either
  * deferred voluntarily (it had feasible options but chose to wait), or
  * has no feasible non-defer action left (nothing to do, or its action budget is used up).
So agents get as many decision rounds per clock step as they need (bounded by the per-agent
action limit), which gives them the same decision capacity as the rule-based baselines.
An agent that is merely forced to defer (nothing feasible yet) can still act in a later round
of the same step, e.g. to treat a patient the bed agent has just admitted.

Placeholder: `forecast_provider` supplies the 2 forecast features. Until the forecasting module
exists (a later step) they are zeros.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from src.agents.bed_agent import BedAgent
from src.agents.doctor_agent import DoctorAgent
from src.agents.or_agent import ORAgent
from src.environment.hospital_env import HospitalEnv
from src.environment.hospital_state import AGENTS
from src.rewards.reward_calculator import RewardCalculator


class HospitalMAEnv:
    agents = AGENTS

    def __init__(self, config: dict[str, Any], scenario: str = "normal",
                 seed: Optional[int] = None,
                 forecast_provider: Optional[Callable[[HospitalEnv], Sequence[float]]] = None) -> None:
        self.env = HospitalEnv(config, scenario, seed=seed)
        cfg = self.env.cfg
        self.agent_objs = {a.name: a for a in (BedAgent(cfg), DoctorAgent(cfg), ORAgent(cfg))}
        self.reward_calc = RewardCalculator(cfg)
        self.forecast_provider = forecast_provider
        self.obs_dims = {n: a.obs_dim for n, a in self.agent_objs.items()}
        self.n_actions = {n: a.n_actions for n, a in self.agent_objs.items()}
        self.rounds_taken = 0
        self._finished = {a: False for a in AGENTS}
        self._masks: dict[str, np.ndarray] = {}

    # ---- helpers -----------------------------------------------------------------------------
    def _forecast(self) -> Sequence[float]:
        return self.forecast_provider(self.env) if self.forecast_provider else (0.0, 0.0)

    def _observe(self) -> dict[str, np.ndarray]:
        f = self._forecast()
        return {n: a.observe(self.env, f) for n, a in self.agent_objs.items()}

    def _compute_masks(self) -> dict[str, np.ndarray]:
        return {n: a.action_mask(self.env, self._finished[n]) for n, a in self.agent_objs.items()}

    # ---- API ---------------------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.env.reset(seed)
        self.rounds_taken = 0
        self._finished = {a: False for a in AGENTS}
        self._masks = self._compute_masks()
        return self._observe(), {"masks": self._masks, "time": self.env.state.time}

    def step(self, actions: Mapping[str, int]):
        if self.env.done:
            raise RuntimeError("episode is finished; call reset()")
        missing = set(AGENTS) - set(actions)
        if missing:
            raise ValueError(f"missing actions for agents: {sorted(missing)}")

        # 1. decode every agent's choice from the SAME state, then execute them as one round
        decoded = {n: [self.agent_objs[n].decode(self.env, int(actions[n]))] for n in AGENTS}
        results = self.env.execute_round(decoded)
        self.rounds_taken += 1

        # 2. a voluntary defer (while options existed) ends that agent's turn for this step
        for n in AGENTS:
            if int(actions[n]) == self.agent_objs[n].defer_index and self._masks[n][:-1].any():
                self._finished[n] = True

        # 3. advance the clock only when nobody can or wants to act any more
        masks = self._compute_masks()
        all_done = all(self._finished[n] or not masks[n][:-1].any() for n in AGENTS)
        advance_stats = None
        if all_done:
            advance_stats = self.env.advance().stats
            self._finished = {a: False for a in AGENTS}
            masks = self._compute_masks()
        self._masks = masks

        # 4. shared reward + diagnostics
        breakdown = self.reward_calc.compute(self.env.state, results, advance_stats)
        rewards = {n: breakdown.total for n in AGENTS}
        info = {
            "masks": masks,
            "time": self.env.state.time,
            "clock_advanced": advance_stats is not None,
            "round_results": results,
            "reward_components": breakdown.as_dict(),
            "agent_metrics": self.reward_calc.agent_diagnostics(self.env.state, results),
        }
        if advance_stats is not None:
            info["step_stats"] = advance_stats
        return self._observe(), rewards, self.env.done, info

    @property
    def done(self) -> bool:
        return self.env.done

    def episode_summary(self) -> dict[str, Any]:
        return self.env.get_episode_summary()
