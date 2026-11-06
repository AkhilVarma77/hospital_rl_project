"""Shared logic for rule-based dispatch baselines.

A dispatch policy repeatedly:
  1. asks the environment for every individually FEASIBLE action of each agent,
  2. picks, per agent, the waiting patient with the best priority key (ties -> lowest resource id),
  3. commits the single best action across the three agents, then repeats,
until nothing feasible is left or each agent's per-step action limit is reached.

Because every action is committed against the live state, dispatch policies never submit an
invalid action. They are therefore a "safe, centralized" reference: subclasses differ ONLY in
`priority_key`, so differences between baselines come purely from the queueing rule.
"""
from __future__ import annotations

from typing import Optional

from src.environment.hospital_env import EXECUTED, HospitalEnv, StepResult
from src.environment.hospital_state import AGENT_RANK, AGENTS, Action, PatientRecord


class DispatchPolicy:
    """Base class. Subclasses implement `priority_key` (smaller tuple = served first)."""

    name = "dispatch"

    def priority_key(self, patient: PatientRecord) -> tuple:
        raise NotImplementedError

    # ---- policy interface used by the evaluation code --------------------------------
    def run_step(self, env: HospitalEnv) -> StepResult:
        """Make this step's decisions and advance the clock."""
        self.dispatch(env)
        return env.advance()

    # ---- implementation ----------------------------------------------------------------
    def _best_action(self, env: HospitalEnv, agent: str) -> Optional[Action]:
        feasible = env.get_feasible_actions(agent)
        if not feasible:
            return None
        patients = env.state.patients
        return min(feasible,
                   key=lambda a: (self.priority_key(patients[a.patient_id]), a.resource_id))

    def dispatch(self, env: HospitalEnv) -> None:
        patients = env.state.patients
        while True:
            best: Optional[tuple[tuple, Action]] = None
            for agent in AGENTS:
                if env.actions_remaining(agent) <= 0:
                    continue
                action = self._best_action(env, agent)
                if action is None:
                    continue
                key = (self.priority_key(patients[action.patient_id]), AGENT_RANK[agent])
                if best is None or key < best[0]:
                    best = (key, action)
            if best is None:
                return
            result = env.apply_action(best[1])
            if result.status != EXECUTED:  # cannot happen for a feasible action; avoid looping
                return
