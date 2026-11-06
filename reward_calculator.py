"""Shared (cooperative) reward, kept completely separate from the environment.

    R = -w1*W - w2*E + w3*U + w4*T - w5*C

Every term is an UNBOUNDED, non-negative number divided by a configured scale (see
`rewards.norm` in the config), so all terms are roughly O(1) in a normal scenario. They are
not clipped, so heavy congestion still produces a stronger penalty.

Per CLOCK STEP (computed when the clock advances, from the state after the step):
  W = (patients waiting for a bed + for a doctor + for surgery) / norm.waiting_patients
      (summed over steps this equals total patient waiting time in units of one step)
  E = (emergency patients in those same queues) / norm.emergency_waiting_patients
  U = mean over beds, doctors, ORs of  min(busy fraction, ceiling) / ceiling
      Rewards use up to the ceiling and nothing beyond it, so agents are not pushed to fill
      every bed/doctor/OR; it is RAW busy time, not a judgement of clinical appropriateness
      (type/specialization compatibility is already guaranteed by the hard constraints).
  T = patients discharged during the step / norm.throughput_per_step
Per DECISION ROUND (computed every round):
  C = invalid actions in the round / norm.conflicts
      (with action masks the only source is two agents competing for the same resource)

Rounds in which the clock does not advance only carry the -w5*C term; the other terms are
added in the round that advances the clock. There is NO separate sum of per-agent rewards:
adding one on top would double-count these terms. Per-agent numbers are diagnostics only.
Hard safety constraints are enforced by the environment, never through these weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from src.environment.hospital_env import INVALID, ActionResult
from src.environment.hospital_state import (
    AGENT_BED, AGENT_DOCTOR, AGENT_OR, AGENTS, HospitalState, Stage,
)


@dataclass
class RewardBreakdown:
    total: float
    waiting: float = 0.0          # W   (normalised metric, before weighting)
    emergency: float = 0.0        # E
    utilization: float = 0.0      # U
    throughput: float = 0.0       # T
    conflict: float = 0.0         # C
    advanced: bool = False        # did the clock advance in this round?

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class RewardCalculator:
    def __init__(self, cfg: dict[str, Any]) -> None:
        r = cfg.get("rewards", {})
        w = r.get("weights", {})
        n = r.get("norm", {})
        c = r.get("utilization_ceiling", {})
        self.w = {"waiting": float(w.get("waiting", 1.0)), "emergency": float(w.get("emergency", 2.0)),
                  "utilization": float(w.get("utilization", 0.5)),
                  "throughput": float(w.get("throughput", 1.0)),
                  "conflict": float(w.get("conflict", 1.0))}
        self.n = {"waiting": float(n.get("waiting_patients", 20)),
                  "emergency": float(n.get("emergency_waiting_patients", 3)),
                  "throughput": float(n.get("throughput_per_step", 1.0)),
                  "conflict": float(n.get("conflicts", 1.0))}
        self.ceiling = {"beds": float(c.get("beds", 0.85)), "doctors": float(c.get("doctors", 0.9)),
                        "ors": float(c.get("ors", 0.9))}

    # ---- helpers ----------------------------------------------------------------------
    @staticmethod
    def pool_busy_fractions(state: HospitalState) -> dict[str, Optional[float]]:
        def busy(items: Sequence[Any]) -> Optional[float]:
            return sum(1 for x in items if not x.is_free) / len(items) if items else None
        return {"beds": busy(state.beds), "doctors": busy(state.doctors), "ors": busy(state.ors)}

    def utilization_score(self, state: HospitalState) -> float:
        terms = []
        for pool, frac in self.pool_busy_fractions(state).items():
            if frac is not None:
                terms.append(min(frac, self.ceiling[pool]) / self.ceiling[pool])
        return sum(terms) / len(terms) if terms else 0.0

    # ---- main entry point -------------------------------------------------------------------
    def compute(self, state: HospitalState, round_results: Sequence[ActionResult],
                advance_stats: Optional[dict[str, Any]]) -> RewardBreakdown:
        """`advance_stats` is `StepResult.stats` if the clock advanced this round, else None."""
        invalid = sum(1 for r in round_results if r.status == INVALID)
        conflict = invalid / self.n["conflict"]
        if advance_stats is None:
            return RewardBreakdown(total=-self.w["conflict"] * conflict, conflict=conflict)

        waiting = sum(advance_stats["waiting"].values()) / self.n["waiting"]
        emergency = sum(advance_stats["waiting_emergency"].values()) / self.n["emergency"]
        utilization = self.utilization_score(state)
        throughput = advance_stats["discharges"] / self.n["throughput"]
        total = (-self.w["waiting"] * waiting - self.w["emergency"] * emergency
                 + self.w["utilization"] * utilization + self.w["throughput"] * throughput
                 - self.w["conflict"] * conflict)
        return RewardBreakdown(total, waiting, emergency, utilization, throughput, conflict, True)

    # ---- per-agent diagnostics (NOT used for training) --------------------------------------
    def agent_diagnostics(self, state: HospitalState,
                          round_results: Sequence[ActionResult]) -> dict[str, dict[str, float]]:
        stage = {AGENT_BED: Stage.WAITING_BED, AGENT_DOCTOR: Stage.WAITING_DOCTOR,
                 AGENT_OR: Stage.WAITING_SURGERY}
        pool = {AGENT_BED: "beds", AGENT_DOCTOR: "doctors", AGENT_OR: "ors"}
        busy = self.pool_busy_fractions(state)
        out: dict[str, dict[str, float]] = {}
        for agent in AGENTS:
            waiting = state.patients_in_stage(stage[agent])
            mine = [r for r in round_results if r.action.agent == agent]
            out[agent] = {
                "waiting": float(len(waiting)),
                "emergency_waiting": float(sum(1 for p in waiting if p.is_emergency)),
                "pool_busy_fraction": float(busy[pool[agent]] or 0.0),
                "executed": float(sum(1 for r in mine if r.status == "executed")),
                "invalid": float(sum(1 for r in mine if r.status == INVALID)),
                "deferred": float(sum(1 for r in mine if r.status == "deferred")),
            }
        return out
