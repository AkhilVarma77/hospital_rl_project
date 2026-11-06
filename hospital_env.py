"""The shared hospital simulation (single source of truth).

Time model
----------
* The clock advances in fixed steps of `simulation.step_minutes` minutes.
* At decision time t the agents see all patients with arrival_time <= t.
* `step(actions)` (1) validates and executes the submitted actions at time t, then
  (2) advances the clock to t + step and processes completions, discharges, arrivals and
  outages. Resources finishing during the step become free at the NEW decision time.

Joint action execution (deterministic, documented)
--------------------------------------------------
All actions from all agents in one step are sorted by
    (patient urgency, highest first; agent order bed < doctor < or; patient_id; submission index)
and validated/executed ONE AT A TIME against the live state. Therefore:
* a patient admitted by the bed action can be treated in the same step,
* when two actions compete for the same resource the more urgent patient wins,
* the loser is rejected as INVALID (state unchanged) and counted.

Two ways to drive the environment
---------------------------------
* `execute_round(actions)` then `advance()` (or `step(actions)` = both): batch interface; RL
  agents act simultaneously on the same state. Several rounds may precede one `advance()`.
* `apply_action(a)` x N, then `advance()` : sequential interface (rule-based dispatchers commit
  one action at a time against the live state, so they never create joint conflicts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from src.data.synthetic_data_generator import SyntheticDataGenerator
from src.environment import constraints as C
from src.environment.hospital_events import apply_outages
from src.environment.hospital_state import (
    AGENT_BED, AGENT_DOCTOR, AGENT_OR, AGENT_RANK, AGENTS, Action, BedState,
    DoctorState, HospitalState, ORState, PatientRecord, Stage,
)
from src.evaluation.metrics import compute_episode_metrics

EXECUTED, DEFERRED, INVALID = "executed", "deferred", "invalid"


@dataclass
class ActionResult:
    action: Action
    status: str                 # executed | deferred | invalid
    reason: str = ""            # reason code when invalid


@dataclass
class StepResult:
    time: int                   # clock AFTER the step
    done: bool
    action_results: list[ActionResult]
    stats: dict[str, Any] = field(default_factory=dict)


class HospitalEnv:
    """Custom multi-agent hospital environment (no learning code inside)."""

    def __init__(self, config: dict[str, Any], scenario: str = "normal",
                 seed: Optional[int] = None,
                 data: Optional[dict[str, pd.DataFrame]] = None) -> None:
        self.scenario = scenario
        self._config = config
        self._generator = SyntheticDataGenerator(config, scenario, seed)
        self.cfg = self._generator.cfg
        self.seed = self._generator.seed
        self.step_minutes = self._generator.step_minutes
        self.horizon = self._generator.horizon_minutes
        self.rules = C.ClinicalRules.from_config(self.cfg)
        self.max_actions = int(self.cfg.get("environment", {})
                               .get("max_actions_per_agent_per_step", 3))
        self._fixed_data = data
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None) -> HospitalState:
        """Start a new episode. A new seed regenerates the synthetic patient data."""
        if seed is not None and self._fixed_data is None:
            self._generator = SyntheticDataGenerator(self._config, self.scenario, seed)
            self.seed = self._generator.seed
        data = self._fixed_data if self._fixed_data is not None else self._generator.generate_all()
        self._events = data["events"]

        state = HospitalState()
        for row in data["beds"].to_dict("records"):
            state.beds.append(BedState(int(row["bed_id"]), str(row["bed_type"])))
        for row in data["doctors"].to_dict("records"):
            state.doctors.append(DoctorState(int(row["doctor_id"]), str(row["specialization"])))
        for row in data["operating_rooms"].to_dict("records"):
            types = frozenset(str(row["supported_surgery_types"]).split("|"))
            state.ors.append(ORState(int(row["or_id"]), types))
        for lst, key in ((state.beds, "bed_id"), (state.doctors, "doctor_id"), (state.ors, "or_id")):
            assert [getattr(x, key) for x in lst] == list(range(len(lst))), "resource ids must be 0..n-1"

        self._pending = [PatientRecord.from_row(r) for r in
                         data["patients"].sort_values(["arrival_time", "patient_id"]).to_dict("records")]
        self._next_arrival = 0
        self.state = state
        self.action_counts = {a: {EXECUTED: 0, DEFERRED: 0, INVALID: 0} for a in AGENTS}
        self.invalid_reasons: dict[str, dict[str, int]] = {a: {} for a in AGENTS}
        self.steps_taken = 0
        self._step_results: list[ActionResult] = []
        self._step_counts = {a: 0 for a in AGENTS}
        apply_outages(self.state, self._events)
        self._admit_arrivals()
        return self.state

    @property
    def done(self) -> bool:
        return self.state.time >= self.horizon

    # --------------------------------------------------------------- queries
    def validate(self, action: Action) -> Optional[str]:
        """Check an action against the CURRENT state without executing it."""
        return C.validate_action(self.state, self.rules, action)

    def get_feasible_actions(self, agent: str) -> list[Action]:
        return C.feasible_actions(self.state, self.rules, agent)

    # ------------------------------------------------------------------ step
    def step(self, actions: Mapping[str, Sequence[Action]]) -> StepResult:
        """Batch interface: one decision round, then advance the clock."""
        self.execute_round(actions)
        return self.advance()

    def execute_round(self, actions: Mapping[str, Sequence[Action]]) -> list[ActionResult]:
        """One decision round WITHOUT advancing the clock; returns this round's results.

        Actions are validated against the state at the start of the round, so two individually
        feasible actions can still conflict; the more urgent one wins (see module docstring).
        Explicit defers never count toward the per-agent action limit; every other submitted
        action does (limit = `max_actions_per_agent_per_step`, shared by all rounds of a step).
        Several rounds may be played before `advance()` is called.
        """
        if self.done:
            raise RuntimeError("episode is finished; call reset()")
        unknown = set(actions) - set(AGENTS)
        if unknown:
            raise ValueError(f"unknown agent keys: {sorted(unknown)}")

        first = len(self._step_results)
        queue: list[tuple[tuple, Action]] = []
        for agent in AGENTS:
            for idx, act in enumerate(actions.get(agent, [])):
                if act.agent != agent:
                    self._record(ActionResult(act, INVALID, C.WRONG_AGENT))
                    continue
                if not act.is_defer:
                    if idx >= self.max_actions or self._step_counts[agent] >= self.max_actions:
                        self._record(ActionResult(act, INVALID, C.TOO_MANY_ACTIONS))
                        continue
                    self._step_counts[agent] += 1
                p = self.state.patients.get(act.patient_id) if not act.is_defer else None
                urgency = p.urgency_level if p is not None else 0
                queue.append(((-urgency, AGENT_RANK[agent],
                               -1 if act.is_defer else act.patient_id, idx), act))
        queue.sort(key=lambda item: item[0])
        for _, act in queue:
            self._process(act)
        return self._step_results[first:]

    # ---- sequential interface (used by the rule-based baselines) --------------------
    def actions_remaining(self, agent: str) -> int:
        """How many more non-defer actions `agent` may still submit in the current step."""
        return self.max_actions - self._step_counts[agent]

    def apply_action(self, act: Action) -> ActionResult:
        """Validate and execute ONE action immediately at the current time (no clock advance).

        Call `advance()` after the last action of the step. Because each action is checked
        against the live state, a caller that commits actions one by one never produces a
        joint conflict. The per-agent action limit still applies (defers are free).
        """
        if self.done:
            raise RuntimeError("episode is finished; call reset()")
        if act.agent not in AGENTS:
            raise ValueError(f"unknown agent '{act.agent}'")
        if not act.is_defer:
            if self._step_counts[act.agent] >= self.max_actions:
                return self._record(ActionResult(act, INVALID, C.TOO_MANY_ACTIONS))
            self._step_counts[act.agent] += 1
        return self._process(act)

    def advance(self) -> StepResult:
        """Advance the clock by one step and return the results of this step's actions."""
        if self.done:
            raise RuntimeError("episode is finished; call reset()")
        results, self._step_results = self._step_results, []
        stats = self._advance_time()
        for key, status in (("executed", EXECUTED), ("invalid", INVALID), ("deferred", DEFERRED)):
            stats[key] = {a: sum(1 for r in results if r.action.agent == a and r.status == status)
                          for a in AGENTS}
        self._step_counts = {a: 0 for a in AGENTS}
        self.steps_taken += 1
        return StepResult(self.state.time, self.done, results, stats)

    def _record(self, result: ActionResult) -> ActionResult:
        self._step_results.append(result)
        agent = result.action.agent
        self.action_counts[agent][result.status] += 1
        if result.status == INVALID:
            bucket = self.invalid_reasons[agent]
            bucket[result.reason] = bucket.get(result.reason, 0) + 1
        return result

    def _process(self, act: Action) -> ActionResult:
        """Defer, reject or execute one action against the live state, and record the outcome."""
        if act.is_defer:
            return self._record(ActionResult(act, DEFERRED))
        reason = C.validate_action(self.state, self.rules, act)
        if reason is not None:
            return self._record(ActionResult(act, INVALID, reason))
        self._execute(act)
        return self._record(ActionResult(act, EXECUTED))

    # ------------------------------------------------------------- execution
    def _execute(self, a: Action) -> None:
        t, s = self.state.time, self.state
        p = s.patients[a.patient_id]
        if a.agent == AGENT_BED:
            bed = s.beds[a.resource_id]
            bed.occupant_id, bed.occupancy_start = p.patient_id, t
            bed.expected_release = t + p.expected_length_of_stay
            p.bed_id, p.admit_time = bed.bed_id, t
            s.set_stage(p, Stage.WAITING_DOCTOR)
        elif a.agent == AGENT_DOCTOR:
            doc = s.doctors[a.resource_id]
            doc.patient_id, doc.activity = p.patient_id, "treatment"
            doc.busy_until = t + p.treatment_duration
            p.doctor_id, p.treatment_start = doc.doctor_id, t
            s.set_stage(p, Stage.IN_TREATMENT)
        else:  # AGENT_OR: surgeon = lowest-id free doctor of the required specialization
            room = s.ors[a.resource_id]
            surgeon = s.free_doctors(p.specialization)[0]
            end = t + p.surgery_duration
            surgeon.patient_id, surgeon.activity, surgeon.busy_until = p.patient_id, "surgery", end
            room.patient_id, room.start, room.end = p.patient_id, t, end
            p.surgeon_id, p.or_id, p.surgery_start = surgeon.doctor_id, room.or_id, t
            s.set_stage(p, Stage.IN_SURGERY)

    # ----------------------------------------------------------- time advance
    def _advance_time(self) -> dict[str, Any]:
        s = self.state
        new_t = s.time + self.step_minutes
        s.time = new_t
        stats = {"treatments_completed": 0, "surgeries_completed": 0, "discharges": 0}

        for doc in s.doctors:
            if doc.patient_id != -1 and doc.busy_until <= new_t:
                p = s.patients[doc.patient_id]
                if doc.activity == "treatment":
                    p.treatment_released = new_t
                    stats["treatments_completed"] += 1
                    if p.requires_surgery:
                        s.set_stage(p, Stage.WAITING_SURGERY)
                    else:
                        p.care_done_time = new_t
                        s.set_stage(p, Stage.RECOVERY)
                else:
                    p.surgery_released = new_t
                    p.care_done_time = new_t
                    stats["surgeries_completed"] += 1
                    s.set_stage(p, Stage.RECOVERY)
                doc.patient_id, doc.activity, doc.busy_until = -1, "idle", -1
        for room in s.ors:
            if room.patient_id != -1 and room.end <= new_t:
                room.patient_id, room.start, room.end = -1, -1, -1

        for p in s.patients_in_stage(Stage.RECOVERY):
            due = max(p.admit_time + p.length_of_stay, p.care_done_time)
            if due <= new_t:
                bed = s.beds[p.bed_id]
                bed.occupant_id, bed.occupancy_start, bed.expected_release = -1, -1, -1
                p.discharge_time = new_t
                s.set_stage(p, Stage.DISCHARGED)
                stats["discharges"] += 1

        stats["new_arrivals"] = self._admit_arrivals()
        apply_outages(s, self._events)
        stats["waiting"] = {
            "bed": len(s.stage_sets[Stage.WAITING_BED]),
            "doctor": len(s.stage_sets[Stage.WAITING_DOCTOR]),
            "surgery": len(s.stage_sets[Stage.WAITING_SURGERY]),
        }
        stats["waiting_emergency"] = {
            "bed": sum(1 for p in s.patients_in_stage(Stage.WAITING_BED) if p.is_emergency),
            "doctor": sum(1 for p in s.patients_in_stage(Stage.WAITING_DOCTOR) if p.is_emergency),
            "surgery": sum(1 for p in s.patients_in_stage(Stage.WAITING_SURGERY) if p.is_emergency),
        }
        return stats

    def _admit_arrivals(self) -> int:
        """Move patients with arrival_time <= now from the pending list into the hospital."""
        n = 0
        while (self._next_arrival < len(self._pending)
               and self._pending[self._next_arrival].arrival_time <= self.state.time):
            self.state.add_patient(self._pending[self._next_arrival])
            self._next_arrival += 1
            n += 1
        return n

    # --------------------------------------------------------------- results
    def get_episode_summary(self) -> dict[str, Any]:
        """Raw performance metrics so far (definitions in src/evaluation/metrics.py)."""
        horizon = min(self.state.time, self.horizon) if self.state.time > 0 else self.horizon
        return compute_episode_metrics(self.state, horizon, self.action_counts, self.invalid_reasons)