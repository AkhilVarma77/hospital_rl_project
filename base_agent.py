"""Common machinery for the three learning agents' INTERFACE (observation, actions, masks).

This file contains NO learning code. It defines what an agent sees and what it can do.

Action space (identical for PPO and DQN): Discrete(K + 1)
    index 0..K-1 : serve the patient in queue slot i
    index K      : defer (do nothing)
Slots hold the agent's K highest-priority waiting patients, ordered by
(emergency first, higher urgency, higher severity, earlier arrival, patient id).
The concrete resource is bound deterministically (lowest-id free resource of the right type).
This loses nothing because resources inside a pool are interchangeable in this model
(same bed type / same specialization / every OR supports every surgery type).

Observation = [time-of-day sin, cos | forecast (2) | can-act fraction | own resource pool
               | own queue length, emergencies | K slots x slot features].
It is LOCAL: own queue, own resource pool and the availability flags that decide whether its own
actions are feasible. Only observable quantities are used (estimates, never true durations).
Cross-agent information is NOT included; the communication module adds it in a later step,
so that "independent agents" really are independent.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from src.environment.hospital_env import HospitalEnv
from src.environment.hospital_state import Action, DoctorState, HospitalState, PatientRecord, Stage

CLIP = 3.0  # normalised features are clipped to [0, CLIP]


def _clip(x: float, hi: float = CLIP) -> float:
    return float(min(max(x, 0.0), hi))


def _frac(num: int, den: int) -> float:
    return num / den if den > 0 else 0.0


def queue_order_key(p: PatientRecord) -> tuple:
    return (0 if p.is_emergency else 1, -p.urgency_level, -p.severity, p.arrival_time, p.patient_id)


class BaseAgent:
    name: str = ""
    stage: Stage = Stage.WAITING_BED

    def __init__(self, cfg: dict[str, Any]) -> None:
        o = cfg.get("observation", {})
        self.K = int(o.get("max_queue", 8))
        self.wait_norm = float(o.get("wait_norm_minutes", 240))
        self.duration_norm = float(o.get("duration_norm_minutes", 240))
        self.los_norm = float(o.get("los_norm_minutes", 1440))
        self.release_norm = float(o.get("release_norm_minutes", 240))
        self.count_norm = float(o.get("queue_count_norm", 20))
        self.specs: list[str] = list(cfg["patients"]["specialization_probs_emergency"])
        self.surgery_types: list[str] = list(
            dict.fromkeys(cfg["patients"]["surgery_type_by_specialization"].values()))

    # ---- sizes ------------------------------------------------------------------------
    @property
    def n_actions(self) -> int:
        return self.K + 1

    @property
    def defer_index(self) -> int:
        return self.K

    @property
    def pool_dim(self) -> int:
        raise NotImplementedError

    @property
    def slot_dim(self) -> int:  # 1 validity flag + features
        raise NotImplementedError

    @property
    def obs_dim(self) -> int:
        return 5 + self.pool_dim + 2 + self.K * self.slot_dim

    # ---- to be provided by subclasses -----------------------------------------------------
    def pool_features(self, state: HospitalState) -> list[float]:
        raise NotImplementedError

    def slot_features(self, state: HospitalState, p: PatientRecord) -> list[float]:
        raise NotImplementedError

    def default_resource(self, state: HospitalState, p: PatientRecord) -> Optional[int]:
        raise NotImplementedError

    # ---- queue ---------------------------------------------------------------------------------
    def queue(self, state: HospitalState) -> list[PatientRecord]:
        """The K highest-priority patients waiting for this agent."""
        return sorted(state.patients_in_stage(self.stage), key=queue_order_key)[: self.K]

    # ---- observation -----------------------------------------------------------------------
    def observe(self, env: HospitalEnv, forecast: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
        state = env.state
        angle = 2.0 * math.pi * ((state.time % 1440) / 1440.0)
        vec: list[float] = [math.sin(angle), math.cos(angle), float(forecast[0]), float(forecast[1]),
                            env.actions_remaining(self.name) / env.max_actions]
        vec += self.pool_features(state)
        waiting = state.patients_in_stage(self.stage)
        vec += [_clip(len(waiting) / self.count_norm),
                _clip(sum(1 for p in waiting if p.is_emergency) / self.count_norm)]
        slots = self.queue(state)
        n_feat = self.slot_dim - 1
        for i in range(self.K):
            if i < len(slots):
                feats = self.slot_features(state, slots[i])
                assert len(feats) == n_feat
                vec += [1.0] + feats
            else:
                vec += [0.0] * self.slot_dim
        out = np.asarray(vec, dtype=np.float32)
        assert out.shape == (self.obs_dim,), (out.shape, self.obs_dim)
        return out

    # ---- actions --------------------------------------------------------------------------------
    def action_mask(self, env: HospitalEnv, finished: bool = False) -> np.ndarray:
        """Boolean mask over K+1 actions. Defer is always allowed. A slot is allowed only if the
        action it would produce passes ALL hard constraints right now. A `finished` agent (it
        deferred voluntarily this step) or one with no action budget left may only defer."""
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[self.defer_index] = True
        if finished or env.actions_remaining(self.name) <= 0:
            return mask
        state = env.state
        for i, p in enumerate(self.queue(state)):
            rid = self.default_resource(state, p)
            if rid is not None and env.validate(Action(self.name, p.patient_id, rid)) is None:
                mask[i] = True
        return mask

    def decode(self, env: HospitalEnv, index: int) -> Action:
        """Turn an action index into an environment Action (using the CURRENT state).

        An empty slot or a slot with no free resource is returned as a deliberately invalid
        action, so choosing a masked index shows up as an invalid attempt instead of crashing."""
        if not 0 <= index <= self.K:
            raise ValueError(f"action index {index} outside 0..{self.K}")
        if index == self.defer_index:
            return Action.defer(self.name)
        slots = self.queue(env.state)
        if index >= len(slots):
            return Action(self.name, -1, None)             # no such patient -> unknown_patient
        p = slots[index]
        return Action(self.name, p.patient_id, self.default_resource(env.state, p))

    # ---- shared helpers for subclasses -----------------------------------------------------
    def doctor_remaining_estimate(self, state: HospitalState, doc: DoctorState) -> float:
        """Estimated minutes until a busy doctor is free (from ESTIMATED durations only)."""
        p = state.patients[doc.patient_id]
        if doc.activity == "treatment":
            rem = p.est_treatment_duration - (state.time - p.treatment_start)
        else:
            rem = p.est_surgery_duration - (state.time - p.surgery_start)
        return max(float(rem), 0.0)
