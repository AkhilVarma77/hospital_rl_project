"""Hard clinical and resource constraints.

`validate_action` returns None if an action is allowed, otherwise a REASON string.
These rules are enforced by the environment, never traded off inside the reward.

Terminology
-----------
* INVALID  : the action violates a hard constraint -> rejected, state unchanged, counted.
* DEFERRED : the agent explicitly chose to do nothing -> neutral, patients keep waiting.
* EXECUTED : the action was valid and has been carried out (it may still earn a low reward).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.environment.hospital_state import (
    AGENT_BED, AGENT_DOCTOR, AGENT_OR, AGENTS, ICU, Action, HospitalState,
    PatientRecord, Stage,
)

# ---- reason codes ---------------------------------------------------------------
UNKNOWN_AGENT = "unknown_agent"
WRONG_AGENT = "wrong_agent"
TOO_MANY_ACTIONS = "too_many_actions"
UNKNOWN_PATIENT = "unknown_patient"
UNKNOWN_RESOURCE = "unknown_resource"
WRONG_STAGE = "wrong_stage"
RESOURCE_BUSY = "resource_busy"
RESOURCE_UNAVAILABLE = "resource_unavailable"        # temporarily out of service (outage)
BED_TYPE_MISMATCH = "bed_type_mismatch"
SPECIALIZATION_MISMATCH = "specialization_mismatch"
SURGERY_TYPE_UNSUPPORTED = "surgery_type_unsupported"
NO_SURGEON_FREE = "no_surgeon_free"
EMERGENCY_PRIORITY = "emergency_priority_violation"
ICU_RESERVE = "icu_reserve_violation"


@dataclass(frozen=True)
class ClinicalRules:
    emergency_priority: bool = True
    icu_reserve_for_emergency: int = 1

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ClinicalRules":
        r = cfg.get("clinical_rules", {})
        return cls(bool(r.get("emergency_priority", True)),
                   int(r.get("icu_reserve_for_emergency", 1)))


# ---- helpers ------------------------------------------------------------------------
def _waiting_emergencies_for_beds(state: HospitalState, bed_type: str) -> int:
    return sum(1 for p in state.patients_in_stage(Stage.WAITING_BED)
               if p.is_emergency and p.bed_type == bed_type)


def _emergency_doctor_demand(state: HospitalState, specialization: str) -> int:
    """Emergency patients who still need a doctor of this specialization
    (for treatment, or as surgeon)."""
    count = 0
    for stage in (Stage.WAITING_DOCTOR, Stage.WAITING_SURGERY):
        count += sum(1 for p in state.patients_in_stage(stage)
                     if p.is_emergency and p.specialization == specialization)
    return count


def _waiting_emergency_surgeries(state: HospitalState) -> int:
    return sum(1 for p in state.patients_in_stage(Stage.WAITING_SURGERY) if p.is_emergency)


# ---- per-agent validation ---------------------------------------------------------
def _validate_bed(state: HospitalState, rules: ClinicalRules, p: PatientRecord,
                  bed_id: int) -> Optional[str]:
    if p.stage != Stage.WAITING_BED:
        return WRONG_STAGE
    if not 0 <= bed_id < len(state.beds):
        return UNKNOWN_RESOURCE
    if bed_id in state.blocked_beds:
        return RESOURCE_UNAVAILABLE
    bed = state.beds[bed_id]
    if not bed.is_free:
        return RESOURCE_BUSY
    if bed.bed_type != p.bed_type:
        return BED_TYPE_MISMATCH
    if not p.is_emergency:
        if (bed.bed_type == ICU and rules.icu_reserve_for_emergency > 0
                and len(state.free_beds(ICU)) <= rules.icu_reserve_for_emergency):
            return ICU_RESERVE
        if (rules.emergency_priority and
                len(state.free_beds(p.bed_type)) <= _waiting_emergencies_for_beds(state, p.bed_type)):
            return EMERGENCY_PRIORITY
    return None


def _validate_doctor(state: HospitalState, rules: ClinicalRules, p: PatientRecord,
                     doctor_id: int) -> Optional[str]:
    if p.stage != Stage.WAITING_DOCTOR:
        return WRONG_STAGE
    if not 0 <= doctor_id < len(state.doctors):
        return UNKNOWN_RESOURCE
    if doctor_id in state.blocked_doctors:
        return RESOURCE_UNAVAILABLE
    doc = state.doctors[doctor_id]
    if not doc.is_free:
        return RESOURCE_BUSY
    if doc.specialization != p.specialization:
        return SPECIALIZATION_MISMATCH
    if (not p.is_emergency and rules.emergency_priority and
            len(state.free_doctors(p.specialization)) <= _emergency_doctor_demand(state, p.specialization)):
        return EMERGENCY_PRIORITY
    return None


def _validate_or(state: HospitalState, rules: ClinicalRules, p: PatientRecord,
                 or_id: int) -> Optional[str]:
    if p.stage != Stage.WAITING_SURGERY:
        return WRONG_STAGE
    if not 0 <= or_id < len(state.ors):
        return UNKNOWN_RESOURCE
    if or_id in state.blocked_ors:
        return RESOURCE_UNAVAILABLE
    room = state.ors[or_id]
    if not room.is_free:
        return RESOURCE_BUSY
    if p.surgery_type not in room.supported_types:
        return SURGERY_TYPE_UNSUPPORTED
    if not state.free_doctors(p.specialization):
        return NO_SURGEON_FREE
    if not p.is_emergency and rules.emergency_priority:
        if len(state.free_ors()) <= _waiting_emergency_surgeries(state):
            return EMERGENCY_PRIORITY
        if len(state.free_doctors(p.specialization)) <= _emergency_doctor_demand(state, p.specialization):
            return EMERGENCY_PRIORITY
    return None


def validate_action(state: HospitalState, rules: ClinicalRules, action: Action) -> Optional[str]:
    """Return None if `action` is allowed in `state`, else a reason code."""
    if action.agent not in AGENTS:
        return UNKNOWN_AGENT
    if action.is_defer:
        return None
    p = state.patients.get(action.patient_id)
    if p is None:
        return UNKNOWN_PATIENT
    if action.resource_id is None:
        return UNKNOWN_RESOURCE
    if action.agent == AGENT_BED:
        return _validate_bed(state, rules, p, action.resource_id)
    if action.agent == AGENT_DOCTOR:
        return _validate_doctor(state, rules, p, action.resource_id)
    return _validate_or(state, rules, p, action.resource_id)


def feasible_actions(state: HospitalState, rules: ClinicalRules, agent: str) -> list[Action]:
    """All non-defer actions that are individually valid right now (defer is always allowed).

    Individually valid does not guarantee jointly valid: two feasible actions can compete for
    the same resource, in which case the environment executes the more urgent one first.
    This list is also the basis for action masks in later steps.
    """
    out: list[Action] = []
    if agent == AGENT_BED:
        for p in state.patients_in_stage(Stage.WAITING_BED):
            for b in state.free_beds(p.bed_type):
                a = Action(agent, p.patient_id, b.bed_id)
                if validate_action(state, rules, a) is None:
                    out.append(a)
    elif agent == AGENT_DOCTOR:
        for p in state.patients_in_stage(Stage.WAITING_DOCTOR):
            for d in state.free_doctors(p.specialization):
                a = Action(agent, p.patient_id, d.doctor_id)
                if validate_action(state, rules, a) is None:
                    out.append(a)
    elif agent == AGENT_OR:
        for p in state.patients_in_stage(Stage.WAITING_SURGERY):
            for o in state.free_ors():
                a = Action(agent, p.patient_id, o.or_id)
                if validate_action(state, rules, a) is None:
                    out.append(a)
    else:
        raise ValueError(f"unknown agent '{agent}'")
    return out
