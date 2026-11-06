"""Plain data structures describing the hospital at one moment in time.

The state is mutated ONLY by `HospitalEnv`. Agents and baselines must treat it as read-only.

Patient pathway (stages)
------------------------
    WAITING_BED -> WAITING_DOCTOR -> IN_TREATMENT -> (WAITING_SURGERY -> IN_SURGERY ->)
    RECOVERY -> DISCHARGED

* A bed is needed from admission until discharge.
* A doctor of the required specialization treats the patient (only after admission).
* If surgery is required, it happens after treatment and needs a free OR AND a free doctor
  of the required specialization (the "surgeon") for the whole surgery.
* After all care is done the patient stays in RECOVERY until admission_time + length_of_stay,
  then is discharged and the bed is released.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

AGENT_BED = "bed"
AGENT_DOCTOR = "doctor"
AGENT_OR = "or"
AGENTS = (AGENT_BED, AGENT_DOCTOR, AGENT_OR)
AGENT_RANK = {name: i for i, name in enumerate(AGENTS)}  # tie-break order in joint execution

ICU = "icu"
GENERAL = "general"


class Stage(str, Enum):
    WAITING_BED = "waiting_bed"
    WAITING_DOCTOR = "waiting_doctor"
    IN_TREATMENT = "in_treatment"
    WAITING_SURGERY = "waiting_surgery"
    IN_SURGERY = "in_surgery"
    RECOVERY = "recovery"
    DISCHARGED = "discharged"


@dataclass(frozen=True)
class Action:
    """One decision by one agent.

    bed agent:    (patient_id, bed_id)
    doctor agent: (patient_id, doctor_id)
    OR agent:     (patient_id, or_id)   -- the environment picks the surgeon
    Defer (do nothing): patient_id is None.
    """
    agent: str
    patient_id: Optional[int] = None
    resource_id: Optional[int] = None

    @property
    def is_defer(self) -> bool:
        return self.patient_id is None

    @classmethod
    def defer(cls, agent: str) -> "Action":
        return cls(agent)


@dataclass
class PatientRecord:
    """A patient who has arrived. TRUE durations are for the simulator only;
    the `est_*` / `expected_*` fields are what agents may observe."""
    patient_id: int
    arrival_time: int
    severity: int
    urgency_level: int
    is_emergency: bool
    bed_type: str
    specialization: str
    treatment_duration: int          # true
    est_treatment_duration: int      # observable estimate
    requires_surgery: bool
    surgery_type: str
    surgery_duration: int            # true
    est_surgery_duration: int        # observable estimate
    length_of_stay: int              # true
    expected_length_of_stay: int     # observable estimate

    stage: Stage = Stage.WAITING_BED
    bed_id: int = -1
    doctor_id: int = -1              # treating doctor
    surgeon_id: int = -1
    or_id: int = -1
    admit_time: int = -1
    treatment_start: int = -1
    treatment_released: int = -1     # step boundary at which the doctor became free again
    surgery_start: int = -1
    surgery_released: int = -1
    care_done_time: int = -1
    discharge_time: int = -1

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "PatientRecord":
        return cls(
            patient_id=int(row["patient_id"]),
            arrival_time=int(row["arrival_time"]),
            severity=int(row["severity"]),
            urgency_level=int(row["urgency_level"]),
            is_emergency=bool(row["is_emergency"]),
            bed_type=str(row["required_bed_type"]),
            specialization=str(row["required_specialization"]),
            treatment_duration=int(row["treatment_duration"]),
            est_treatment_duration=int(row["est_treatment_duration"]),
            requires_surgery=bool(row["requires_surgery"]),
            surgery_type=str(row["surgery_type"]),
            surgery_duration=int(row["surgery_duration"]),
            est_surgery_duration=int(row["est_surgery_duration"]),
            length_of_stay=int(row["length_of_stay"]),
            expected_length_of_stay=int(row["expected_length_of_stay"]),
        )


@dataclass
class BedState:
    bed_id: int
    bed_type: str
    occupant_id: int = -1
    occupancy_start: int = -1
    expected_release: int = -1       # based on the ESTIMATED length of stay

    @property
    def is_free(self) -> bool:
        return self.occupant_id == -1


@dataclass
class DoctorState:
    doctor_id: int
    specialization: str
    patient_id: int = -1
    activity: str = "idle"           # idle | treatment | surgery
    busy_until: int = -1             # true end time of current activity

    @property
    def is_free(self) -> bool:
        return self.patient_id == -1


@dataclass
class ORState:
    or_id: int
    supported_types: frozenset[str]
    patient_id: int = -1
    start: int = -1
    end: int = -1                    # true end time of current surgery

    @property
    def is_free(self) -> bool:
        return self.patient_id == -1


@dataclass
class HospitalState:
    """Everything that defines the hospital right now."""
    time: int = 0
    patients: dict[int, PatientRecord] = field(default_factory=dict)  # all ARRIVED patients
    beds: list[BedState] = field(default_factory=list)
    doctors: list[DoctorState] = field(default_factory=list)
    ors: list[ORState] = field(default_factory=list)
    blocked_beds: set[int] = field(default_factory=set)      # temporarily out of service
    blocked_doctors: set[int] = field(default_factory=set)
    blocked_ors: set[int] = field(default_factory=set)
    stage_sets: dict[Stage, set[int]] = field(
        default_factory=lambda: {s: set() for s in Stage})

    # ---- patient bookkeeping -------------------------------------------------
    def add_patient(self, p: PatientRecord) -> None:
        self.patients[p.patient_id] = p
        self.stage_sets[p.stage].add(p.patient_id)

    def set_stage(self, p: PatientRecord, stage: Stage) -> None:
        self.stage_sets[p.stage].discard(p.patient_id)
        p.stage = stage
        self.stage_sets[stage].add(p.patient_id)

    def patients_in_stage(self, stage: Stage) -> list[PatientRecord]:
        """Patients in a stage, ordered by (arrival_time, patient_id)."""
        recs = [self.patients[i] for i in self.stage_sets[stage]]
        return sorted(recs, key=lambda p: (p.arrival_time, p.patient_id))

    # ---- free-resource queries (blocked resources are never "free") -----------
    def free_beds(self, bed_type: Optional[str] = None) -> list[BedState]:
        return [b for b in self.beds if b.is_free and b.bed_id not in self.blocked_beds
                and (bed_type is None or b.bed_type == bed_type)]

    def free_doctors(self, specialization: Optional[str] = None) -> list[DoctorState]:
        return [d for d in self.doctors if d.is_free and d.doctor_id not in self.blocked_doctors
                and (specialization is None or d.specialization == specialization)]

    def free_ors(self) -> list[ORState]:
        return [o for o in self.ors if o.is_free and o.or_id not in self.blocked_ors]

    # ---- debugging ---------------------------------------------------------------
    def validate_consistency(self) -> list[str]:
        """Return a list of human-readable problems (empty list = state is consistent)."""
        problems: list[str] = []
        for b in self.beds:
            if b.occupant_id != -1:
                p = self.patients.get(b.occupant_id)
                if p is None:
                    problems.append(f"bed {b.bed_id} occupied by unknown patient {b.occupant_id}")
                elif p.bed_id != b.bed_id or p.stage in (Stage.WAITING_BED, Stage.DISCHARGED):
                    problems.append(f"bed {b.bed_id} / patient {p.patient_id} mismatch")
                elif p.bed_type != b.bed_type:
                    problems.append(f"bed {b.bed_id} type does not match patient {p.patient_id}")
        for p in self.patients.values():
            if p.stage not in (Stage.WAITING_BED, Stage.DISCHARGED):
                if p.bed_id < 0 or self.beds[p.bed_id].occupant_id != p.patient_id:
                    problems.append(f"patient {p.patient_id} in {p.stage.value} but bed link broken")
            in_sets = [s for s, ids in self.stage_sets.items() if p.patient_id in ids]
            if in_sets != [p.stage]:
                problems.append(f"patient {p.patient_id} stage sets inconsistent: {in_sets}")
        for d in self.doctors:
            if d.patient_id != -1:
                p = self.patients.get(d.patient_id)
                ok = p is not None and (
                    (d.activity == "treatment" and p.doctor_id == d.doctor_id
                     and p.stage == Stage.IN_TREATMENT) or
                    (d.activity == "surgery" and p.surgeon_id == d.doctor_id
                     and p.stage == Stage.IN_SURGERY))
                if not ok:
                    problems.append(f"doctor {d.doctor_id} assignment inconsistent")
                elif p.specialization != d.specialization:
                    problems.append(f"doctor {d.doctor_id} specialization mismatch")
        for o in self.ors:
            if o.patient_id != -1:
                p = self.patients.get(o.patient_id)
                if p is None or p.or_id != o.or_id or p.stage != Stage.IN_SURGERY:
                    problems.append(f"OR {o.or_id} assignment inconsistent")
                elif p.surgery_type not in o.supported_types:
                    problems.append(f"OR {o.or_id} does not support surgery of patient {p.patient_id}")
        return problems
