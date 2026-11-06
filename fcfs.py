"""First-Come-First-Served baseline.

Within each queue (bed, doctor, OR) the patient who ARRIVED at the hospital earliest is served
first, among the actions that satisfy the hard clinical constraints. It ignores severity.
"""
from __future__ import annotations

from src.baselines.base import DispatchPolicy
from src.environment.hospital_state import PatientRecord


class FCFSPolicy(DispatchPolicy):
    name = "fcfs"

    def priority_key(self, patient: PatientRecord) -> tuple:
        return (patient.arrival_time, patient.patient_id)
