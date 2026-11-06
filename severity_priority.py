"""Severity / emergency-priority baseline.

Order of service (all agents): emergencies before electives, then higher urgency level, then
higher severity, then earliest arrival. This is the classic "triage" rule.
"""
from __future__ import annotations

from src.baselines.base import DispatchPolicy
from src.environment.hospital_state import PatientRecord


class SeverityPriorityPolicy(DispatchPolicy):
    name = "severity"

    def priority_key(self, patient: PatientRecord) -> tuple:
        return (0 if patient.is_emergency else 1, -patient.urgency_level, -patient.severity,
                patient.arrival_time, patient.patient_id)
