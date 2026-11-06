"""Episode-level performance metrics computed from the final hospital state.

Definitions (all times in simulated minutes; completions are processed at step boundaries)
--------------------------------------------------------------------------------------
bed_wait        = admit_time - arrival_time                (patients admitted)
doctor_wait     = treatment_start - admit_time             (patients whose treatment started)
wait_to_treatment = treatment_start - arrival_time         (= bed_wait + doctor_wait)
emergency_response_delay = mean wait_to_treatment over EMERGENCY patients
or_wait         = surgery_start - treatment_released       (patients whose surgery started)
Averages cover only patients for whom the event happened. Patients still unserved at the
end are reported separately (count and censored wait) so they are not silently ignored.

utilization     = resource-minutes HELD / (number of units * episode length)
                  beds: admission -> discharge; doctors: treatment + surgery time held;
                  ORs: surgery time held. "Held" runs until the step boundary at which the
                  resource became free again. This is RAW utilization; it says nothing about
                  whether the use was appropriate.
throughput      = number of patients discharged during the episode.
"""
from __future__ import annotations

from typing import Any, Optional

from src.environment.hospital_state import GENERAL, ICU, HospitalState, Stage


def _mean(values: list[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _held(start: int, end: int, horizon: int) -> int:
    """Minutes between start and end, clipped to the episode horizon."""
    if start < 0:
        return 0
    stop = horizon if end < 0 else min(end, horizon)
    return max(0, stop - start)


def compute_episode_metrics(state: HospitalState, horizon: int,
                            action_counts: dict[str, dict[str, int]],
                            invalid_reasons: dict[str, dict[str, int]]) -> dict[str, Any]:
    pats = list(state.patients.values())
    admitted = [p for p in pats if p.admit_time >= 0]
    treated = [p for p in pats if p.treatment_start >= 0]
    operated = [p for p in pats if p.surgery_start >= 0]
    emergencies = [p for p in treated if p.is_emergency]
    electives = [p for p in treated if not p.is_emergency]
    unserved = [p for p in pats if p.treatment_start < 0]

    bed_minutes = {ICU: 0, GENERAL: 0}
    for p in admitted:
        bed_minutes[p.bed_type] += _held(p.admit_time, p.discharge_time, horizon)
    doctor_minutes = sum(_held(p.treatment_start, p.treatment_released, horizon) +
                         _held(p.surgery_start, p.surgery_released, horizon) for p in pats)
    or_minutes = sum(_held(p.surgery_start, p.surgery_released, horizon) for p in pats)

    n_icu = sum(1 for b in state.beds if b.bed_type == ICU)
    n_gen = len(state.beds) - n_icu

    def util(minutes: int, units: int) -> Optional[float]:
        return minutes / (units * horizon) if units > 0 else None

    return {
        "patients_arrived": len(pats),
        "patients_discharged": sum(1 for p in pats if p.stage == Stage.DISCHARGED),
        "throughput": sum(1 for p in pats if p.stage == Stage.DISCHARGED),
        "treatments_completed": sum(1 for p in pats if p.treatment_released >= 0),
        "surgeries_completed": sum(1 for p in pats if p.surgery_released >= 0),
        "mean_bed_wait": _mean([p.admit_time - p.arrival_time for p in admitted]),
        "mean_doctor_wait": _mean([p.treatment_start - p.admit_time for p in treated]),
        "mean_wait_to_treatment": _mean([p.treatment_start - p.arrival_time for p in treated]),
        "emergency_response_delay": _mean([p.treatment_start - p.arrival_time for p in emergencies]),
        "elective_wait_to_treatment": _mean([p.treatment_start - p.arrival_time for p in electives]),
        "mean_or_wait": _mean([p.surgery_start - p.treatment_released for p in operated]),
        "emergency_surgery_wait": _mean([p.surgery_start - p.treatment_released
                                         for p in operated if p.is_emergency]),
        "unserved_patients": len(unserved),
        "unserved_mean_censored_wait": _mean([horizon - p.arrival_time for p in unserved]),
        "bed_utilization": util(bed_minutes[ICU] + bed_minutes[GENERAL], len(state.beds)),
        "icu_bed_utilization": util(bed_minutes[ICU], n_icu),
        "general_bed_utilization": util(bed_minutes[GENERAL], n_gen),
        "doctor_utilization": util(doctor_minutes, len(state.doctors)),
        "or_utilization": util(or_minutes, len(state.ors)),
        "actions_executed": {a: c.get("executed", 0) for a, c in action_counts.items()},
        "actions_deferred": {a: c.get("deferred", 0) for a, c in action_counts.items()},
        "invalid_actions": {a: c.get("invalid", 0) for a, c in action_counts.items()},
        "invalid_actions_total": sum(c.get("invalid", 0) for c in action_counts.values()),
        "invalid_reasons": {a: dict(r) for a, r in invalid_reasons.items()},
    }
