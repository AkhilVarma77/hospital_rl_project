"""Applying configured hospital events to the live state.

Arrival-rate events (emergency_surge, elective_increase) are already baked into the
generated patient table. Only `resource_outage` events act on the running hospital:
for the duration of the event, the highest-numbered `magnitude` units of the target are
out of service (cannot be newly assigned; work already in progress finishes normally).
"""
from __future__ import annotations

import pandas as pd

from src.environment.hospital_state import GENERAL, ICU, HospitalState

OUTAGE_TARGETS = ("general_beds", "icu_beds", "doctors", "operating_rooms")


def active_outages(events: pd.DataFrame, t: int) -> dict[str, int]:
    """Units out of service at time t, per target."""
    out = {k: 0 for k in OUTAGE_TARGETS}
    if events is None or events.empty:
        return out
    for ev in events.itertuples():
        if ev.event_type == "resource_outage" and ev.start_minute <= t < ev.end_minute:
            if ev.target not in out:
                raise ValueError(f"unknown outage target '{ev.target}'")
            out[ev.target] += int(ev.magnitude)
    return out


def apply_outages(state: HospitalState, events: pd.DataFrame) -> None:
    """Recompute which resources are blocked at state.time."""
    outages = active_outages(events, state.time)

    def top_ids(ids: list[int], n: int) -> set[int]:
        return set(sorted(ids)[len(ids) - min(n, len(ids)):]) if n > 0 else set()

    state.blocked_beds = (
        top_ids([b.bed_id for b in state.beds if b.bed_type == GENERAL], outages["general_beds"])
        | top_ids([b.bed_id for b in state.beds if b.bed_type == ICU], outages["icu_beds"]))
    state.blocked_doctors = top_ids([d.doctor_id for d in state.doctors], outages["doctors"])
    state.blocked_ors = top_ids([o.or_id for o in state.ors], outages["operating_rooms"])
