"""Bed allocation agent interface: chooses which waiting patient gets a bed."""
from __future__ import annotations

from typing import Optional

from src.agents.base_agent import BaseAgent, _clip, _frac
from src.environment.hospital_state import AGENT_BED, GENERAL, ICU, HospitalState, PatientRecord, Stage


class BedAgent(BaseAgent):
    name = AGENT_BED
    stage = Stage.WAITING_BED

    @property
    def pool_dim(self) -> int:
        return 4

    @property
    def slot_dim(self) -> int:
        return 8

    def pool_features(self, state: HospitalState) -> list[float]:
        """[free ICU fraction, free general fraction, est. time to next ICU / general release]"""
        free_frac, release = [], []
        for bed_type in (ICU, GENERAL):
            beds = [b for b in state.beds if b.bed_type == bed_type]
            free = [b for b in beds if b.is_free and b.bed_id not in state.blocked_beds]
            free_frac.append(_frac(len(free), len(beds)))
            if free:
                release.append(0.0)
            else:
                waits = [max(b.expected_release - state.time, 0) for b in beds if not b.is_free]
                release.append(_clip(min(waits) / self.release_norm, 1.0) if waits else 1.0)
        return free_frac + release

    def slot_features(self, state: HospitalState, p: PatientRecord) -> list[float]:
        return [
            float(p.is_emergency),
            p.urgency_level / 4.0,
            p.severity / 5.0,
            _clip((state.time - p.arrival_time) / self.wait_norm),
            float(p.bed_type == ICU),
            _clip(p.expected_length_of_stay / self.los_norm),
            float(p.requires_surgery),
        ]

    def default_resource(self, state: HospitalState, p: PatientRecord) -> Optional[int]:
        free = state.free_beds(p.bed_type)
        return free[0].bed_id if free else None
