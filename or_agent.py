"""Operating-room scheduling agent interface: chooses which patient is operated on next."""
from __future__ import annotations

from typing import Optional

from src.agents.base_agent import BaseAgent, _clip, _frac
from src.environment.hospital_state import AGENT_OR, HospitalState, PatientRecord, Stage


class ORAgent(BaseAgent):
    name = AGENT_OR
    stage = Stage.WAITING_SURGERY

    @property
    def pool_dim(self) -> int:
        return 2

    @property
    def slot_dim(self) -> int:
        return 1 + 6 + len(self.surgery_types)

    def pool_features(self, state: HospitalState) -> list[float]:
        """[free OR fraction, est. time until the next OR is free]"""
        free = [o for o in state.ors if o.is_free and o.or_id not in state.blocked_ors]
        if free:
            wait = 0.0
        else:
            rem = []
            for o in state.ors:
                if not o.is_free:
                    p = state.patients[o.patient_id]
                    rem.append(max(p.est_surgery_duration - (state.time - p.surgery_start), 0))
            wait = _clip(min(rem) / self.release_norm, 1.0) if rem else 1.0
        return [_frac(len(free), len(state.ors)), wait]

    def slot_features(self, state: HospitalState, p: PatientRecord) -> list[float]:
        surgeons = [d for d in state.doctors if d.specialization == p.specialization]
        free_surgeons = [d for d in surgeons if d.is_free and d.doctor_id not in state.blocked_doctors]
        return [
            float(p.is_emergency),
            p.urgency_level / 4.0,
            p.severity / 5.0,
            _clip((state.time - p.treatment_released) / self.wait_norm),   # waiting for surgery
            _clip(p.est_surgery_duration / self.duration_norm),
            _frac(len(free_surgeons), len(surgeons)),                        # needed to be feasible
        ] + [float(p.surgery_type == s) for s in self.surgery_types]

    def default_resource(self, state: HospitalState, p: PatientRecord) -> Optional[int]:
        free = state.free_ors()
        return free[0].or_id if free else None
