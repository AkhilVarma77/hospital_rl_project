"""Doctor / patient scheduling agent interface: chooses which admitted patient is treated next."""
from __future__ import annotations

from typing import Optional

from src.agents.base_agent import BaseAgent, _clip, _frac
from src.environment.hospital_state import AGENT_DOCTOR, HospitalState, PatientRecord, Stage


class DoctorAgent(BaseAgent):
    name = AGENT_DOCTOR
    stage = Stage.WAITING_DOCTOR

    @property
    def pool_dim(self) -> int:
        return 2 * len(self.specs)

    @property
    def slot_dim(self) -> int:
        return 1 + 7 + len(self.specs)

    def pool_features(self, state: HospitalState) -> list[float]:
        """Per specialization: free-doctor fraction, then est. time until one is free."""
        fracs, waits_out = [], []
        for spec in self.specs:
            docs = [d for d in state.doctors if d.specialization == spec]
            free = [d for d in docs if d.is_free and d.doctor_id not in state.blocked_doctors]
            fracs.append(_frac(len(free), len(docs)))
            if free:
                waits_out.append(0.0)
            else:
                rem = [self.doctor_remaining_estimate(state, d) for d in docs if not d.is_free]
                waits_out.append(_clip(min(rem) / self.release_norm, 1.0) if rem else 1.0)
        return fracs + waits_out

    def slot_features(self, state: HospitalState, p: PatientRecord) -> list[float]:
        docs = [d for d in state.doctors if d.specialization == p.specialization]
        free = [d for d in docs if d.is_free and d.doctor_id not in state.blocked_doctors]
        return [
            float(p.is_emergency),
            p.urgency_level / 4.0,
            p.severity / 5.0,
            _clip((state.time - p.arrival_time) / self.wait_norm),
            _clip(p.est_treatment_duration / self.duration_norm),
            float(p.requires_surgery),
            _frac(len(free), len(docs)),
        ] + [float(p.specialization == s) for s in self.specs]

    def default_resource(self, state: HospitalState, p: PatientRecord) -> Optional[int]:
        free = state.free_doctors(p.specialization)
        return free[0].doctor_id if free else None
