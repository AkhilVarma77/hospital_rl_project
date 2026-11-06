"""Synthetic hospital data generator.

Produces reproducible, SIMULATED data (never real hospital records):
    patients, beds, doctors, operating_rooms, events, arrival_timeseries

Design notes
------------
* Time is measured in minutes since the start of the episode. One simulation step is
  `simulation.step_minutes` minutes.
* Each patient has TRUE durations (used by the simulator to decide when treatment,
  surgery and stay end) and ESTIMATED durations (the only ones agents may observe).
  This prevents agents from "seeing the future".
* Sentinel value -1 means "not set" (e.g. bed has no occupant).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.utils.config import get_scenario_config
from src.utils.seeding import make_rng

# Independent random streams (see seeding.make_rng)
_STREAM_PATIENTS = 1

_DEFAULT_EVENT_TARGET = {
    "emergency_surge": "emergency_arrivals",
    "elective_increase": "elective_arrivals",
}


def _lognormal(rng: np.random.Generator, median: np.ndarray | float, sigma: float,
               size: int) -> np.ndarray:
    """Lognormal samples with the given median (median = exp(mu))."""
    return np.asarray(median) * np.exp(sigma * rng.standard_normal(size))


class SyntheticDataGenerator:
    """Generates all synthetic tables for one scenario and one seed."""

    def __init__(self, config: dict[str, Any], scenario: str = "normal",
                 seed: int | None = None) -> None:
        self.scenario = scenario
        self.cfg = get_scenario_config(config, scenario)
        self.seed = int(self.cfg["seed"] if seed is None else seed)
        self.step_minutes = int(self.cfg["simulation"]["step_minutes"])
        self.horizon_minutes = int(round(self.cfg["simulation"]["episode_days"] * 1440))
        self.num_steps = self.horizon_minutes // self.step_minutes

    # ------------------------------------------------------------------ events
    def generate_events(self) -> pd.DataFrame:
        """Table of configured hospital events (surges, outages)."""
        rows = []
        for i, ev in enumerate(self.cfg.get("events", [])):
            start = int(round(ev["start_hour"] * 60))
            end = min(start + int(round(ev["duration_hours"] * 60)), self.horizon_minutes)
            rows.append({
                "event_id": i,
                "event_type": ev["type"],
                "start_minute": start,
                "end_minute": end,
                "magnitude": float(ev["magnitude"]),
                "target": ev.get("target", _DEFAULT_EVENT_TARGET.get(ev["type"], "")),
            })
        cols = ["event_id", "event_type", "start_minute", "end_minute", "magnitude", "target"]
        return pd.DataFrame(rows, columns=cols)

    # --------------------------------------------------------------- arrivals
    def _expected_arrivals_per_step(self, events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Poisson means per step for (emergency, elective) arrivals."""
        arr = self.cfg["arrivals"]
        em_prof = np.asarray(arr["emergency_hourly_profile"], dtype=float)
        el_prof = np.asarray(arr["elective_hourly_profile"], dtype=float)
        em_prof, el_prof = em_prof / em_prof.mean(), el_prof / el_prof.mean()  # mean 1.0

        total = arr["base_rate_per_hour"] * arr["demand_multiplier"]
        frac = arr["emergency_fraction"]
        step_hours = self.step_minutes / 60.0

        lam_em = np.zeros(self.num_steps)
        lam_el = np.zeros(self.num_steps)
        for s in range(self.num_steps):
            minute = s * self.step_minutes
            hour = (minute // 60) % 24
            m_em = m_el = 1.0
            for ev in events.itertuples():
                if ev.start_minute <= minute < ev.end_minute:
                    if ev.event_type == "emergency_surge" and ev.target == "emergency_arrivals":
                        m_em *= ev.magnitude
                    elif ev.event_type == "elective_increase" and ev.target == "elective_arrivals":
                        m_el *= ev.magnitude
            lam_em[s] = total * frac * em_prof[hour] * step_hours * m_em
            lam_el[s] = total * (1 - frac) * el_prof[hour] * step_hours * m_el
        return lam_em, lam_el

    # --------------------------------------------------------------- patients
    def generate_patients(self, events: pd.DataFrame | None = None) -> pd.DataFrame:
        """One row per patient arriving during the episode, sorted by arrival time."""
        if events is None:
            events = self.generate_events()
        p = self.cfg["patients"]
        rng = make_rng(self.seed, _STREAM_PATIENTS)

        # --- arrival times -------------------------------------------------
        lam_em, lam_el = self._expected_arrivals_per_step(events)
        n_em, n_el = rng.poisson(lam_em), rng.poisson(lam_el)
        steps = np.arange(self.num_steps)
        arrival_step = np.concatenate([np.repeat(steps, n_em), np.repeat(steps, n_el)])
        is_em = np.concatenate([np.ones(n_em.sum(), dtype=bool), np.zeros(n_el.sum(), dtype=bool)])
        n = len(arrival_step)
        arrival_time = arrival_step * self.step_minutes + rng.integers(0, self.step_minutes, size=n)
        order = np.argsort(arrival_time, kind="stable")
        arrival_step, arrival_time, is_em = arrival_step[order], arrival_time[order], is_em[order]

        # --- severity, urgency, bed type -----------------------------------
        sev_em = rng.choice(5, size=n, p=p["severity_probs_emergency"]) + 1
        sev_el = rng.choice(5, size=n, p=p["severity_probs_elective"]) + 1
        severity = np.where(is_em, sev_em, sev_el)
        # Urgency 1 (non-urgent) .. 4 (immediate): severity-1, +1 if emergency, clipped.
        urgency = np.clip(severity - 1 + is_em.astype(int), 1, 4)
        icu_p = np.asarray(p["icu_prob_by_severity"])[severity - 1]
        needs_icu = rng.random(n) < icu_p

        # --- specialization -------------------------------------------------
        spec_names = list(p["specialization_probs_emergency"])
        spec_em = rng.choice(spec_names, size=n,
                             p=[p["specialization_probs_emergency"][s] for s in spec_names])
        spec_el = rng.choice(spec_names, size=n,
                             p=[p["specialization_probs_elective"][s] for s in spec_names])
        specialization = np.where(is_em, spec_em, spec_el)

        # --- surgery ---------------------------------------------------------
        surgery_map = p["surgery_type_by_specialization"]
        surgical_spec = np.isin(specialization, list(surgery_map))
        surg_prob = np.where(is_em, p["surgery_prob_emergency"], p["surgery_prob_elective"])
        needs_surgery = surgical_spec & (rng.random(n) < surg_prob)
        surgery_type = np.where(
            needs_surgery, [surgery_map.get(s, "none") for s in specialization], "none")

        # --- durations (true vs estimated) ----------------------------------
        sigma_est = p["estimate_noise_sigma"]
        med_treat = np.asarray(p["treatment_median_minutes_by_severity"], dtype=float)[severity - 1]
        treat = np.clip(_lognormal(rng, med_treat, p["treatment_sigma"], n),
                        p["treatment_min_minutes"], p["treatment_max_minutes"])
        treat = np.rint(treat).astype(int)
        est_treat = np.maximum(5, np.rint(treat * np.exp(sigma_est * rng.standard_normal(n)))).astype(int)

        surg = np.clip(_lognormal(rng, p["surgery_median_minutes"], p["surgery_sigma"], n),
                       p["surgery_min_minutes"], p["surgery_max_minutes"])
        surg = np.where(needs_surgery, np.rint(surg), 0).astype(int)
        est_surg_raw = np.rint(surg * np.exp(sigma_est * rng.standard_normal(n)))
        est_surg = np.where(needs_surgery, np.maximum(5, est_surg_raw), 0).astype(int)

        stay_med = np.where(needs_icu, p["stay_median_minutes"]["icu"],
                            p["stay_median_minutes"]["general"]).astype(float)
        stay = np.clip(_lognormal(rng, stay_med, p["stay_sigma"], n),
                       p["stay_min_minutes"], p["stay_max_minutes"])
        los = np.maximum(np.rint(stay).astype(int), treat + surg)  # never shorter than care
        est_los = np.rint(los * np.exp(sigma_est * rng.standard_normal(n))).astype(int)
        est_los = np.maximum(est_los, est_treat + est_surg)

        return pd.DataFrame({
            "patient_id": np.arange(n),
            "arrival_time": arrival_time,
            "arrival_step": arrival_step,
            "severity": severity,
            "urgency_level": urgency,
            "is_emergency": is_em,
            "required_bed_type": np.where(needs_icu, "icu", "general"),
            "required_specialization": specialization,
            "treatment_duration": treat,
            "est_treatment_duration": est_treat,
            "requires_surgery": needs_surgery,
            "surgery_type": surgery_type,
            "surgery_duration": surg,
            "est_surgery_duration": est_surg,
            "length_of_stay": los,
            "expected_length_of_stay": est_los,
            "treatment_status": "waiting",
            "discharge_status": "not_discharged",
        })

    # -------------------------------------------------------------- resources
    def generate_beds(self) -> pd.DataFrame:
        """ICU beds first, then general beds. All start empty."""
        r = self.cfg["resources"]
        types = ["icu"] * r["num_icu_beds"] + ["general"] * r["num_general_beds"]
        n = len(types)
        return pd.DataFrame({
            "bed_id": np.arange(n),
            "bed_type": types,
            "available": True,
            "current_occupant_id": -1,
            "occupancy_start_time": -1,
            "expected_release_time": -1,
        })

    def generate_doctors(self) -> pd.DataFrame:
        """One row per doctor, grouped by specialization."""
        rows = []
        for spec, count in self.cfg["resources"]["doctors_per_specialization"].items():
            rows += [spec] * int(count)
        n = len(rows)
        return pd.DataFrame({
            "doctor_id": np.arange(n),
            "specialization": rows,
            "available": True,
            "current_workload": 0,
            "scheduled_assignments": 0,
        })

    def generate_operating_rooms(self) -> pd.DataFrame:
        """Operating rooms; every OR supports all configured surgery types."""
        r = self.cfg["resources"]
        n = int(r["num_operating_rooms"])
        return pd.DataFrame({
            "or_id": np.arange(n),
            "available": True,
            "supported_surgery_types": "|".join(r["supported_surgery_types"]),
            "current_surgery_patient_id": -1,
            "surgery_start_time": -1,
            "surgery_expected_end_time": -1,
        })

    # ------------------------------------------------------------ time series
    def generate_arrival_timeseries(self, patients: pd.DataFrame) -> pd.DataFrame:
        """Arrivals per simulation step (the input for the forecasting module later)."""
        steps = pd.RangeIndex(self.num_steps, name="step")
        grp = patients.groupby("arrival_step")
        ts = pd.DataFrame(index=steps)
        ts["total_arrivals"] = grp.size().reindex(steps, fill_value=0)
        ts["emergency_arrivals"] = grp["is_emergency"].sum().reindex(steps, fill_value=0)
        ts["surgery_arrivals"] = grp["requires_surgery"].sum().reindex(steps, fill_value=0)
        ts["elective_arrivals"] = ts["total_arrivals"] - ts["emergency_arrivals"]
        ts = ts.astype(int).reset_index()
        ts.insert(1, "time_minute", ts["step"] * self.step_minutes)
        ts.insert(2, "hour_of_day", (ts["time_minute"] // 60) % 24)
        return ts

    # ------------------------------------------------------------------- all
    def generate_all(self) -> dict[str, pd.DataFrame]:
        events = self.generate_events()
        patients = self.generate_patients(events)
        return {
            "patients": patients,
            "beds": self.generate_beds(),
            "doctors": self.generate_doctors(),
            "operating_rooms": self.generate_operating_rooms(),
            "events": events,
            "arrival_timeseries": self.generate_arrival_timeseries(patients),
        }

    def save_all(self, out_dir: str | Path) -> Path:
        """Write every table as CSV plus the exact config used, into out_dir/<scenario>_seed<N>/."""
        folder = Path(out_dir) / f"{self.scenario}_seed{self.seed}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, df in self.generate_all().items():
            df.to_csv(folder / f"{name}.csv", index=False)
        used = dict(self.cfg)
        used["seed"] = self.seed
        with open(folder / "config_used.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(used, f, sort_keys=False)
        return folder
