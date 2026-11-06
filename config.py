"""Loading, merging and validating YAML configuration."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default_config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file (defaults to config/default_config.yaml)."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`. Lists/scalars are replaced."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_scenario_config(config: dict[str, Any], scenario: str = "normal") -> dict[str, Any]:
    """Return the base config merged with the named scenario's overrides (validated)."""
    scenarios = config.get("scenarios", {})
    if scenario not in scenarios:
        raise KeyError(f"Unknown scenario '{scenario}'. Available: {sorted(scenarios)}")
    base = {k: v for k, v in config.items() if k != "scenarios"}
    merged = deep_merge(base, scenarios[scenario] or {})
    merged["scenario_name"] = scenario
    validate_config(merged)
    return merged


def _check_probs(name: str, probs: Any) -> None:
    values = list(probs.values()) if isinstance(probs, dict) else list(probs)
    if any(p < 0 for p in values) or abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"'{name}' must be non-negative and sum to 1 (got sum={sum(values)})")


def validate_config(cfg: dict[str, Any]) -> None:
    """Raise ValueError if the configuration is inconsistent."""
    step = cfg["simulation"]["step_minutes"]
    if step <= 0 or 1440 % step != 0:
        raise ValueError("simulation.step_minutes must be positive and divide 1440")
    if cfg["simulation"]["episode_days"] <= 0:
        raise ValueError("simulation.episode_days must be positive")

    arr = cfg["arrivals"]
    for key in ("emergency_hourly_profile", "elective_hourly_profile"):
        if len(arr[key]) != 24 or sum(arr[key]) <= 0:
            raise ValueError(f"arrivals.{key} needs 24 values with a positive sum")
    if not 0.0 <= arr["emergency_fraction"] <= 1.0:
        raise ValueError("arrivals.emergency_fraction must be in [0, 1]")

    pat = cfg["patients"]
    _check_probs("severity_probs_emergency", pat["severity_probs_emergency"])
    _check_probs("severity_probs_elective", pat["severity_probs_elective"])
    _check_probs("specialization_probs_emergency", pat["specialization_probs_emergency"])
    _check_probs("specialization_probs_elective", pat["specialization_probs_elective"])
    if set(pat["specialization_probs_emergency"]) != set(pat["specialization_probs_elective"]):
        raise ValueError("emergency/elective specialization lists must match")
    for key in ("icu_prob_by_severity", "treatment_median_minutes_by_severity"):
        if len(pat[key]) != 5:
            raise ValueError(f"patients.{key} needs 5 values (severity 1..5)")

    specs = set(pat["specialization_probs_emergency"])
    if not set(pat["surgery_type_by_specialization"]) <= specs:
        raise ValueError("surgery_type_by_specialization uses an unknown specialization")
    if not set(cfg["resources"]["doctors_per_specialization"]) <= specs:
        raise ValueError("doctors_per_specialization uses an unknown specialization")

    for ev in cfg.get("events", []):
        if ev["type"] not in ("emergency_surge", "elective_increase", "resource_outage"):
            raise ValueError(f"Unknown event type '{ev['type']}'")
