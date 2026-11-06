"""Name -> policy class lookup for the rule-based baselines."""
from __future__ import annotations

from src.baselines.base import DispatchPolicy
from src.baselines.fcfs import FCFSPolicy
from src.baselines.severity_priority import SeverityPriorityPolicy

BASELINES: dict[str, type[DispatchPolicy]] = {
    FCFSPolicy.name: FCFSPolicy,
    SeverityPriorityPolicy.name: SeverityPriorityPolicy,
}


def make_baseline(name: str) -> DispatchPolicy:
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline '{name}'. Available: {sorted(BASELINES)}")
    return BASELINES[name]()
