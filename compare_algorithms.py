"""Aggregate per-episode results into comparison tables (means and variability across seeds).

Per-episode rows have: method, train_seed, scenario, seed (evaluation seed) and the metrics.

Statistical unit
----------------
* Learned methods: one TRAINING RUN (train_seed). Per (method, scenario) the metric is first
  averaged over the evaluation seeds of that run; then mean and standard deviation (ddof=1) are
  taken ACROSS TRAINING RUNS. This is the variability that matters when someone retrains the method.
* Rule-based baselines: no training, so the unit is the EVALUATION SEED (std across evaluation seeds).
  Their spread therefore measures scenario randomness, not training randomness, and is not
  directly comparable to the spread of learned methods.
With very few seeds the standard deviation is itself noisy. No significance tests are run, and
differences smaller than the spread should not be read as real.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# metric -> (label, True if lower is better)
METRICS: dict[str, tuple[str, bool]] = {
    "mean_wait_to_treatment": ("Mean wait to treatment (min)", True),
    "emergency_response_delay": ("Emergency response delay (min)", True),
    "emergency_surgery_wait": ("Emergency surgery wait (min)", True),
    "unserved_patients": ("Patients never treated", True),
    "throughput": ("Throughput (discharged)", False),
    "treatments_completed": ("Treatments completed", False),
    "surgeries_completed": ("Surgeries completed", False),
    "bed_utilization": ("Bed utilization", False),
    "doctor_utilization": ("Doctor utilization", False),
    "or_utilization": ("OR utilization", False),
    "invalid_actions_total": ("Invalid actions (conflicts) per episode", True),
    "episode_return": ("Shared episode reward", False),
}
BASELINE_TRAIN_SEED = -1


def _unit(df: pd.DataFrame) -> pd.Series:
    """Statistical unit id: training seed for learned methods, evaluation seed for baselines."""
    return df["train_seed"].where(df["train_seed"] != BASELINE_TRAIN_SEED, 10_000_000 + df["seed"])


def summarize(per_episode: pd.DataFrame, metrics: Optional[Sequence[str]] = None,
              by_scenario: bool = True) -> pd.DataFrame:
    """-> DataFrame indexed by (method[, scenario]) with columns <metric>_mean, <metric>_std, n_units."""
    metrics = [m for m in (metrics or list(METRICS)) if m in per_episode.columns]
    df = per_episode.copy()
    df["unit"] = _unit(df)
    keys = ["method", "scenario"] if by_scenario else ["method"]
    per_unit = df.groupby(keys + ["unit"])[metrics].mean().reset_index()
    grouped = per_unit.groupby(keys)
    out = pd.concat({"mean": grouped[metrics].mean(), "std": grouped[metrics].std(ddof=1)}, axis=1)
    out.columns = [f"{m}_{stat}" for stat, m in out.columns]
    out["n_units"] = grouped.size()
    return out.reset_index()


def relative_to_baseline(summary: pd.DataFrame, metric: str, baseline: str = "fcfs") -> pd.DataFrame:
    """Percent difference of each method's mean to the baseline's mean, per scenario
    (positive = larger value than the baseline; check METRICS to see whether that is good)."""
    pivot = summary.pivot(index="scenario", columns="method", values=f"{metric}_mean")
    return (pivot.div(pivot[baseline], axis=0) - 1.0) * 100.0


def format_table(summary: pd.DataFrame, metric: str, methods: Optional[Sequence[str]] = None,
                 digits: int = 2) -> pd.DataFrame:
    """Scenarios x methods table of 'mean ± std' strings."""
    mean = summary.pivot(index="scenario", columns="method", values=f"{metric}_mean")
    std = summary.pivot(index="scenario", columns="method", values=f"{metric}_std")
    methods = [m for m in (methods or list(mean.columns)) if m in mean.columns]
    table = pd.DataFrame(index=mean.index, columns=methods, dtype=object)
    for m in methods:
        for sc in mean.index:
            mu, sd = mean.loc[sc, m], std.loc[sc, m]
            table.loc[sc, m] = ("n/a" if pd.isna(mu) else
                                f"{mu:.{digits}f}" + ("" if pd.isna(sd) else f" ± {sd:.{digits}f}"))
    return table


def to_markdown(table: pd.DataFrame) -> str:
    cols = [table.index.name or "scenario"] + [str(c) for c in table.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in table.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.tolist()]) + " |")
    return "\n".join(lines)


def training_time_table(runs: pd.DataFrame) -> pd.DataFrame:
    """runs: columns method, train_seed, training_seconds -> mean/std seconds per method."""
    g = runs.groupby("method")["training_seconds"]
    return pd.DataFrame({"mean_seconds": g.mean(), "std_seconds": g.std(ddof=1), "runs": g.size()})


def write_report(per_episode: pd.DataFrame, runs: pd.DataFrame, path, header: str = "") -> str:
    """Markdown report with every table; returns the text and writes it to `path`."""
    methods = list(dict.fromkeys(per_episode["method"]))
    by_sc = summarize(per_episode)
    overall = summarize(per_episode, by_scenario=False)
    parts = ["# Experiment results (simulated data)\n", header,
             "\nValues are mean ± standard deviation. Learned methods: across independent training runs "
             "(each run first averaged over evaluation seeds). Baselines: across evaluation seeds. "
             "No significance tests were run; differences smaller than the spread should not be "
             "interpreted.\n"]
    ov = pd.DataFrame(index=pd.Index([METRICS[m][0] for m in METRICS if f"{m}_mean" in overall], name="metric"),
                      columns=methods, dtype=object)
    for m in METRICS:
        if f"{m}_mean" not in overall:
            continue
        for meth in methods:
            row = overall[overall.method == meth]
            mu, sd = row[f"{m}_mean"].iloc[0], row[f"{m}_std"].iloc[0]
            ov.loc[METRICS[m][0], meth] = "n/a" if pd.isna(mu) else (
                f"{mu:.2f}" + ("" if pd.isna(sd) else f" ± {sd:.2f}"))
    parts.append("\n## Overall (average over all evaluated scenarios)\n\n" + to_markdown(ov) + "\n")
    for m in METRICS:
        if f"{m}_mean" not in by_sc or by_sc[f"{m}_mean"].isna().all():
            continue
        direction = "lower is better" if METRICS[m][1] else "higher is better"
        parts.append(f"\n## {METRICS[m][0]} ({direction})\n\n" + to_markdown(format_table(by_sc, m, methods)) + "\n")
    if "fcfs" in methods:
        for m in ("mean_wait_to_treatment", "emergency_response_delay"):
            if f"{m}_mean" not in by_sc or by_sc[f"{m}_mean"].isna().all():
                continue
            rel = relative_to_baseline(by_sc, m).round(1)
            parts.append(f"\n## {METRICS[m][0]}: percent difference to FCFS (negative = lower than FCFS)\n\n"
                         + to_markdown(rel.astype(str)) + "\n")
    if len(runs):
        tt = training_time_table(runs).round(1)
        parts.append("\n## Training time per run (seconds, wall clock on the machine that ran it)\n\n"
                     + to_markdown(tt.astype(str).rename_axis("method")) + "\n")
    text = "\n".join(parts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
