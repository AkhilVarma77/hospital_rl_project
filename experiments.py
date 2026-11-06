"""Run the complete experiment suite: baselines, independent PPO, cooperative PPO (without /
with attention), attention + forecasting, and DQN.

Design (so the comparison is as fair as the project allows)
* Every learned method is trained for the same number of decision rounds on the SAME mixture of
  scenarios, with several independent training seeds, using standard UNTUNED hyperparameters.
* Every method (baselines included) is evaluated by the same code, on the same simulator, scenarios
  and evaluation seeds (1000+, never used in training).
* `comm_none` is the same network as the communicating agents with communication switched off, so
  the effect of communication is measured against it (not against `ppo_independent`, which has a
  different architecture).
* The forecaster is the LSTM trained by scripts/run_forecaster.py (trained automatically if missing).
Runs are resumable: finished training/evaluation runs are reused, so the suite can be stopped and
restarted. Changing evaluation settings requires a new output folder (checked).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import yaml

from src.baselines.registry import make_baseline
from src.evaluation.compare_algorithms import BASELINE_TRAIN_SEED, summarize, write_report
from src.evaluation.evaluate import evaluate_ma_policy, evaluate_policy
from src.training.train_comm_ppo import load_comm_policy_set, train_comm_ppo
from src.training.train_dqn import load_dqn_policy_set, train_dqn
from src.training.train_ppo import load_policy_set, train_ppo
from src.visualization.plot_comparison import plot_metric_by_scenario
from src.visualization.plot_training import plot_attention_weights, plot_learning_curves

METHODS: dict[str, dict[str, Any]] = {
    "fcfs": {"kind": "baseline"},
    "severity": {"kind": "baseline"},
    "ppo_independent": {"kind": "learned", "algo": "ppo"},
    "comm_none": {"kind": "learned", "algo": "comm_ppo", "mode": "none"},
    "comm_mean": {"kind": "learned", "algo": "comm_ppo", "mode": "mean"},
    "comm_attention": {"kind": "learned", "algo": "comm_ppo", "mode": "attention"},
    "comm_attention_forecast": {"kind": "learned", "algo": "comm_ppo", "mode": "attention", "forecast": True},
    "dqn": {"kind": "learned", "algo": "dqn"},
}
PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {"seeds": 1, "iterations": 2, "rollout_steps": 256, "eval_seeds": 1,
              "scenarios": ["normal", "bed_shortage"]},
    "default": {"seeds": 3, "iterations": 40, "rollout_steps": None, "eval_seeds": 5, "scenarios": None},
    "full": {"seeds": 5, "iterations": 100, "rollout_steps": None, "eval_seeds": 10, "scenarios": None},
}
FIGURE_METRICS = ["mean_wait_to_treatment", "emergency_response_delay", "throughput", "unserved_patients",
                  "invalid_actions_total", "bed_utilization", "doctor_utilization", "or_utilization"]
_LOADERS = {"ppo": load_policy_set, "comm_ppo": load_comm_policy_set, "dqn": load_dqn_policy_set}


def _train(spec: dict[str, Any], config: dict[str, Any], train_scenarios: list[str], run_dir: Path,
           seed: int, iterations: int, rollout_steps: Optional[int], forecast_spec: Optional[dict]):
    common = dict(seed=seed, checkpoint_every=0, verbose=False, forecast_spec=forecast_spec)
    if spec["algo"] == "ppo":
        return train_ppo(config, train_scenarios, run_dir, iterations,
                         ppo_overrides={"rollout_steps": rollout_steps} if rollout_steps else None, **common)
    if spec["algo"] == "comm_ppo":
        return train_comm_ppo(config, train_scenarios, run_dir, iterations, mode=spec["mode"],
                              ppo_overrides={"rollout_steps": rollout_steps} if rollout_steps else None, **common)
    return train_dqn(config, train_scenarios, run_dir, iterations,
                     dqn_overrides={"steps_per_iteration": rollout_steps} if rollout_steps else None, **common)


def _check_signature(out: Path, signature: dict[str, Any]) -> None:
    path = out / "experiment_signature.yaml"
    if path.exists():
        old = yaml.safe_load(path.read_text(encoding="utf-8"))
        if old != signature:
            diff = {k: (old.get(k), signature.get(k)) for k in signature if old.get(k) != signature.get(k)}
            raise ValueError(f"settings differ from the existing runs in {out} (old, new): {diff}. "
                             "Use a new --out-dir (results from different settings must not be mixed).")
    else:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(signature, sort_keys=True), encoding="utf-8")


def run_experiments(config: dict[str, Any], out_dir: str | Path, methods: Sequence[str],
                    num_seeds: int, iterations: int, eval_seeds: Sequence[int],
                    scenarios: Sequence[str], train_scenarios: Optional[Sequence[str]] = None,
                    rollout_steps: Optional[int] = None,
                    forecast_model: str = "results/forecaster/lstm.pt", seed_start: int = 0,
                    aggregate_only: bool = False, verbose: bool = True) -> dict[str, Any]:
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise KeyError(f"unknown methods {unknown}; available: {list(METHODS)}")
    out = Path(out_dir)
    runs_dir = out / "runs"
    train_scenarios = list(train_scenarios or scenarios)
    _check_signature(out, {"iterations": iterations, "rollout_steps": rollout_steps,
                           "eval_seeds": list(eval_seeds), "scenarios": list(scenarios),
                           "train_scenarios": train_scenarios, "forecast_model": forecast_model})
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    forecast_spec = {"path": forecast_model}
    if not aggregate_only and any(METHODS[m].get("forecast") for m in methods) and not Path(forecast_model).exists():
        from src.forecasting.train_forecaster import run_forecasting_experiment
        log(f"forecaster {forecast_model} not found -> training it first")
        run_forecasting_experiment(config, Path(forecast_model).parent, verbose=False)

    seeds = list(range(seed_start, seed_start + num_seeds))
    if not aggregate_only:
        for method in methods:
            spec = METHODS[method]
            if spec["kind"] == "baseline":
                f = runs_dir / method / "eval_per_episode.csv"
                if f.exists():
                    continue
                log(f"[{method}] evaluating on {len(scenarios)} scenarios x {len(eval_seeds)} seeds")
                frames = [evaluate_policy(config, sc, lambda m=method: make_baseline(m), eval_seeds, method)
                          for sc in scenarios]
                df = pd.concat(frames, ignore_index=True).rename(columns={"policy": "method"})
                df["train_seed"] = BASELINE_TRAIN_SEED
                f.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(f, index=False)
                continue
            for seed in seeds:
                run_dir = runs_dir / f"{method}_seed{seed}"
                f = run_dir / "eval_per_episode.csv"
                if f.exists():
                    continue
                ckpt = run_dir / "checkpoints" / "final.pt"
                if ckpt.exists():
                    policies = _LOADERS[spec["algo"]](ckpt)
                    log(f"[{method} seed {seed}] re-using trained model")
                else:
                    log(f"[{method} seed {seed}] training {iterations} iterations ...")
                    t0 = time.perf_counter()
                    policies = _train(spec, config, train_scenarios, run_dir, seed, iterations, rollout_steps,
                                      forecast_spec if spec.get("forecast") else None)
                    log(f"[{method} seed {seed}] trained in {time.perf_counter() - t0:.0f}s")
                log(f"[{method} seed {seed}] evaluating")
                frames = [evaluate_ma_policy(config, sc, policies, eval_seeds, method) for sc in scenarios]
                df = pd.concat(frames, ignore_index=True).rename(columns={"policy": "method"})
                df["train_seed"] = seed
                df.to_csv(f, index=False)

    return aggregate(out, methods, log=log)


def aggregate(out_dir: str | Path, methods: Sequence[str], log=print) -> dict[str, Any]:
    """Combine all finished evaluations of `methods`, write tables, report and figures."""
    out = Path(out_dir)
    runs_dir = out / "runs"
    frames, run_rows = [], []
    for method in methods:
        for f in sorted(runs_dir.glob(f"{method}/eval_per_episode.csv")) + \
                sorted(runs_dir.glob(f"{method}_seed*/eval_per_episode.csv")):
            frames.append(pd.read_csv(f))
            tl = f.parent / "train_log.csv"
            if tl.exists():
                run_rows.append({"method": method, "train_seed": int(f.parent.name.rsplit("seed", 1)[1]),
                                 "training_seconds": float(pd.read_csv(tl)["elapsed_seconds"].iloc[-1])})
    if not frames:
        raise RuntimeError(f"no finished evaluations found in {runs_dir}")
    per_episode = pd.concat(frames, ignore_index=True)
    per_episode["method"] = pd.Categorical(per_episode["method"], categories=[m for m in methods], ordered=True)
    per_episode = per_episode.sort_values(["method", "train_seed", "scenario", "seed"]).reset_index(drop=True)
    per_episode["method"] = per_episode["method"].astype(str)
    runs = pd.DataFrame(run_rows, columns=["method", "train_seed", "training_seconds"])

    per_episode.to_csv(out / "per_episode.csv", index=False)
    by_sc = summarize(per_episode)
    by_sc.to_csv(out / "summary_by_scenario.csv", index=False)
    summarize(per_episode, by_scenario=False).to_csv(out / "summary_overall.csv", index=False)
    sig = yaml.safe_load((out / "experiment_signature.yaml").read_text(encoding="utf-8"))
    header = (f"Training: {sig['iterations']} iterations per run on scenarios {sig['train_scenarios']}; "
              f"evaluation: scenarios {sig['scenarios']}, {len(sig['eval_seeds'])} evaluation seeds "
              f"({sig['eval_seeds'][0]}..{sig['eval_seeds'][-1]}). Methods: {list(dict.fromkeys(per_episode.method))}.")
    write_report(per_episode, runs, out / "results_summary.md", header)
    for metric in FIGURE_METRICS:
        if f"{metric}_mean" in by_sc and not by_sc[f"{metric}_mean"].isna().all():
            plot_metric_by_scenario(by_sc, metric, out / "figures" / f"{metric}.png", methods=list(methods))
    if any((runs_dir).glob("*_seed*/train_log.csv")):
        plot_learning_curves(runs_dir, out / "figures" / "learning_curves.png")
        plot_attention_weights(runs_dir, out / "figures" / "attention_weights.png")
    log(f"results written to {out}")
    return {"per_episode": per_episode, "by_scenario": by_sc, "runs": runs, "out_dir": out}
