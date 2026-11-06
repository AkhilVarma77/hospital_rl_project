"""Demand forecasting: replaceable predictors of the NEXT step's patient arrivals.

Target: [total arrivals, emergency arrivals] in the next simulation step (15 min).
Input : counts of the previous `window` steps (+ time of day). Nothing else.

No leakage: at decision time t = k * step_minutes the history covers steps k-window .. k-1, i.e.
patients with arrival_time < t. The step being predicted (k, arrival_time in [t, t+step)) is never
visible. At the start of an episode the history is zero-padded and marked invalid.

The forecaster only PREDICTS. It never allocates beds, schedules doctors or assigns operating
rooms; the agents merely receive its output as two extra observation features.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TARGETS = 2   # total arrivals, emergency arrivals


# ----------------------------------------------------------------------------- windows
def make_windows(counts: np.ndarray, window: int) -> np.ndarray:
    """counts [T, 2] (per step) -> histories [T, window, 2]; histories[k] = steps k-window..k-1.

    The row for step k never contains counts[k] or anything later. Steps before the start of the
    episode are zero-padded.
    """
    T = counts.shape[0]
    padded = np.vstack([np.zeros((window, counts.shape[1])), counts.astype(float)])
    return np.stack([padded[k:k + window] for k in range(T)])


# --------------------------------------------------------------------------- base class
class DemandForecaster:
    name = "base"
    window: int = 1
    steps_per_day: int = 96

    def predict_batch(self, histories: np.ndarray, step_indices: np.ndarray) -> np.ndarray:
        """histories [B, window, 2], step_indices [B] (the step to predict) -> [B, 2]."""
        raise NotImplementedError

    def predict(self, history: np.ndarray, step_index: int) -> np.ndarray:
        return self.predict_batch(history[None], np.asarray([step_index]))[0]

    def save(self, path: str | Path) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------- moving average
class MovingAverageForecaster(DemandForecaster):
    """Predict the mean of the last `window` observed steps (fewer at the start of an episode).
    With no history at all it returns `prior` (the mean of the training counts)."""
    name = "moving_average"

    def __init__(self, window: int, steps_per_day: int, prior: Sequence[float] = (0.0, 0.0)) -> None:
        self.window, self.steps_per_day = int(window), int(steps_per_day)
        self.prior = np.asarray(prior, dtype=float)

    def predict_batch(self, histories: np.ndarray, step_indices: np.ndarray) -> np.ndarray:
        k = np.asarray(step_indices)
        n_avail = np.minimum(k, self.window)
        sums = histories[:, -self.window:, :].sum(axis=1)             # padded rows are zero
        pred = sums / np.maximum(n_avail, 1)[:, None]
        pred[n_avail == 0] = self.prior
        return pred

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"kind": "moving_average", "window": self.window,
                                          "steps_per_day": self.steps_per_day,
                                          "prior": self.prior.tolist()}, indent=2))


def fit_moving_average(histories: np.ndarray, step_indices: np.ndarray, targets: np.ndarray,
                       candidates: Sequence[int], steps_per_day: int) -> tuple["MovingAverageForecaster", dict[int, float]]:
    """Choose the window with the lowest MAE on VALIDATION data (ties -> smaller window)."""
    prior = targets.mean(axis=0)
    scores: dict[int, float] = {}
    for m in candidates:
        fc = MovingAverageForecaster(m, steps_per_day, prior)
        scores[m] = float(np.abs(fc.predict_batch(histories, step_indices) - targets).mean())
    best = min(candidates, key=lambda m: (scores[m], m))
    return MovingAverageForecaster(best, steps_per_day, prior), scores


# ------------------------------------------------------------------------------- LSTM
def build_features(histories: np.ndarray, step_indices: np.ndarray, window: int,
                   steps_per_day: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """-> x [B, window, 5] = (total/scale, emergency/scale, sin tod, cos tod, valid) per history step,
          static [B, 2] = (sin, cos) of the time of day of the step being predicted."""
    k = np.asarray(step_indices)
    steps = k[:, None] - window + np.arange(window)[None, :]          # absolute step of each row
    valid = (steps >= 0).astype(np.float32)
    ang = 2.0 * math.pi * (np.mod(steps, steps_per_day) / steps_per_day)
    x = np.stack([histories[..., 0] / scale, histories[..., 1] / scale,
                  np.sin(ang), np.cos(ang), valid], axis=-1).astype(np.float32)
    ang_t = 2.0 * math.pi * (np.mod(k, steps_per_day) / steps_per_day)
    static = np.stack([np.sin(ang_t), np.cos(ang_t)], axis=-1).astype(np.float32)
    return x, static


class LSTMNet(nn.Module):
    """LSTM over the history, then an MLP on [last hidden state, time of day of the target step].
    Output = predicted counts / scale (softplus keeps them non-negative)."""

    def __init__(self, hidden: int = 32, n_features: int = 5) -> None:
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(), nn.Linear(hidden, N_TARGETS))

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return F.softplus(self.head(torch.cat([h[-1], static], dim=-1)))


class LSTMDemandForecaster(DemandForecaster):
    name = "lstm"

    def __init__(self, net: LSTMNet, window: int, steps_per_day: int, scale: float) -> None:
        self.net, self.window, self.steps_per_day, self.scale = net, int(window), int(steps_per_day), float(scale)

    @torch.no_grad()
    def predict_batch(self, histories: np.ndarray, step_indices: np.ndarray) -> np.ndarray:
        self.net.eval()
        x, static = build_features(histories, step_indices, self.window, self.steps_per_day, self.scale)
        return self.net(torch.as_tensor(x), torch.as_tensor(static)).numpy().astype(float) * self.scale

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(),
                    "meta": {"window": self.window, "steps_per_day": self.steps_per_day,
                             "scale": self.scale, "hidden": self.net.hidden}}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMDemandForecaster":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        m = ckpt["meta"]
        net = LSTMNet(m["hidden"])
        net.load_state_dict(ckpt["state"])
        return cls(net, m["window"], m["steps_per_day"], m["scale"])


@lru_cache(maxsize=16)
def _load_cached(path: str, mtime: float) -> DemandForecaster:
    if path.endswith(".json"):
        d = json.loads(Path(path).read_text())
        return MovingAverageForecaster(d["window"], d["steps_per_day"], d["prior"])
    if path.endswith(".pt"):
        return LSTMDemandForecaster.load(path)
    raise ValueError(f"forecaster file must end in .json or .pt: {path}")


def load_forecaster(path: str | Path) -> DemandForecaster:
    p = Path(path)
    return _load_cached(str(p), p.stat().st_mtime)


# ------------------------------------------------------------- adapter to the environment
class ForecastProvider:
    """Callable used by HospitalMAEnv: HospitalEnv -> two forecast features (counts / scale).

    Builds the history ONLY from patients with arrival_time < now, so nothing about the current or
    future step leaks. The result is cached per clock step (several decision rounds share one step).
    """

    def __init__(self, forecaster: DemandForecaster, step_minutes: int = 15, scale: float = 3.0) -> None:
        self.forecaster, self.step_minutes, self.scale = forecaster, int(step_minutes), float(scale)
        self._key: Optional[tuple[int, int]] = None
        self._value: tuple[float, float] = (0.0, 0.0)

    def history(self, env: Any) -> tuple[np.ndarray, int]:
        """(history [window, 2], k) with k = index of the step about to be predicted."""
        state, W = env.state, self.forecaster.window
        t = state.time
        k = t // self.step_minutes
        hist = np.zeros((W, 2))
        for p in state.patients.values():
            if p.arrival_time < t:
                idx = p.arrival_time // self.step_minutes - (k - W)
                if 0 <= idx < W:
                    hist[idx, 0] += 1
                    hist[idx, 1] += p.is_emergency
        return hist, k

    def __call__(self, env: Any) -> tuple[float, float]:
        state = env.state
        key = (id(state), state.time)
        if key != self._key:
            hist, k = self.history(env)
            pred = self.forecaster.predict(hist, k) / self.scale
            self._key, self._value = key, (float(pred[0]), float(pred[1]))
        return self._value


def make_forecast_provider(spec: Optional[dict[str, Any]], step_minutes: int = 15) -> Optional[ForecastProvider]:
    """spec = {"path": "results/forecaster/lstm.pt"} (or a .json moving average); None -> no forecast."""
    if not spec:
        return None
    return ForecastProvider(load_forecaster(spec["path"]), step_minutes=step_minutes,
                            scale=float(spec.get("scale", 3.0)))
