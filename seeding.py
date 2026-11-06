"""Seeding helpers so every experiment is reproducible."""
from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch global random generators."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch is optional for the data/simulation layers
        pass


def make_rng(seed: int, stream: int = 0) -> np.random.Generator:
    """Return an independent NumPy Generator for (seed, stream).

    Different `stream` ids give statistically independent random numbers from the same
    base seed, so e.g. patient generation and noise do not interfere with each other.
    """
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))
