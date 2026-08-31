from __future__ import annotations

import numpy as np

from .settings import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED


def bootstrap_mean_ci(values, n_resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for the mean of a per-prompt vector."""
    x = np.asarray(list(values), dtype=float)
    n = x.size
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n": 0, "n_resamples": n_resamples, "seed": seed}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(x.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "std": float(x.std(ddof=1)) if n > 1 else float("nan"),
    }


def ci_within(stat: dict, low: float, high: float) -> bool:
    """True when the whole CI sits inside [low, high] -- the equivalence test."""
    return bool(stat["ci_low"] >= low and stat["ci_high"] <= high)
