from __future__ import annotations

import math

import numpy as np

from .settings import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, TRIM_PROPORTION


def trim_count(n: int, proportion: float = TRIM_PROPORTION) -> int:
    """Observations dropped from each tail; 0 when trimming would empty the sample."""
    k = int(n * proportion)
    return 0 if 2 * k >= n else k


def _statistic(samples: np.ndarray, statistic: str, proportion: float) -> np.ndarray:
    if statistic == "mean":
        return samples.mean(axis=-1)
    if statistic == "median":
        return np.median(samples, axis=-1)
    if statistic == "trimmed_mean":
        k = trim_count(samples.shape[-1], proportion)
        ordered = np.sort(samples, axis=-1)
        return ordered[..., k: samples.shape[-1] - k].mean(axis=-1)
    raise ValueError(f"unknown statistic {statistic!r}")


def bootstrap_ci(values, statistic: str = "mean", n_resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED, alpha: float = 0.05,
                 proportion: float = TRIM_PROPORTION) -> dict:
    """Percentile bootstrap CI for a statistic of a per-prompt vector."""
    x = np.asarray(list(values), dtype=float)
    n = x.size
    if n == 0:
        return {"statistic": statistic, "mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0, "n_resamples": n_resamples, "seed": seed}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    point = _statistic(x, statistic, proportion)
    replicates = _statistic(x[idx], statistic, proportion)
    lo, hi = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    out = {
        "statistic": statistic,
        "mean": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "std": float(x.std(ddof=1)) if n > 1 else float("nan"),
    }
    if statistic == "trimmed_mean":
        out["trim_proportion"] = float(proportion)
        out["n_trimmed_per_tail"] = trim_count(n, proportion)
    return out


def bootstrap_mean_ci(values, n_resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> dict:
    return bootstrap_ci(values, "mean", n_resamples, seed, alpha)


def log_delta_to_pct(stat: dict) -> dict:
    """A difference of mean log-likelihoods as a perplexity ratio, in percent."""
    def pct(value: float) -> float:
        return float("nan") if value != value else 100.0 * (math.exp(value) - 1.0)

    return {"mean": pct(stat["mean"]), "ci_low": pct(stat["ci_low"]),
            "ci_high": pct(stat["ci_high"]), "n": stat["n"],
            "transform": "100 * (exp(d) - 1)"}


def ci_within(stat: dict, low: float, high: float) -> bool:
    """True when the whole CI sits inside [low, high] -- the equivalence test."""
    return bool(stat["ci_low"] >= low and stat["ci_high"] <= high)


def ci_covers(stat: dict, value: float = 0.0) -> bool:
    return bool(stat["ci_low"] <= value <= stat["ci_high"])
