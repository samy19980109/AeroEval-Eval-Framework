"""Numeric utility functions for NaN/Inf guard operations."""

from __future__ import annotations

import numpy as np


def check_finite(values: list[float]) -> dict[str, int]:
    """Check a list of values for NaN and Inf occurrences."""
    arr = np.array(values, dtype=np.float64)
    return {
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "finite_count": int(np.isfinite(arr).sum()),
        "total": len(arr),
    }


def compute_drift(
    baseline: list[float],
    current: list[float],
    sigma_threshold: float = 3.0,
) -> dict[str, float | bool]:
    """Compare two logprob distributions for drift.

    Returns drift metrics including whether drift exceeds the sigma threshold.
    """
    b = np.array(baseline, dtype=np.float64)
    c = np.array(current, dtype=np.float64)

    b_clean = b[np.isfinite(b)]
    c_clean = c[np.isfinite(c)]

    if len(b_clean) == 0 or len(c_clean) == 0:
        return {"drift_detected": False, "error": "insufficient data"}

    b_mean = float(np.mean(b_clean))
    b_std = float(np.std(b_clean))
    c_mean = float(np.mean(c_clean))
    c_std = float(np.std(c_clean))

    if b_std == 0:
        return {
            "drift_detected": False,
            "baseline_mean": b_mean,
            "current_mean": c_mean,
            "mean_shift_sigma": 0.0,
        }

    mean_shift = abs(c_mean - b_mean) / b_std
    std_ratio = c_std / b_std if b_std > 0 else 0.0

    return {
        "drift_detected": mean_shift > sigma_threshold,
        "mean_shift_sigma": mean_shift,
        "std_ratio": std_ratio,
        "baseline_mean": b_mean,
        "baseline_std": b_std,
        "current_mean": c_mean,
        "current_std": c_std,
    }
