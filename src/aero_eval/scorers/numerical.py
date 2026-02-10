"""Numerical Stability Guard — the "Cerebras Special".

Detects NaN/Inf values and logit drift via 3-sigma rule
on rolling windows of logprob sequences.
"""

from __future__ import annotations

import numpy as np
from deepeval.test_case import LLMTestCase

from aero_eval.models import L4SystemConfig, ScorerTier
from aero_eval.registry import ScorerRegistry
from aero_eval.scorers.base import BaseScorer


@ScorerRegistry.register("numerical_stability", ScorerTier.L4)
class NumericalStabilityScorer(BaseScorer):
    """Checks for NaN/Inf in logprobs and detects logit drift via 3-sigma rule.

    This is designed for large-scale hardware (e.g., Cerebras) where numerical
    overflow is a common concern. The scorer:
    1. Checks for NaN values in logprobs
    2. Checks for Inf values in logprobs
    3. Detects logit drift using a rolling window 3-sigma test

    If the standard deviation of logits shifts by more than 3σ compared to
    the window baseline, it flags a "Numerical Regression."

    Expects logprobs in test_case.additional_metadata["logprobs"].
    """

    tier = ScorerTier.L4

    def __init__(self, config: L4SystemConfig):
        super().__init__(config)
        self._sigma_threshold = config.sigma_threshold
        self._window_size = config.window_size

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        logprobs: list[float] | None = None

        if hasattr(test_case, "additional_metadata") and test_case.additional_metadata:
            logprobs = test_case.additional_metadata.get("logprobs")

        if not logprobs:
            self.score = 1.0
            self.reason = "No logprobs available; skipping numerical checks"
            self.is_successful()
            return self.score

        arr = np.array(logprobs, dtype=np.float64)
        issues: list[str] = []

        # NaN check
        nan_count = int(np.isnan(arr).sum())
        if nan_count > 0:
            issues.append(f"{nan_count} NaN values detected")

        # Inf check
        inf_count = int(np.isinf(arr).sum())
        if inf_count > 0:
            issues.append(f"{inf_count} Inf values detected")

        # 3-sigma logit drift detection
        clean = arr[np.isfinite(arr)]
        if len(clean) >= self._window_size:
            # Use the first half as baseline, second half as test
            baseline = clean[: len(clean) // 2]
            test_window = clean[len(clean) // 2 :]

            baseline_mean = float(np.mean(baseline))
            baseline_std = float(np.std(baseline))

            if baseline_std > 0:
                # Check how many points in test window exceed sigma threshold
                deviations = np.abs(test_window - baseline_mean) / baseline_std
                outlier_count = int(np.sum(deviations > self._sigma_threshold))

                if outlier_count > 0:
                    outlier_pct = outlier_count / len(test_window) * 100
                    issues.append(
                        f"{outlier_count} logit drift outliers "
                        f"({outlier_pct:.1f}% of window, >{self._sigma_threshold}σ)"
                    )

                # Also check if overall mean has shifted significantly
                test_mean = float(np.mean(test_window))
                mean_shift_sigma = abs(test_mean - baseline_mean) / baseline_std
                if mean_shift_sigma > self._sigma_threshold:
                    issues.append(
                        f"Mean logit shift: {mean_shift_sigma:.2f}σ "
                        f"(baseline={baseline_mean:.3f}, current={test_mean:.3f})"
                    )
        elif len(clean) > 0:
            # Window too small for drift detection, just check basic stats
            std = float(np.std(clean))
            mean = float(np.mean(clean))
            self.score_breakdown = {
                "mean_logprob": mean,
                "std_logprob": std,
                "sample_size": float(len(clean)),
            }

        # Score calculation
        if not issues:
            self.score = 1.0
            self.reason = "No numerical issues detected"
        else:
            # Penalize based on severity
            penalty = 0.0
            if nan_count > 0:
                penalty += 0.5  # NaN is severe
            if inf_count > 0:
                penalty += 0.3  # Inf is severe
            # Drift is proportional
            if len(issues) > (1 if nan_count else 0) + (1 if inf_count else 0):
                penalty += 0.2

            self.score = max(0.0, 1.0 - penalty)
            self.reason = "NUMERICAL REGRESSION: " + "; ".join(issues)

        self.score_breakdown = self.score_breakdown or {}
        self.score_breakdown.update({
            "nan_count": float(nan_count),
            "inf_count": float(inf_count),
            "total_logprobs": float(len(arr)),
            "clean_logprobs": float(len(clean)) if len(clean) > 0 else 0.0,
        })

        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "L4::NumericalStability"


@ScorerRegistry.register("logit_drift", ScorerTier.L4)
class LogitDriftScorer(BaseScorer):
    """Focused logit drift detection using rolling window comparison.

    Compares the distribution of logprobs from the current inference
    against a stored baseline to detect model degradation.
    """

    tier = ScorerTier.L4

    def __init__(self, config: L4SystemConfig):
        super().__init__(config)
        self._sigma_threshold = config.sigma_threshold
        self._window_size = config.window_size
        self._baseline_stats: dict[str, float] | None = None

    def set_baseline(self, logprobs: list[float]) -> None:
        """Set baseline statistics from a known-good model run."""
        arr = np.array(logprobs, dtype=np.float64)
        clean = arr[np.isfinite(arr)]
        if len(clean) > 0:
            self._baseline_stats = {
                "mean": float(np.mean(clean)),
                "std": float(np.std(clean)),
                "count": float(len(clean)),
            }

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        logprobs: list[float] | None = None
        if hasattr(test_case, "additional_metadata") and test_case.additional_metadata:
            logprobs = test_case.additional_metadata.get("logprobs")

        if not logprobs:
            self.score = 1.0
            self.reason = "No logprobs available"
            self.is_successful()
            return self.score

        arr = np.array(logprobs, dtype=np.float64)
        clean = arr[np.isfinite(arr)]

        if len(clean) == 0:
            self.score = 0.0
            self.reason = "All logprobs are NaN/Inf"
            self.is_successful()
            return self.score

        current_mean = float(np.mean(clean))
        current_std = float(np.std(clean))

        if self._baseline_stats and self._baseline_stats["std"] > 0:
            mean_shift = abs(current_mean - self._baseline_stats["mean"]) / self._baseline_stats["std"]
            std_ratio = current_std / self._baseline_stats["std"]

            drift_detected = mean_shift > self._sigma_threshold or std_ratio > 2.0

            if drift_detected:
                self.score = max(0.0, 1.0 - mean_shift / (self._sigma_threshold * 2))
                self.reason = (
                    f"Logit drift detected: mean_shift={mean_shift:.2f}σ, "
                    f"std_ratio={std_ratio:.2f}"
                )
            else:
                self.score = 1.0
                self.reason = f"No drift: mean_shift={mean_shift:.2f}σ, std_ratio={std_ratio:.2f}"

            self.score_breakdown = {
                "mean_shift_sigma": mean_shift,
                "std_ratio": std_ratio,
                "current_mean": current_mean,
                "baseline_mean": self._baseline_stats["mean"],
            }
        else:
            # No baseline — just report stats
            self.score = 1.0
            self.reason = f"No baseline set; current stats: mean={current_mean:.3f}, std={current_std:.3f}"
            self.score_breakdown = {
                "current_mean": current_mean,
                "current_std": current_std,
            }

        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "L4::LogitDrift"
