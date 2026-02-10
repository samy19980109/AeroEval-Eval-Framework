"""L4 Scorers: System/hardware performance metrics."""

from __future__ import annotations

import numpy as np
from deepeval.test_case import LLMTestCase

from aero_eval.models import L4SystemConfig, ScorerTier
from aero_eval.registry import ScorerRegistry
from aero_eval.scorers.base import BaseScorer


@ScorerRegistry.register("ttft", ScorerTier.L4)
class TTFTScorer(BaseScorer):
    """Time-to-first-token scorer.

    Expects test_case metadata to contain 'ttft_ms' from an InferenceResult.
    Score is 1.0 if TTFT <= threshold, scaled down linearly otherwise.
    """

    tier = ScorerTier.L4

    def __init__(self, config: L4SystemConfig):
        super().__init__(config)
        self._ttft_threshold = config.ttft_threshold_ms

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        ttft = None
        if hasattr(test_case, "additional_metadata") and test_case.additional_metadata:
            ttft = test_case.additional_metadata.get("ttft_ms")

        if ttft is None:
            self.score = 1.0
            self.reason = "No TTFT data available; skipping"
            self.is_successful()
            return self.score

        if ttft <= self._ttft_threshold:
            self.score = 1.0
        else:
            # Linear decay: score drops to 0 at 2x threshold
            self.score = max(0.0, 1.0 - (ttft - self._ttft_threshold) / self._ttft_threshold)

        self.reason = f"TTFT={ttft:.1f}ms (threshold={self._ttft_threshold:.1f}ms)"
        self.score_breakdown = {"ttft_ms": ttft, "threshold_ms": self._ttft_threshold}
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "L4::TTFT"


@ScorerRegistry.register("latency_p99", ScorerTier.L4)
class LatencyP99Scorer(BaseScorer):
    """P99 latency scorer.

    Expects test_case metadata to contain 'latency_ms'.
    Accumulates latencies across calls and computes P99.
    """

    tier = ScorerTier.L4
    _latencies: list[float] = []

    def __init__(self, config: L4SystemConfig):
        super().__init__(config)
        self._p99_threshold = config.p99_threshold_ms
        self._latencies = []

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        latency = None
        if hasattr(test_case, "additional_metadata") and test_case.additional_metadata:
            latency = test_case.additional_metadata.get("total_latency_ms")

        if latency is None:
            self.score = 1.0
            self.reason = "No latency data available; skipping"
            self.is_successful()
            return self.score

        self._latencies.append(latency)

        if len(self._latencies) >= 2:
            p99 = float(np.percentile(self._latencies, 99))
        else:
            p99 = latency

        if p99 <= self._p99_threshold:
            self.score = 1.0
        else:
            self.score = max(0.0, 1.0 - (p99 - self._p99_threshold) / self._p99_threshold)

        self.reason = f"P99={p99:.1f}ms (threshold={self._p99_threshold:.1f}ms, samples={len(self._latencies)})"
        self.score_breakdown = {
            "p99_ms": p99,
            "threshold_ms": self._p99_threshold,
            "sample_count": float(len(self._latencies)),
        }
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "L4::LatencyP99"


@ScorerRegistry.register("throughput", ScorerTier.L4)
class ThroughputScorer(BaseScorer):
    """Throughput scorer (tokens per second).

    Expects test_case metadata to contain 'tokens_generated' and 'total_latency_ms'.
    """

    tier = ScorerTier.L4

    def __init__(self, config: L4SystemConfig):
        super().__init__(config)
        self._min_tps = config.throughput_min_tps

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        tokens = None
        latency = None

        if hasattr(test_case, "additional_metadata") and test_case.additional_metadata:
            tokens = test_case.additional_metadata.get("tokens_generated")
            latency = test_case.additional_metadata.get("total_latency_ms")

        if tokens is None or latency is None or latency == 0:
            self.score = 1.0
            self.reason = "No throughput data available; skipping"
            self.is_successful()
            return self.score

        tps = tokens / (latency / 1000.0)

        if tps >= self._min_tps:
            self.score = 1.0
        else:
            self.score = max(0.0, tps / self._min_tps)

        self.reason = f"Throughput={tps:.1f} tok/s (min={self._min_tps:.1f} tok/s)"
        self.score_breakdown = {
            "tokens_per_second": tps,
            "min_tps": self._min_tps,
            "tokens": float(tokens),
            "latency_ms": latency,
        }
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "L4::Throughput"
