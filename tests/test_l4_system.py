"""Tests for L4 system/hardware scorers."""

import pytest
from deepeval.test_case import LLMTestCase

from aero_eval.models import L4SystemConfig
from aero_eval.scorers.l4_system import (
    LatencyP99Scorer,
    ThroughputScorer,
    TTFTScorer,
)


def _make_case_with_metadata(metadata: dict) -> LLMTestCase:
    case = LLMTestCase(input="test", actual_output="response")
    case.additional_metadata = metadata
    return case


class TestTTFTScorer:
    def test_within_threshold(self):
        config = L4SystemConfig(scorer_name="ttft", ttft_threshold_ms=500.0)
        scorer = TTFTScorer(config)
        case = _make_case_with_metadata({"ttft_ms": 200.0})
        score = scorer.measure(case)
        assert score == 1.0
        assert scorer.is_successful()

    def test_exceeds_threshold(self):
        config = L4SystemConfig(scorer_name="ttft", ttft_threshold_ms=100.0)
        scorer = TTFTScorer(config)
        case = _make_case_with_metadata({"ttft_ms": 150.0})
        score = scorer.measure(case)
        assert 0.0 < score < 1.0

    def test_no_data(self):
        config = L4SystemConfig(scorer_name="ttft")
        scorer = TTFTScorer(config)
        case = LLMTestCase(input="test", actual_output="response")
        score = scorer.measure(case)
        assert score == 1.0
        assert "skipping" in scorer.reason.lower()

    def test_double_threshold_is_zero(self):
        config = L4SystemConfig(scorer_name="ttft", ttft_threshold_ms=100.0)
        scorer = TTFTScorer(config)
        case = _make_case_with_metadata({"ttft_ms": 200.0})
        score = scorer.measure(case)
        assert score == 0.0


class TestLatencyP99Scorer:
    def test_within_threshold(self):
        config = L4SystemConfig(scorer_name="latency_p99", p99_threshold_ms=5000.0)
        scorer = LatencyP99Scorer(config)
        case = _make_case_with_metadata({"total_latency_ms": 1000.0})
        score = scorer.measure(case)
        assert score == 1.0

    def test_no_data(self):
        config = L4SystemConfig(scorer_name="latency_p99")
        scorer = LatencyP99Scorer(config)
        case = LLMTestCase(input="test", actual_output="response")
        score = scorer.measure(case)
        assert score == 1.0


class TestThroughputScorer:
    def test_high_throughput(self):
        config = L4SystemConfig(scorer_name="throughput", throughput_min_tps=10.0)
        scorer = ThroughputScorer(config)
        # 100 tokens in 1000ms = 100 tok/s
        case = _make_case_with_metadata({
            "tokens_generated": 100,
            "total_latency_ms": 1000.0,
        })
        score = scorer.measure(case)
        assert score == 1.0

    def test_low_throughput(self):
        config = L4SystemConfig(scorer_name="throughput", throughput_min_tps=100.0)
        scorer = ThroughputScorer(config)
        # 10 tokens in 1000ms = 10 tok/s < 100 min
        case = _make_case_with_metadata({
            "tokens_generated": 10,
            "total_latency_ms": 1000.0,
        })
        score = scorer.measure(case)
        assert score == pytest.approx(0.1, abs=0.01)

    def test_no_data(self):
        config = L4SystemConfig(scorer_name="throughput")
        scorer = ThroughputScorer(config)
        case = LLMTestCase(input="test", actual_output="response")
        score = scorer.measure(case)
        assert score == 1.0
