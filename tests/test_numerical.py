"""Tests for numerical stability scorers."""

import math

import numpy as np
import pytest
from deepeval.test_case import LLMTestCase

from aero_eval.models import L4SystemConfig
from aero_eval.scorers.numerical import LogitDriftScorer, NumericalStabilityScorer


def _make_case_with_logprobs(logprobs: list[float]) -> LLMTestCase:
    case = LLMTestCase(input="test", actual_output="response")
    case.additional_metadata = {"logprobs": logprobs}
    return case


class TestNumericalStabilityScorer:
    def test_clean_logprobs(self):
        config = L4SystemConfig(scorer_name="numerical_stability", window_size=10)
        scorer = NumericalStabilityScorer(config)
        # Use a constant value sequence - no drift possible
        rng = np.random.default_rng(42)
        logprobs = list(rng.normal(-2.0, 0.1, 100))
        case = _make_case_with_logprobs(logprobs)
        score = scorer.measure(case)
        assert score == 1.0
        assert "No numerical issues" in scorer.reason

    def test_nan_detection(self):
        config = L4SystemConfig(scorer_name="numerical_stability")
        scorer = NumericalStabilityScorer(config)
        logprobs = [-1.0, -2.0, float("nan"), -1.5, float("nan")]
        case = _make_case_with_logprobs(logprobs)
        score = scorer.measure(case)
        assert score < 1.0
        assert "NaN" in scorer.reason

    def test_inf_detection(self):
        config = L4SystemConfig(scorer_name="numerical_stability")
        scorer = NumericalStabilityScorer(config)
        logprobs = [-1.0, float("inf"), -2.0, float("-inf")]
        case = _make_case_with_logprobs(logprobs)
        score = scorer.measure(case)
        assert score < 1.0
        assert "Inf" in scorer.reason

    def test_logit_drift_detection(self):
        config = L4SystemConfig(
            scorer_name="numerical_stability",
            sigma_threshold=2.0,
            window_size=20,
        )
        scorer = NumericalStabilityScorer(config)
        # Baseline: normal around -2.0
        baseline = list(np.random.normal(-2.0, 0.3, 50))
        # Drift: shifted mean to -5.0
        drifted = list(np.random.normal(-5.0, 0.3, 50))
        logprobs = baseline + drifted
        case = _make_case_with_logprobs(logprobs)
        score = scorer.measure(case)
        assert score < 1.0
        assert "drift" in scorer.reason.lower() or "shift" in scorer.reason.lower()

    def test_no_logprobs(self):
        config = L4SystemConfig(scorer_name="numerical_stability")
        scorer = NumericalStabilityScorer(config)
        case = LLMTestCase(input="test", actual_output="response")
        score = scorer.measure(case)
        assert score == 1.0
        assert "skipping" in scorer.reason.lower()

    def test_score_breakdown(self):
        config = L4SystemConfig(scorer_name="numerical_stability")
        scorer = NumericalStabilityScorer(config)
        logprobs = [-1.0, float("nan"), -2.0]
        case = _make_case_with_logprobs(logprobs)
        scorer.measure(case)
        assert scorer.score_breakdown is not None
        assert "nan_count" in scorer.score_breakdown
        assert scorer.score_breakdown["nan_count"] == 1.0


class TestLogitDriftScorer:
    def test_no_baseline(self):
        config = L4SystemConfig(scorer_name="logit_drift")
        scorer = LogitDriftScorer(config)
        logprobs = [-1.0, -2.0, -1.5]
        case = _make_case_with_logprobs(logprobs)
        score = scorer.measure(case)
        assert score == 1.0
        assert "No baseline" in scorer.reason

    def test_no_drift(self):
        config = L4SystemConfig(scorer_name="logit_drift", sigma_threshold=3.0)
        scorer = LogitDriftScorer(config)
        baseline = list(np.random.normal(-2.0, 0.5, 100))
        scorer.set_baseline(baseline)
        # Current is similar to baseline
        current = list(np.random.normal(-2.0, 0.5, 50))
        case = _make_case_with_logprobs(current)
        score = scorer.measure(case)
        assert score > 0.8
        assert "No drift" in scorer.reason or "drift" not in scorer.reason.lower()

    def test_drift_detected(self):
        config = L4SystemConfig(scorer_name="logit_drift", sigma_threshold=2.0)
        scorer = LogitDriftScorer(config)
        baseline = list(np.random.normal(-2.0, 0.3, 100))
        scorer.set_baseline(baseline)
        # Current has a large mean shift
        current = list(np.random.normal(-10.0, 0.3, 50))
        case = _make_case_with_logprobs(current)
        score = scorer.measure(case)
        assert score < 1.0
        assert "drift" in scorer.reason.lower()

    def test_all_nan_inf(self):
        config = L4SystemConfig(scorer_name="logit_drift")
        scorer = LogitDriftScorer(config)
        logprobs = [float("nan"), float("inf")]
        case = _make_case_with_logprobs(logprobs)
        score = scorer.measure(case)
        assert score == 0.0
