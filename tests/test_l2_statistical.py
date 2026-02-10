"""Tests for L2 statistical scorers."""

import pytest
from deepeval.test_case import LLMTestCase

from aero_eval.models import L2StatConfig
from aero_eval.scorers.l2_statistical import (
    BERTScoreScorer,
    CosineSimilarityScorer,
    ROUGEScorer,
)


def _make_case(actual: str, expected: str) -> LLMTestCase:
    return LLMTestCase(input="test", actual_output=actual, expected_output=expected)


class TestROUGEScorer:
    def test_identical_strings(self):
        config = L2StatConfig(scorer_name="rouge", threshold=0.5)
        scorer = ROUGEScorer(config)
        score = scorer.measure(_make_case("hello world", "hello world"))
        assert score == pytest.approx(1.0, abs=0.01)

    def test_different_strings(self):
        config = L2StatConfig(scorer_name="rouge", threshold=0.5)
        scorer = ROUGEScorer(config)
        score = scorer.measure(
            _make_case("the quick brown fox", "completely different text here")
        )
        assert 0.0 <= score <= 1.0

    def test_partial_overlap(self):
        config = L2StatConfig(scorer_name="rouge", threshold=0.3)
        scorer = ROUGEScorer(config)
        score = scorer.measure(
            _make_case("The capital is Paris", "Paris is the capital of France")
        )
        assert score > 0.0

    def test_score_breakdown(self):
        config = L2StatConfig(
            scorer_name="rouge",
            rouge_types=["rouge1", "rouge2", "rougeL"],
            threshold=0.3,
        )
        scorer = ROUGEScorer(config)
        scorer.measure(_make_case("hello world test", "hello world test"))
        assert "rouge1" in scorer.score_breakdown
        assert "rouge2" in scorer.score_breakdown


@pytest.mark.slow
class TestBERTScoreScorer:
    def test_similar_sentences(self):
        config = L2StatConfig(
            scorer_name="bertscore",
            bertscore_model="roberta-large",
            threshold=0.5,
        )
        scorer = BERTScoreScorer(config)
        score = scorer.measure(
            _make_case(
                "Paris is the capital of France",
                "The capital of France is Paris",
            )
        )
        assert score > 0.5
        assert 0.0 <= score <= 1.0

    def test_score_breakdown(self):
        config = L2StatConfig(
            scorer_name="bertscore",
            bertscore_model="roberta-large",
            threshold=0.5,
        )
        scorer = BERTScoreScorer(config)
        scorer.measure(
            _make_case("Hello world", "Hello world")
        )
        assert "precision" in scorer.score_breakdown
        assert "recall" in scorer.score_breakdown
        assert "f1" in scorer.score_breakdown


@pytest.mark.slow
class TestCosineSimilarityScorer:
    def test_identical_sentences(self):
        config = L2StatConfig(scorer_name="cosine", threshold=0.5)
        scorer = CosineSimilarityScorer(config)
        score = scorer.measure(
            _make_case("Hello world", "Hello world")
        )
        assert score > 0.9

    def test_similar_sentences(self):
        config = L2StatConfig(scorer_name="cosine", threshold=0.5)
        scorer = CosineSimilarityScorer(config)
        score = scorer.measure(
            _make_case(
                "The cat sat on the mat",
                "A cat was sitting on a mat",
            )
        )
        assert score > 0.5

    def test_dissimilar_sentences(self):
        config = L2StatConfig(scorer_name="cosine", threshold=0.5)
        scorer = CosineSimilarityScorer(config)
        score = scorer.measure(
            _make_case(
                "Advanced quantum computing algorithms",
                "How to bake chocolate chip cookies",
            )
        )
        assert score < 0.5
