"""Tests for RAG scorer (structure only — requires LLM API for full test)."""

import os

import pytest

from aero_eval.models import RAGScorerConfig
from aero_eval.scorers.rag import RAGTripleCheckScorer, FaithfulnessScorer, AnswerRelevancyScorer

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


class TestRAGScorerInitialization:
    """Test that RAG scorers can be instantiated (requires OpenAI API key for DeepEval)."""

    @requires_openai
    def test_triple_check_init(self):
        config = RAGScorerConfig()
        scorer = RAGTripleCheckScorer(config)
        assert scorer.__name__ == "RAG::TripleCheck"
        assert scorer.threshold == 0.7

    @requires_openai
    def test_faithfulness_init(self):
        config = RAGScorerConfig(scorer_name="faithfulness")
        scorer = FaithfulnessScorer(config)
        assert scorer.__name__ == "RAG::Faithfulness"

    @requires_openai
    def test_answer_relevancy_init(self):
        config = RAGScorerConfig(scorer_name="answer_relevancy")
        scorer = AnswerRelevancyScorer(config)
        assert scorer.__name__ == "RAG::AnswerRelevancy"


class TestRAGScorerRegistry:
    def test_registered(self):
        from aero_eval.registry import ScorerRegistry

        names = [n for n, _ in ScorerRegistry.list_all()]
        assert "rag_triple_check" in names
        assert "faithfulness" in names
        assert "answer_relevancy" in names
