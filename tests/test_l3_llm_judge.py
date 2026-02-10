"""Tests for L3 LLM-Judge scorers (structure only — requires LLM API for full test)."""

import os

import pytest

from aero_eval.models import L3JudgeConfig
from aero_eval.scorers.l3_llm_judge import GEvalScorer, CustomJudgeScorer

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


class TestL3ScorerInitialization:
    """Test that L3 scorers can be instantiated (requires OpenAI API key)."""

    @requires_openai
    def test_geval_with_criteria(self):
        config = L3JudgeConfig(
            scorer_name="geval",
            criteria="Is the answer factually correct?",
        )
        scorer = GEvalScorer(config)
        assert scorer.__name__ == "L3::GEval"
        assert scorer.threshold == 0.5

    @requires_openai
    def test_geval_with_steps(self):
        config = L3JudgeConfig(
            scorer_name="geval",
            evaluation_steps=["Check factual accuracy", "Check tone"],
        )
        scorer = GEvalScorer(config)
        assert scorer.__name__ == "L3::GEval"

    @requires_openai
    def test_custom_judge(self):
        config = L3JudgeConfig(
            scorer_name="custom_judge",
            criteria="Evaluate reasoning clarity",
        )
        scorer = CustomJudgeScorer(config)
        assert scorer.__name__ == "L3::CustomJudge"


class TestL3ScorerRegistry:
    def test_registered(self):
        from aero_eval.registry import ScorerRegistry

        names = [n for n, _ in ScorerRegistry.list_all()]
        assert "geval" in names
        assert "custom_judge" in names
