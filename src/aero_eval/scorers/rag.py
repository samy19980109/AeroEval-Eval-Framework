"""RAG Triple-Check Scorer.

Wraps DeepEval's Faithfulness, Answer Relevancy, and Contextual Precision
metrics into a single composite scorer.
"""

from __future__ import annotations

import asyncio

from deepeval.test_case import LLMTestCase

from aero_eval.models import RAGScorerConfig, ScorerTier
from aero_eval.registry import ScorerRegistry
from aero_eval.scorers.base import BaseScorer


@ScorerRegistry.register("rag_triple_check", ScorerTier.RAG)
class RAGTripleCheckScorer(BaseScorer):
    """Triple-check RAG scorer: faithfulness + answer relevancy + context precision.

    The final score is the minimum of the three sub-scores (conservative approach).
    """

    tier = ScorerTier.RAG

    def __init__(self, config: RAGScorerConfig):
        super().__init__(config)
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            FaithfulnessMetric,
        )

        self._faithfulness = FaithfulnessMetric(
            threshold=config.faithfulness_threshold,
            model=config.evaluation_model,
        )
        self._relevancy = AnswerRelevancyMetric(
            threshold=config.relevancy_threshold,
            model=config.evaluation_model,
        )
        self._precision = ContextualPrecisionMetric(
            threshold=config.precision_threshold,
            model=config.evaluation_model,
        )

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self._faithfulness.measure(test_case)
        self._relevancy.measure(test_case)
        self._precision.measure(test_case)

        scores = {
            "faithfulness": self._faithfulness.score or 0.0,
            "relevancy": self._relevancy.score or 0.0,
            "precision": self._precision.score or 0.0,
        }

        self.score = min(scores.values())
        self.reason = (
            f"Faithfulness={scores['faithfulness']:.3f}, "
            f"Relevancy={scores['relevancy']:.3f}, "
            f"Precision={scores['precision']:.3f}"
        )
        self.score_breakdown = scores
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        await asyncio.gather(
            self._faithfulness.a_measure(test_case),
            self._relevancy.a_measure(test_case),
            self._precision.a_measure(test_case),
        )

        scores = {
            "faithfulness": self._faithfulness.score or 0.0,
            "relevancy": self._relevancy.score or 0.0,
            "precision": self._precision.score or 0.0,
        }

        self.score = min(scores.values())
        self.reason = (
            f"Faithfulness={scores['faithfulness']:.3f}, "
            f"Relevancy={scores['relevancy']:.3f}, "
            f"Precision={scores['precision']:.3f}"
        )
        self.score_breakdown = scores
        self.is_successful()
        return self.score

    @property
    def __name__(self) -> str:
        return "RAG::TripleCheck"


@ScorerRegistry.register("faithfulness", ScorerTier.RAG)
class FaithfulnessScorer(BaseScorer):
    """Standalone faithfulness scorer."""

    tier = ScorerTier.RAG

    def __init__(self, config: RAGScorerConfig):
        super().__init__(config)
        from deepeval.metrics import FaithfulnessMetric

        self._metric = FaithfulnessMetric(
            threshold=config.faithfulness_threshold,
            model=config.evaluation_model,
        )

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self._metric.measure(test_case)
        self.score = self._metric.score or 0.0
        self.reason = self._metric.reason
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        await self._metric.a_measure(test_case)
        self.score = self._metric.score or 0.0
        self.reason = self._metric.reason
        self.is_successful()
        return self.score

    @property
    def __name__(self) -> str:
        return "RAG::Faithfulness"


@ScorerRegistry.register("answer_relevancy", ScorerTier.RAG)
class AnswerRelevancyScorer(BaseScorer):
    """Standalone answer relevancy scorer."""

    tier = ScorerTier.RAG

    def __init__(self, config: RAGScorerConfig):
        super().__init__(config)
        from deepeval.metrics import AnswerRelevancyMetric

        self._metric = AnswerRelevancyMetric(
            threshold=config.relevancy_threshold,
            model=config.evaluation_model,
        )

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self._metric.measure(test_case)
        self.score = self._metric.score or 0.0
        self.reason = self._metric.reason
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        await self._metric.a_measure(test_case)
        self.score = self._metric.score or 0.0
        self.reason = self._metric.reason
        self.is_successful()
        return self.score

    @property
    def __name__(self) -> str:
        return "RAG::AnswerRelevancy"
