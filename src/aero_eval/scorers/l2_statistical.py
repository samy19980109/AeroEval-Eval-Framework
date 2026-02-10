"""L2 Scorers: Statistical/semantic similarity metrics."""

from __future__ import annotations

import asyncio

from deepeval.test_case import LLMTestCase

from aero_eval.models import L2StatConfig, ScorerTier
from aero_eval.registry import ScorerRegistry
from aero_eval.scorers.base import BaseScorer


@ScorerRegistry.register("bertscore", ScorerTier.L2)
class BERTScoreScorer(BaseScorer):
    """Computes BERTScore F1 between actual_output and expected_output."""

    tier = ScorerTier.L2

    def __init__(self, config: L2StatConfig):
        super().__init__(config)
        self._model_type = config.bertscore_model

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        from bert_score import score as bert_score_fn

        cands = [test_case.actual_output or ""]
        refs = [test_case.expected_output or ""]

        P, R, F1 = bert_score_fn(
            cands,
            refs,
            model_type=self._model_type,
            lang="en",
            verbose=False,
        )
        self.score = F1[0].item()
        self.reason = (
            f"BERTScore P={P[0].item():.3f} "
            f"R={R[0].item():.3f} "
            f"F1={self.score:.3f}"
        )
        self.score_breakdown = {
            "precision": P[0].item(),
            "recall": R[0].item(),
            "f1": self.score,
        }
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.measure, test_case)

    @property
    def __name__(self) -> str:
        return "L2::BERTScore"


@ScorerRegistry.register("rouge", ScorerTier.L2)
class ROUGEScorer(BaseScorer):
    """Computes ROUGE scores between actual and expected output."""

    tier = ScorerTier.L2

    def __init__(self, config: L2StatConfig):
        super().__init__(config)
        from rouge_score import rouge_scorer

        self._scorer = rouge_scorer.RougeScorer(
            config.rouge_types, use_stemmer=True
        )
        self._rouge_types = config.rouge_types

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        actual = test_case.actual_output or ""
        expected = test_case.expected_output or ""
        scores = self._scorer.score(expected, actual)

        f1_scores = [scores[rt].fmeasure for rt in self._rouge_types]
        self.score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        details = ", ".join(
            f"{rt}={scores[rt].fmeasure:.3f}" for rt in self._rouge_types
        )
        self.reason = f"ROUGE: {details}"
        self.score_breakdown = {
            rt: scores[rt].fmeasure for rt in self._rouge_types
        }
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "L2::ROUGE"


@ScorerRegistry.register("cosine", ScorerTier.L2)
class CosineSimilarityScorer(BaseScorer):
    """Computes cosine similarity between sentence embeddings."""

    tier = ScorerTier.L2

    def __init__(self, config: L2StatConfig):
        super().__init__(config)
        self._model_name = config.embedding_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        import numpy as np

        model = self._get_model()
        actual = test_case.actual_output or ""
        expected = test_case.expected_output or ""

        embeddings = model.encode([actual, expected])
        norm_a = np.linalg.norm(embeddings[0])
        norm_b = np.linalg.norm(embeddings[1])
        cosine_sim = float(
            np.dot(embeddings[0], embeddings[1]) / (norm_a * norm_b + 1e-8)
        )

        self.score = max(0.0, cosine_sim)
        self.reason = f"Cosine similarity: {self.score:.3f}"
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.measure, test_case)

    @property
    def __name__(self) -> str:
        return "L2::CosineSimilarity"
