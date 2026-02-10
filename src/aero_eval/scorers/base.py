"""Base scorer class extending DeepEval's BaseMetric."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from aero_eval.models import ScorerTier

if TYPE_CHECKING:
    from aero_eval.models import ScorerConfig


class BaseScorer(BaseMetric):
    """Abstract base scorer that extends DeepEval's BaseMetric.

    By extending BaseMetric, every Aero-Eval scorer plugs directly
    into deepeval.evaluate() and deepeval.assert_test().
    """

    tier: ScorerTier
    config: ScorerConfig

    def __init__(self, config: ScorerConfig, **kwargs):
        self.config = config
        self.threshold = config.threshold
        self.score = 0.0
        self.reason = None
        self.success = None
        self.error = None
        self.score_breakdown = {}
        self.strict_mode = False
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_cost = None
        self.verbose_logs = None
        self.skipped = False

    @abstractmethod
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        ...

    @abstractmethod
    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        ...

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
            return False
        self.success = self.score >= self.threshold
        return self.success

    @property
    @abstractmethod
    def __name__(self) -> str:
        ...
