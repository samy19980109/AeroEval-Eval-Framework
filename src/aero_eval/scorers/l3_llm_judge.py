"""L3 Scorers: LLM-as-judge evaluation via DeepEval's GEval."""

from __future__ import annotations

from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from aero_eval.models import L3JudgeConfig, ScorerTier
from aero_eval.registry import ScorerRegistry
from aero_eval.scorers.base import BaseScorer


@ScorerRegistry.register("geval", ScorerTier.L3)
class GEvalScorer(BaseScorer):
    """Wraps DeepEval's GEval metric for LLM-based evaluation."""

    tier = ScorerTier.L3

    def __init__(self, config: L3JudgeConfig):
        super().__init__(config)
        from deepeval.metrics import GEval

        eval_params = [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ]

        if config.evaluation_steps:
            self._geval = GEval(
                name=config.scorer_name,
                evaluation_steps=config.evaluation_steps,
                evaluation_params=eval_params,
                model=config.evaluation_model,
                threshold=config.threshold,
            )
        else:
            self._geval = GEval(
                name=config.scorer_name,
                criteria=config.criteria,
                evaluation_params=eval_params,
                model=config.evaluation_model,
                threshold=config.threshold,
            )

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self._geval.measure(test_case)
        self.score = self._geval.score
        self.reason = self._geval.reason
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        await self._geval.a_measure(test_case)
        self.score = self._geval.score
        self.reason = self._geval.reason
        self.is_successful()
        return self.score

    @property
    def __name__(self) -> str:
        return "L3::GEval"


@ScorerRegistry.register("custom_judge", ScorerTier.L3)
class CustomJudgeScorer(BaseScorer):
    """Custom LLM judge using evaluation steps for structured grading."""

    tier = ScorerTier.L3

    def __init__(self, config: L3JudgeConfig):
        super().__init__(config)
        from deepeval.metrics import GEval

        eval_params = [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ]

        if config.criteria:
            eval_params.append(LLMTestCaseParams.EXPECTED_OUTPUT)

        self._geval = GEval(
            name=f"custom_{config.scorer_name}",
            criteria=config.criteria,
            evaluation_steps=config.evaluation_steps or None,
            evaluation_params=eval_params,
            model=config.evaluation_model,
            threshold=config.threshold,
        )

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self._geval.measure(test_case)
        self.score = self._geval.score
        self.reason = self._geval.reason
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        await self._geval.a_measure(test_case)
        self.score = self._geval.score
        self.reason = self._geval.reason
        self.is_successful()
        return self.score

    @property
    def __name__(self) -> str:
        return "L3::CustomJudge"
