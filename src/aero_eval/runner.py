"""Evaluation runner that orchestrates the full eval pipeline."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from statistics import mean

from deepeval.test_case import LLMTestCase

from aero_eval.data.factory import DataFactory
from aero_eval.models import (
    EvalConfig,
    EvalResult,
    EvalRunSummary,
    ScorerResult,
    ScorerTier,
    TestCaseData,
)
from aero_eval.registry import ScorerRegistry
from aero_eval.scorers.base import BaseScorer


class EvalRunner:
    """Orchestrates a complete evaluation run."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self._scorers: list[BaseScorer] = []

    def _build_scorers(self) -> list[BaseScorer]:
        """Instantiate scorer objects from config."""
        scorers: list[BaseScorer] = []
        for scorer_config in self.config.scorers:
            scorer_cls = ScorerRegistry.get(scorer_config.scorer_name)
            scorer = scorer_cls(config=scorer_config)
            scorers.append(scorer)
        return scorers

    async def run(self) -> EvalRunSummary:
        """Execute the full evaluation pipeline."""
        start_time = datetime.now(UTC)
        run_id = str(uuid.uuid4())[:8]

        # 1. Load data
        test_cases = DataFactory.load(self.config.data_source)
        deepeval_cases = DataFactory.to_deepeval(test_cases)

        # 2. Build scorers
        self._scorers = self._build_scorers()

        # 3. Run evaluation
        eval_results = self._run_evaluation(deepeval_cases, test_cases)

        # 4. Build summary
        end_time = datetime.now(UTC)
        summary = self._build_summary(run_id, eval_results, start_time, end_time)

        # 5. W&B logging
        if self.config.wandb_project:
            self._log_to_wandb(summary)

        return summary

    def _run_evaluation(
        self,
        deepeval_cases: list[LLMTestCase],
        original_cases: list[TestCaseData],
    ) -> list[EvalResult]:
        """Run all scorers against all test cases."""
        results: list[EvalResult] = []

        for i, (de_case, orig_case) in enumerate(
            zip(deepeval_cases, original_cases)
        ):
            scorer_results: list[ScorerResult] = []

            for scorer in self._scorers:
                t0 = time.perf_counter()
                try:
                    scorer.measure(de_case)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    scorer_results.append(
                        ScorerResult(
                            scorer_name=scorer.__name__,
                            tier=ScorerTier(scorer.config.tier),
                            score=scorer.score,
                            passed=scorer.is_successful(),
                            reason=scorer.reason,
                            latency_ms=elapsed_ms,
                            metadata=scorer.score_breakdown or {},
                        )
                    )
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    scorer_results.append(
                        ScorerResult(
                            scorer_name=scorer.__name__,
                            tier=ScorerTier(scorer.config.tier),
                            score=0.0,
                            passed=False,
                            reason=f"Error: {str(e)[:200]}",
                            latency_ms=elapsed_ms,
                        )
                    )

                    if self.config.fail_fast:
                        break

            aggregate = (
                mean(r.score for r in scorer_results) if scorer_results else 0.0
            )
            all_passed = all(r.passed for r in scorer_results)

            results.append(
                EvalResult(
                    test_case_id=orig_case.id or f"case-{i}",
                    input=orig_case.input,
                    actual_output=orig_case.actual_output,
                    scorer_results=scorer_results,
                    aggregate_score=aggregate,
                    passed=all_passed,
                )
            )

        return results

    def _build_summary(
        self,
        run_id: str,
        results: list[EvalResult],
        start_time: datetime,
        end_time: datetime,
    ) -> EvalRunSummary:
        """Aggregate individual results into a run summary."""
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        # Per-scorer aggregations
        scorer_scores: dict[str, list[float]] = {}
        for result in results:
            for sr in result.scorer_results:
                scorer_scores.setdefault(sr.scorer_name, []).append(sr.score)

        scorer_summaries: dict[str, dict[str, float]] = {}
        for name, scores in scorer_scores.items():
            scorer_summaries[name] = {
                "mean": mean(scores) if scores else 0.0,
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
                "count": float(len(scores)),
            }

        duration = (end_time - start_time).total_seconds()

        return EvalRunSummary(
            run_id=run_id,
            config_name=self.config.name,
            total_cases=len(results),
            passed_cases=passed,
            failed_cases=failed,
            results=results,
            scorer_summaries=scorer_summaries,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )

    def _log_to_wandb(self, summary: EvalRunSummary) -> None:
        """Log results to Weights & Biases."""
        from aero_eval.tracking.wandb import WandbTracker

        tracker = WandbTracker(
            project=self.config.wandb_project,
            tags=self.config.wandb_tags,
        )
        tracker.start(config=self.config.model_dump())
        tracker.log_summary(summary)
        tracker.finish()
