"""Tests for the EvalRunner."""

import asyncio

import pytest

from aero_eval.runner import EvalRunner


class TestEvalRunner:
    def test_run_basic(self, minimal_eval_config):
        runner = EvalRunner(minimal_eval_config)
        summary = asyncio.run(runner.run())

        assert summary.config_name == "test-eval"
        assert summary.total_cases == 2
        assert summary.passed_cases == 2
        assert summary.failed_cases == 0
        assert summary.duration_seconds >= 0

    def test_scorer_results_populated(self, minimal_eval_config):
        runner = EvalRunner(minimal_eval_config)
        summary = asyncio.run(runner.run())

        for result in summary.results:
            assert len(result.scorer_results) == 1
            sr = result.scorer_results[0]
            assert sr.scorer_name == "L1::Length"
            assert sr.score == 1.0
            assert sr.passed is True
            assert sr.latency_ms >= 0

    def test_scorer_summaries(self, minimal_eval_config):
        runner = EvalRunner(minimal_eval_config)
        summary = asyncio.run(runner.run())

        assert "L1::Length" in summary.scorer_summaries
        stats = summary.scorer_summaries["L1::Length"]
        assert stats["mean"] == 1.0
        assert stats["count"] == 2.0

    def test_failed_cases(self, golden_qa_path):
        from aero_eval.models import (
            DataSourceConfig,
            DataSourceType,
            EvalConfig,
            L1RuleConfig,
        )

        config = EvalConfig(
            name="strict-eval",
            data_source=DataSourceConfig(
                source_type=DataSourceType.JSONL,
                path=str(golden_qa_path),
            ),
            scorers=[
                L1RuleConfig(
                    scorer_name="length",
                    max_length=5,  # Very short, should fail
                    threshold=1.0,
                ),
            ],
        )
        runner = EvalRunner(config)
        summary = asyncio.run(runner.run())

        assert summary.failed_cases == 2
        assert summary.passed_cases == 0
