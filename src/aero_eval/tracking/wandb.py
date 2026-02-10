"""Weights & Biases experiment tracking integration."""

from __future__ import annotations

from typing import Any

from aero_eval.models import EvalRunSummary


class WandbTracker:
    """Tracks evaluation runs in Weights & Biases."""

    def __init__(self, project: str, tags: list[str] | None = None):
        import wandb

        self._wandb = wandb
        self._run = None
        self._project = project
        self._tags = tags or []

    def start(self, config: dict[str, Any] | None = None) -> None:
        """Initialize a W&B run."""
        self._run = self._wandb.init(
            project=self._project,
            config=config,
            tags=self._tags,
        )

    def log_summary(self, summary: EvalRunSummary) -> None:
        """Log evaluation summary as W&B metrics and tables."""
        if not self._run:
            return

        # Scalar metrics
        pass_rate = (
            summary.passed_cases / summary.total_cases
            if summary.total_cases
            else 0
        )
        self._run.log(
            {
                "total_cases": summary.total_cases,
                "passed_cases": summary.passed_cases,
                "failed_cases": summary.failed_cases,
                "pass_rate": pass_rate,
                "duration_seconds": summary.duration_seconds,
            }
        )

        # Per-scorer metrics
        for scorer_name, stats in summary.scorer_summaries.items():
            for stat_name, value in stats.items():
                self._run.log({f"scorers/{scorer_name}/{stat_name}": value})

        # Results table
        columns = [
            "case_id",
            "input",
            "actual_output",
            "aggregate_score",
            "passed",
        ]
        table = self._wandb.Table(columns=columns)
        for r in summary.results:
            table.add_data(
                r.test_case_id,
                r.input[:200],
                (r.actual_output or "")[:200],
                r.aggregate_score,
                r.passed,
            )
        self._run.log({"results_table": table})

        # Scorer breakdown table
        scorer_columns = [
            "case_id",
            "scorer",
            "tier",
            "score",
            "passed",
            "reason",
        ]
        scorer_table = self._wandb.Table(columns=scorer_columns)
        for r in summary.results:
            for sr in r.scorer_results:
                scorer_table.add_data(
                    r.test_case_id,
                    sr.scorer_name,
                    sr.tier.value,
                    sr.score,
                    sr.passed,
                    (sr.reason or "")[:200],
                )
        self._run.log({"scorer_details": scorer_table})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log arbitrary metrics."""
        if self._run:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        """End the W&B run."""
        if self._run:
            self._run.finish()
            self._run = None
