"""CLI entry point for Aero-Eval."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="aero-eval",
    help="Aero-Eval: High-performance LLM evaluation framework",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    config: Path = typer.Argument(
        ..., help="Path to YAML config file", exists=True, readable=True
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to write JSON results"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run an evaluation suite from a YAML config."""
    from aero_eval.config import load_config
    from aero_eval.runner import EvalRunner

    try:
        eval_config = load_config(config)
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Running eval: {eval_config.name}[/bold]")
    console.print(f"  Scorers: {len(eval_config.scorers)}")
    console.print(f"  Data: {eval_config.data_source.source_type.value}")

    runner = EvalRunner(eval_config)
    summary = asyncio.run(runner.run())

    _print_summary(summary, verbose)

    if output:
        output.write_text(summary.model_dump_json(indent=2))
        console.print(f"\n[green]Results written to {output}[/green]")

    if summary.failed_cases > 0:
        raise typer.Exit(code=1)


@app.command()
def validate(
    config: Path = typer.Argument(
        ..., help="Path to YAML config file", exists=True, readable=True
    ),
):
    """Validate an evaluation config file."""
    from aero_eval.config import load_config, validate_config

    try:
        eval_config = load_config(config)
        warnings = validate_config(config)

        console.print(f"[green]Config '{eval_config.name}' is valid.[/green]")
        console.print(f"  Description: {eval_config.description or '(none)'}")
        console.print(f"  Scorers: {len(eval_config.scorers)}")
        console.print(
            f"  Data source: {eval_config.data_source.source_type.value}"
        )

        for w in warnings:
            console.print(f"  [yellow]Warning: {w}[/yellow]")

    except Exception as e:
        console.print(f"[red]Config validation failed: {e}[/red]")
        raise typer.Exit(code=1)


@app.command(name="list-scorers")
def list_scorers(
    tier: Optional[str] = typer.Option(
        None, "--tier", "-t", help="Filter by tier (L1/L2/L3/L4/RAG)"
    ),
):
    """List all available scorers."""
    # Ensure all scorers are registered
    import aero_eval.scorers  # noqa: F401
    from aero_eval.models import ScorerTier
    from aero_eval.registry import ScorerRegistry

    tier_filter = ScorerTier(tier) if tier else None
    scorers = ScorerRegistry.list_all(tier=tier_filter)

    table = Table(title="Available Scorers")
    table.add_column("Name", style="cyan")
    table.add_column("Tier", style="green")

    for name, scorer_tier in scorers:
        table.add_row(name, scorer_tier.value)

    console.print(table)


@app.command(name="inspect-data")
def inspect_data(
    path: Path = typer.Argument(
        ..., help="Path to JSONL data file", exists=True, readable=True
    ),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of records"),
):
    """Preview records from a JSONL data file."""
    from aero_eval.data.factory import DataFactory

    cases = DataFactory.from_jsonl(path, limit=limit)

    for i, case in enumerate(cases):
        console.print(f"\n[bold]--- Case {i + 1} ---[/bold]")
        console.print(f"  [cyan]ID:[/cyan] {case.id or '(none)'}")
        console.print(f"  [cyan]Input:[/cyan] {case.input[:120]}")
        if case.expected_output:
            console.print(
                f"  [cyan]Expected:[/cyan] {case.expected_output[:120]}"
            )
        if case.actual_output:
            console.print(
                f"  [cyan]Actual:[/cyan] {case.actual_output[:120]}"
            )
        if case.retrieval_context:
            console.print(
                f"  [cyan]Context chunks:[/cyan] {len(case.retrieval_context)}"
            )

    console.print(f"\n[dim]Showing {len(cases)} record(s)[/dim]")


def _print_summary(summary, verbose: bool) -> None:
    """Print a rich table summarizing the evaluation run."""
    console.print(f"\n[bold]Eval Run: {summary.run_id}[/bold]")
    console.print(f"  Config: {summary.config_name}")
    console.print(f"  Duration: {summary.duration_seconds:.2f}s")

    # Overall stats
    pass_rate = (
        summary.passed_cases / summary.total_cases * 100
        if summary.total_cases
        else 0
    )
    color = "green" if pass_rate == 100 else "yellow" if pass_rate >= 50 else "red"
    console.print(
        f"  Result: [{color}]{summary.passed_cases}/{summary.total_cases} passed "
        f"({pass_rate:.0f}%)[/{color}]"
    )

    # Per-scorer summary table
    if summary.scorer_summaries:
        table = Table(title="Scorer Summary")
        table.add_column("Scorer", style="cyan")
        table.add_column("Mean", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")

        for name, stats in summary.scorer_summaries.items():
            table.add_row(
                name,
                f"{stats['mean']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
            )

        console.print(table)

    # Per-case details (verbose)
    if verbose:
        for result in summary.results:
            status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
            console.print(
                f"\n  {status} {result.test_case_id}: "
                f"score={result.aggregate_score:.3f}"
            )
            for sr in result.scorer_results:
                sr_status = "[green]OK[/green]" if sr.passed else "[red]FAIL[/red]"
                console.print(
                    f"    {sr_status} {sr.scorer_name}: "
                    f"{sr.score:.3f} ({sr.latency_ms:.0f}ms)"
                )
                if sr.reason:
                    console.print(f"      {sr.reason}")
