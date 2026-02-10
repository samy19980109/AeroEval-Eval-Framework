# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install/sync all dependencies
uv sync

# Run the CLI
uv run aero-eval run configs/default.yaml [--output results.json] [--verbose]
uv run aero-eval validate configs/default.yaml
uv run aero-eval list-scorers [--tier L1]
uv run aero-eval inspect-data data/samples/golden_qa.jsonl [--limit 5]

# Run tests
uv run pytest tests/ -v                          # All tests
uv run pytest tests/test_l1_rules.py -v           # Single test module
uv run pytest tests/test_l1_rules.py::TestRegexScorer::test_match -v  # Single test
uv run pytest tests/ -m "not slow" -v             # Skip slow tests (BERTScore/Cosine)

# Add dependencies
uv add <package>              # Runtime dependency
uv add --dev <package>        # Dev dependency
```

## Architecture

**Four-tier LLM evaluation framework** with a plugin-based scorer system. The CLI entry point is `aero_eval.cli:app` (Typer).

### Data Flow

```
YAML config → config.py (load + ${ENV_VAR} resolution) → EvalConfig (Pydantic)
    → EvalRunner.run()
        → DataFactory.load() → list[TestCaseData]
        → DataFactory.to_deepeval() → list[LLMTestCase]  (bridge to DeepEval)
        → ScorerRegistry.get() → instantiate scorers
        → scorer.measure(test_case) for each case × scorer
        → EvalRunSummary (with per-scorer stats)
        → optional W&B logging
```

### Scorer Plugin System

All scorers extend `BaseScorer(deepeval.metrics.BaseMetric)` — this makes them compatible with DeepEval's `evaluate()` pipeline. New scorers are added by:

1. Creating a class in `src/aero_eval/scorers/`
2. Decorating with `@ScorerRegistry.register("name", ScorerTier.XX)`
3. Importing the module in `scorers/__init__.py` to trigger registration

Each scorer must implement `measure()`, `a_measure()`, `is_successful()`, and the `__name__` property. The `measure()` method receives a DeepEval `LLMTestCase` and returns a float score in [0, 1].

### Config Discrimination

`ScorerConfig` is a Pydantic discriminated union on the `tier` field (`L1`/`L2`/`L3`/`L4`/`RAG`). Each tier maps to a specific config class (e.g., `L1RuleConfig`, `L2StatConfig`). This allows a single YAML `scorers:` list to contain mixed tier configs that deserialize to the correct types.

### L4 System Scorers & Numerical Stability

L4 scorers and the numerical stability guard expect metrics in `test_case.additional_metadata` (e.g., `ttft_ms`, `total_latency_ms`, `logprobs`). These are populated by inference providers when running live inference. Without metadata, L4 scorers gracefully skip with score=1.0.

### Key Patterns

- **Providers** (`providers/`) are async (`AsyncOpenAI`, `AsyncAnthropic`, `httpx.AsyncClient`). vLLM reuses the OpenAI provider since vLLM exposes an OpenAI-compatible API.
- **L3/RAG scorers** require `OPENAI_API_KEY` at instantiation time (DeepEval dependency). Tests for these are skipped when the key is absent.
- **L2 BERTScore/Cosine** use `run_in_executor()` in `a_measure()` to avoid blocking the event loop. Models are lazy-loaded.
- The runner uses `asyncio.run()` from the CLI but the core `EvalRunner.run()` is async.
