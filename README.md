# Aero-Eval

High-performance LLM evaluation framework with four-tier scoring, designed for CI/CD integration to prevent model regressions.

## Quick Start

```bash
# Install (requires Python 3.12+ and uv)
uv sync

# Run an evaluation
aero-eval run configs/default.yaml --verbose

# Output results as JSON
aero-eval run configs/default.yaml --output results.json
```

## Four-Tier Scoring System

| Tier | Name | Scorers | Purpose |
|------|------|---------|---------|
| **L1** | Unit Rules | `regex`, `json_schema`, `keyword`, `length` | Fast, deterministic output format validation |
| **L2** | Statistical | `bertscore`, `rouge`, `cosine` | Semantic similarity without a judge model |
| **L3** | LLM-Judge | `geval`, `custom_judge` | LLM-based evaluation of reasoning and tone |
| **L4** | System/Hardware | `ttft`, `latency_p99`, `throughput`, `numerical_stability`, `logit_drift` | Hardware efficiency and numerical health |
| **RAG** | Retrieval QA | `rag_triple_check`, `faithfulness`, `answer_relevancy` | Faithfulness, relevancy, and context precision |

## Configuration

Evaluations are defined in YAML config files. Each config specifies a data source and a list of scorers to run:

```yaml
name: "my-evaluation"
description: "Evaluate Q&A quality"

data_source:
  source_type: jsonl           # jsonl | huggingface | dict
  path: "data/samples/golden_qa.jsonl"

scorers:
  - tier: L1
    scorer_name: length
    min_length: 1
    max_length: 5000
    threshold: 1.0

  - tier: L2
    scorer_name: rouge
    rouge_types: ["rouge1", "rouge2", "rougeL"]
    threshold: 0.3

  - tier: L3
    scorer_name: geval
    criteria: "Is the answer factually correct and well-reasoned?"
    evaluation_model: "gpt-4o"
    threshold: 0.5

# Optional
provider:
  provider_type: openai        # openai | anthropic | vllm | ollama
  model_name: "gpt-4o"
  api_key: "${OPENAI_API_KEY}" # Supports env var substitution

wandb_project: "my-evals"     # Optional W&B tracking
```

### Data Format

JSONL files with one record per line:

```json
{"id": "qa-001", "input": "What is the capital of France?", "expected_output": "Paris is the capital of France.", "actual_output": "The capital of France is Paris."}
```

Fields: `id` (optional), `input` (required), `actual_output`, `expected_output`, `context`, `retrieval_context`, `metadata`.

## CLI Commands

```bash
# Run evaluation
aero-eval run <config.yaml> [--output results.json] [--verbose]

# Validate a config file
aero-eval validate <config.yaml>

# List all available scorers
aero-eval list-scorers [--tier L1|L2|L3|L4|RAG]

# Preview a dataset
aero-eval inspect-data <file.jsonl> [--limit 5]
```

## Numerical Stability Guard

The `numerical_stability` and `logit_drift` scorers detect hardware-level issues in large-scale inference:

- **NaN/Inf detection** in logprob sequences
- **3-sigma logit drift** detection using rolling window comparison against a baseline
- Flags "Numerical Regression" when logit standard deviation shifts beyond the configured threshold

These are designed for environments like Cerebras hardware where numerical overflow is a concern.

## RAG Evaluation

The `rag_triple_check` scorer runs three checks in parallel:

1. **Faithfulness** -- Does the answer only use the provided context?
2. **Answer Relevancy** -- Does the answer actually address the user's prompt?
3. **Context Precision** -- Was the retrieved chunk actually relevant?

The final score is the minimum of the three (conservative). Requires `OPENAI_API_KEY` for the LLM judge.

## Inference Providers

Async providers for running live inference during evaluation:

| Provider | Backend | Logprob Support |
|----------|---------|-----------------|
| OpenAI | `AsyncOpenAI` with streaming | Yes |
| Anthropic | `AsyncAnthropic` with streaming | No |
| vLLM | OpenAI-compatible API | Yes |
| Ollama | REST API (`localhost:11434`) | No |

## Adding a Custom Scorer

```python
from deepeval.test_case import LLMTestCase
from aero_eval.models import L1RuleConfig, ScorerTier
from aero_eval.registry import ScorerRegistry
from aero_eval.scorers.base import BaseScorer

@ScorerRegistry.register("my_scorer", ScorerTier.L1)
class MyScorer(BaseScorer):
    tier = ScorerTier.L1

    def __init__(self, config: L1RuleConfig):
        super().__init__(config)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        self.score = 1.0 if "expected" in text else 0.0
        self.reason = "Custom check"
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "L1::MyScorer"
```

Then import the module in `src/aero_eval/scorers/__init__.py` to register it.

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Run fast tests only (skip BERTScore/Cosine model loading)
uv run pytest tests/ -m "not slow" -v

# Lint
uv run ruff check src/ tests/
```

## Project Structure

```
src/aero_eval/
├── cli.py              # Typer CLI entry point
├── config.py           # YAML config loading with ${ENV_VAR} resolution
├── models.py           # Pydantic v2 data models (discriminated unions)
├── registry.py         # Scorer plugin registry
├── runner.py           # EvalRunner orchestrator
├── data/               # DataFactory (JSONL, HuggingFace, dict loaders)
├── providers/          # Async inference providers (OpenAI, Anthropic, vLLM, Ollama)
├── scorers/            # L1-L4, RAG, and numerical stability scorers
├── tracking/           # W&B integration and performance monitor
└── utils/              # Logging and numeric utilities
```
