<p align="center">
  <h1 align="center">Aero-Eval</h1>
  <p align="center">
    <strong>Production-grade LLM evaluation framework with four-tier scoring, numerical stability guards, and CI/CD gating — built for high-performance inference environments.</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &bull;
    <a href="#four-tier-scoring-system">Scoring Tiers</a> &bull;
    <a href="#numerical-stability-guard">Numerical Stability</a> &bull;
    <a href="#inference-providers">Providers</a> &bull;
    <a href="#configuration">Configuration</a>
  </p>
</p>

---

## Why Aero-Eval?

Standard LLM evaluation tools check whether a model's *answers* are good. Aero-Eval goes further — it checks whether the *system producing those answers* is healthy.

- **L1-L3 scorers** validate output quality (format, semantics, reasoning)
- **L4 scorers** validate system performance (TTFT, P99 latency, throughput)
- **Numerical stability guards** detect hardware-level regressions (NaN/Inf, logit drift, FP16 overflow)
- **RAG scorers** validate retrieval-augmented generation pipelines (faithfulness, relevancy, context precision)

This makes Aero-Eval uniquely suited for environments where inference correctness depends on both model quality *and* hardware reliability — such as large-scale deployments on Cerebras, TPUs, and custom silicon.

---

## Quick Start

```bash
# Install (requires Python 3.12+ and uv)
uv sync

# Run an evaluation
aero-eval run configs/default.yaml --verbose

# Output results as JSON
aero-eval run configs/default.yaml --output results.json

# Validate config before running (zero execution, checks structure + deps)
aero-eval validate configs/default.yaml

# List available scorers
aero-eval list-scorers --tier L1

# Preview your dataset
aero-eval inspect-data data/samples/golden_qa.jsonl --limit 5
```

---

## Four-Tier Scoring System

Aero-Eval organizes evaluation into four tiers of increasing complexity. Each tier can run independently or compose into a full-stack evaluation pipeline.

```
L1  Deterministic Rules     ~1ms/case     Format validation, keyword checks, schema compliance
L2  Statistical Similarity  ~50ms/case    BERTScore, ROUGE, cosine similarity (no judge model)
L3  LLM-as-Judge            ~2s/case      GPT-4-powered evaluation of reasoning, tone, safety
L4  System & Hardware       ~1ms/case     TTFT, P99 latency, throughput, numerical stability
RAG Retrieval QA            ~3s/case      Faithfulness, relevancy, context precision
```

| Tier | Scorers | What It Catches |
|------|---------|-----------------|
| **L1** | `regex`, `json_schema`, `keyword`, `length` | Malformed outputs, missing fields, format violations |
| **L2** | `bertscore`, `rouge`, `cosine` | Semantic drift from expected outputs without needing a judge model |
| **L3** | `geval`, `custom_judge` | Reasoning errors, tone problems, safety violations |
| **L4** | `ttft`, `latency_p99`, `throughput` | Performance regressions across inference backends |
| **L4** | `numerical_stability`, `logit_drift` | NaN/Inf in logprobs, FP16 overflow, hardware faults |
| **RAG** | `rag_triple_check`, `faithfulness`, `answer_relevancy` | Hallucinations, irrelevant retrievals, unfaithful answers |

### Tier Composition

A single config can mix scorers from any tier. A test case **passes only if all scorers pass** (conservative gating):

```yaml
scorers:
  - tier: L1                    # Gate: output must be valid JSON
    scorer_name: json_schema
    schema: {"type": "object", "required": ["answer"]}
    threshold: 1.0

  - tier: L2                    # Gate: semantically similar to reference
    scorer_name: bertscore
    threshold: 0.7

  - tier: L3                    # Gate: reasoning must be sound
    scorer_name: geval
    criteria: "Is the answer factually correct and well-reasoned?"
    threshold: 0.6

  - tier: L4                    # Gate: TTFT under 200ms
    scorer_name: ttft
    max_ttft_ms: 200
    threshold: 0.8
```

---

## Numerical Stability Guard

> Designed for environments like Cerebras hardware where numerical overflow is a real production concern.

The `numerical_stability` and `logit_drift` scorers detect hardware-level issues invisible to traditional evaluation metrics:

### How It Works

```
Logprob Sequence → [Split into baseline + test window]

  Baseline (first half)          Test Window (second half)
  ┌──────────────────┐           ┌──────────────────┐
  │  mean=−2.3       │           │  mean=−4.1       │
  │  std=0.8         │           │  std=2.3         │
  └──────────────────┘           └──────────────────┘
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                     NaN/Inf        3-sigma        Mean Shift
                     Detection      Outliers       Detection
                          │             │             │
                          ▼             ▼             ▼
                     −0.5 penalty   −0.2 penalty   flags drift
```

**Three checks per logprob sequence:**

| Check | What It Detects | Scoring |
|-------|-----------------|---------|
| **NaN/Inf detection** | Numerical overflow, division by zero in logit computation | −0.5 penalty per occurrence |
| **3-sigma outlier count** | Logprobs beyond 3 standard deviations from baseline mean | −0.2 penalty when outlier ratio exceeds threshold |
| **Mean shift detection** | Systematic drift in logit distribution between baseline and test window | Flags regression when `\|shift\| / baseline_std > sigma_threshold` |

**`logit_drift` scorer** supports explicit baseline comparison — call `set_baseline(logprobs)` with known-good model output, then score new outputs against it. Drift is detected when `mean_shift > sigma_threshold` or `std_ratio > 2.0`.

### Example Config

```yaml
# configs/cerebras_numerical_guard.yaml
scorers:
  - tier: L4
    scorer_name: numerical_stability
    sigma_threshold: 2.5        # Aggressive (default: 3.0)
    threshold: 0.9

  - tier: L4
    scorer_name: logit_drift
    sigma_threshold: 2.5
    threshold: 0.9

  - tier: L4
    scorer_name: ttft
    max_ttft_ms: 100            # Tight latency budget
    threshold: 0.95
```

---

## RAG Evaluation

The `rag_triple_check` scorer runs three evaluations **in parallel** via `asyncio.gather()` and returns the **minimum score** (conservative):

```
                    ┌─────────────────┐
                    │  rag_triple_    │
                    │  check          │
                    └────────┬────────┘
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
        Faithfulness    Answer         Context
                        Relevancy      Precision
               │             │             │
               ▼             ▼             ▼
          "Does the     "Does the     "Was the
          answer use    answer address retrieved
          only the      the user's    chunk actually
          context?"     prompt?"      relevant?"
               │             │             │
               └─────────────┼─────────────┘
                             ▼
                    score = min(F, R, P)
```

**Score breakdown** is available in results for granular debugging:

```json
{
  "score": 0.78,
  "score_breakdown": {
    "faithfulness": 0.85,
    "relevancy": 0.92,
    "precision": 0.78
  }
}
```

---

## Inference Providers

Four async providers for running live inference during evaluation. All providers capture timing metadata (TTFT, total latency, tokens generated) that feeds into L4 scorers.

| Provider | Backend | Streaming | Logprob Capture | Use Case |
|----------|---------|-----------|-----------------|----------|
| **OpenAI** | `AsyncOpenAI` | Yes | Yes (per-token) | GPT-4o, GPT-4-turbo |
| **Anthropic** | `AsyncAnthropic` | Yes | No | Claude 3.5 Sonnet, Opus |
| **vLLM** | OpenAI-compatible API | Yes | Yes | Self-hosted open models |
| **Ollama** | REST / `httpx.AsyncClient` | Yes | No | Local development |

### Provider Architecture

```python
# All providers share the same async interface
async with OpenAIProvider(config) as provider:
    result = await provider.generate("What is 2+2?")
    # result.text         → "4"
    # result.ttft_ms      → 45.2
    # result.logprobs     → [-0.001, -2.3, ...]
    # result.tokens_generated → 1

# vLLM reuses OpenAI's implementation (21 lines total)
# because vLLM exposes an OpenAI-compatible /v1/chat/completions endpoint
```

Providers populate `test_case.additional_metadata` with timing and logprob data, enabling seamless L4 scoring without manual instrumentation.

---

## Configuration

### Evaluation Config

```yaml
name: "my-evaluation"
description: "Evaluate Q&A quality"

data_source:
  source_type: jsonl               # jsonl | huggingface | dict
  path: "data/samples/golden_qa.jsonl"
  # field_mapping:                 # Optional: remap dataset columns
  #   input: "question"
  #   expected_output: "answer"

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

# Optional: run live inference before scoring
provider:
  provider_type: openai            # openai | anthropic | vllm | ollama
  model_name: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"     # Supports ${ENV_VAR} substitution

wandb_project: "my-evals"         # Optional W&B tracking
```

### Data Sources

**JSONL** — one record per line:
```json
{"id": "qa-001", "input": "What is the capital of France?", "expected_output": "Paris is the capital of France.", "actual_output": "The capital of France is Paris."}
```

**HuggingFace Datasets** — load any dataset with field mapping:
```yaml
data_source:
  source_type: huggingface
  path: "cais/mmlu"
  split: "test"
  field_mapping:
    input: "question"
    expected_output: "answer"
```

**Python dicts** — for programmatic test generation:
```python
data_source = {"source_type": "dict", "records": [{"input": "...", "actual_output": "..."}]}
```

Supported fields: `id` (optional), `input` (required), `actual_output`, `expected_output`, `context`, `retrieval_context`, `metadata`.

### Included Example Configs

| Config | Tiers | Purpose |
|--------|-------|---------|
| `default.yaml` | L1 + L2 | Basic quality check (keyword, length, ROUGE) |
| `cerebras_numerical_guard.yaml` | L4 | Hardware-focused: aggressive numerical stability + tight latency thresholds |
| `full_stack_eval.yaml` | L1-L4 + RAG | All tiers with OpenAI provider and W&B tracking |
| `rag_eval.yaml` | RAG | Faithfulness, relevancy, context precision |
| `llm_judge_detailed.yaml` | L3 | Multi-criteria judge (accuracy, reasoning, tone, safety, conciseness) |
| `code_generation_eval.yaml` | L1-L3 | Code validation with JSON schema, regex, BERTScore, LLM judge |
| `huggingface_mmlu_eval.yaml` | L1-L3 | MMLU benchmark loaded from HuggingFace with field mapping |
| `ollama_local_eval.yaml` | L1-L2 | Fully local evaluation — no API keys required |
| `ci_regression_gate.yaml` | L1 | Fast, deterministic checks for CI/CD pipelines |

---

## CI/CD Integration

Aero-Eval is designed to run as a CI gate. The CLI returns **exit code 1** when any test case fails, making it easy to block deployments on quality regressions.

```yaml
# .github/workflows/eval.yml
- name: Run evaluation gate
  run: |
    uv sync
    aero-eval run configs/ci_regression_gate.yaml --output results.json
    # Exit code 1 if any case fails → blocks the pipeline
```

```yaml
# configs/ci_regression_gate.yaml — fast, deterministic, no API keys
scorers:
  - tier: L1
    scorer_name: length
    min_length: 1
    max_length: 10000
    threshold: 1.0

  - tier: L1
    scorer_name: keyword
    expected_keywords: ["answer"]
    threshold: 0.8
```

---

## Adding a Custom Scorer

Scorers auto-register via the plugin registry. Three steps:

**1. Create a scorer class:**

```python
# src/aero_eval/scorers/my_scorer.py
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

**2. Register it** by importing the module in `src/aero_eval/scorers/__init__.py`.

**3. Use it** in any config:

```yaml
scorers:
  - tier: L1
    scorer_name: my_scorer
    threshold: 0.8
```

---

## Architecture

```
                         ┌─────────────┐
                         │  YAML Config │
                         └──────┬──────┘
                                │
                    config.py   │  ${ENV_VAR} resolution
                    models.py   │  Pydantic v2 validation (discriminated unions)
                                ▼
                         ┌─────────────┐
                         │  EvalRunner  │
                         └──────┬──────┘
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
         ┌────────────┐  ┌───────────┐  ┌──────────────┐
         │ DataFactory │  │  Scorer   │  │   Provider   │
         │             │  │  Registry │  │  (optional)  │
         │ JSONL       │  │           │  │              │
         │ HuggingFace │  │ L1 Rules  │  │ OpenAI       │
         │ Dict        │  │ L2 Stats  │  │ Anthropic    │
         └──────┬──────┘  │ L3 Judge  │  │ vLLM         │
                │         │ L4 System │  │ Ollama       │
                │         │ RAG       │  └──────┬───────┘
                │         └─────┬─────┘         │
                │               │               │
                ▼               ▼               ▼
         ┌─────────────────────────────────────────┐
         │  for each test_case × scorer:           │
         │    score = scorer.measure(test_case)     │
         │    passed = score >= threshold            │
         └────────────────────┬────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
         EvalRunSummary    JSON Output   W&B Tracking
         (Rich console)    (--output)    (optional)
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Pydantic v2 discriminated unions** | A single `scorers:` list in YAML can mix L1/L2/L3/L4/RAG configs — Pydantic deserializes to the correct type based on the `tier` field |
| **DeepEval `BaseMetric` inheritance** | All scorers are DeepEval-compatible, enabling interop with the broader DeepEval ecosystem |
| **Async-first providers** | All inference providers use async clients (`AsyncOpenAI`, `AsyncAnthropic`, `httpx.AsyncClient`) with streaming for TTFT capture |
| **Lazy model loading** | L2 scorers (BERTScore, Cosine) load models on first `measure()` call, reducing startup from ~5s to <100ms |
| **Thread pool for ML inference** | L2 scorers use `run_in_executor()` to run model inference without blocking the async event loop |
| **Plugin registry** | `@ScorerRegistry.register()` decorator enables zero-config scorer registration — add a class, add a decorator, import the module |
| **Conservative pass logic** | A test case passes only if *all* scorers pass — no silent quality regressions |
| **Graceful degradation** | L4 scorers return score=1.0 when metadata is unavailable; GPU monitoring continues with CPU-only if `pynvml` fails |

### Project Structure

```
src/aero_eval/
├── cli.py              # Typer CLI — run, validate, list-scorers, inspect-data
├── config.py           # YAML loading with recursive ${ENV_VAR} resolution
├── models.py           # Pydantic v2 models with discriminated union for scorer configs
├── registry.py         # Plugin-based scorer registry with decorator registration
├── runner.py           # EvalRunner — async orchestrator with per-scorer error isolation
├── data/
│   ├── factory.py      # DataFactory — JSONL, HuggingFace, dict loaders with field mapping
│   └── transforms.py   # Normalize whitespace, truncate outputs, auto-assign IDs
├── providers/
│   ├── base.py         # InferenceResult model + ModelProvider ABC
│   ├── openai.py       # Streaming + per-token logprob capture
│   ├── anthropic.py    # Native async streaming SDK
│   ├── vllm.py         # Inherits OpenAI (21 lines — zero code duplication)
│   └── ollama.py       # Raw HTTP streaming via httpx
├── scorers/
│   ├── base.py         # BaseScorer(BaseMetric) — threshold, score_breakdown, reason
│   ├── l1_rules.py     # regex, json_schema, keyword, length
│   ├── l2_statistical.py # bertscore, rouge, cosine (lazy-loaded models)
│   ├── l3_llm_judge.py # geval, custom_judge (wraps DeepEval GEval)
│   ├── l4_system.py    # ttft, latency_p99, throughput
│   ├── numerical.py    # numerical_stability, logit_drift (3-sigma detection)
│   └── rag.py          # rag_triple_check, faithfulness, answer_relevancy
├── tracking/
│   ├── wandb.py        # W&B logging — scalar metrics + two drill-down tables
│   └── monitor.py      # CPU/memory/GPU telemetry via psutil + pynvml
└── utils/              # Logging and numeric utilities
```

---

## Observability

### Weights & Biases Integration

When `wandb_project` is set in config, Aero-Eval logs:

- **Scalar metrics** — total cases, pass rate, duration
- **Per-scorer aggregates** — mean/min/max for each scorer (e.g., `scorers/L1::Regex/mean`)
- **Results table** — per-case scores with input/output previews
- **Scorer breakdown table** — per-case, per-scorer detail with reasons

### Performance Monitor

Built-in telemetry captures system health during evaluation runs:

- **CPU & memory** utilization via `psutil`
- **GPU utilization & VRAM** via `pynvml` (gracefully skipped if no GPU)
- Point-in-time snapshots with avg/max aggregation

---

## Development

```bash
# Install all dependencies (runtime + dev)
uv sync

# Run all tests
uv run pytest tests/ -v

# Run fast tests only (skip BERTScore/Cosine model loading)
uv run pytest tests/ -m "not slow" -v

# Run a single test
uv run pytest tests/test_l1_rules.py::TestRegexScorer::test_match -v

# Lint
uv run ruff check src/ tests/
```

### Test Suite

12 test modules covering all scorers, data loading, config validation, the runner pipeline, and CLI commands. Tests requiring `OPENAI_API_KEY` (L3, RAG) are automatically skipped when the key is absent. Slow tests (L2 model loading) are marked with `@pytest.mark.slow`.

```bash
# Fast feedback loop — deterministic tests only (<5s)
uv run pytest tests/ -m "not slow" -v

# Full suite including model loading
uv run pytest tests/ -v
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Package Manager | [uv](https://github.com/astral-sh/uv) (Rust-based, 10-100x faster than pip) |
| Data Models | Pydantic v2 |
| CLI | Typer + Rich |
| Eval Framework | DeepEval (BaseMetric integration) |
| NLP Models | BERTScore (`deberta-xlarge-mnli`), Sentence Transformers (`all-MiniLM-L6-v2`) |
| Inference | AsyncOpenAI, AsyncAnthropic, httpx |
| Tracking | Weights & Biases |
| Testing | pytest + pytest-asyncio |
| Linting | Ruff |
