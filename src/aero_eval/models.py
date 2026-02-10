"""Core Pydantic v2 data models for the Aero-Eval framework."""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


# --- Enums ---


class ProviderType(str, enum.Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VLLM = "vllm"
    OLLAMA = "ollama"


class ScorerTier(str, enum.Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    RAG = "RAG"


class DataSourceType(str, enum.Enum):
    JSONL = "jsonl"
    HUGGINGFACE = "huggingface"
    DICT = "dict"


# --- Provider Config ---


class ProviderConfig(BaseModel):
    provider_type: ProviderType
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    enable_logprobs: bool = False
    top_logprobs: int = Field(default=5, ge=1, le=20)
    timeout_seconds: float = 60.0


# --- Scorer Configs ---


class L1RuleConfig(BaseModel):
    tier: Literal["L1"] = "L1"
    scorer_name: str
    pattern: str | None = None
    expected_keywords: list[str] = []
    forbidden_keywords: list[str] = []
    min_length: int | None = None
    max_length: int | None = None
    json_schema: dict[str, Any] | None = None
    case_sensitive: bool = True
    threshold: float = Field(default=1.0, ge=0.0, le=1.0)


class L2StatConfig(BaseModel):
    tier: Literal["L2"] = "L2"
    scorer_name: str
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    rouge_types: list[str] = ["rouge1", "rouge2", "rougeL"]
    embedding_model: str = "all-MiniLM-L6-v2"
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class L3JudgeConfig(BaseModel):
    tier: Literal["L3"] = "L3"
    scorer_name: str
    evaluation_model: str = "gpt-4o"
    criteria: str | None = None
    evaluation_steps: list[str] = []
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    include_reason: bool = True

    @model_validator(mode="after")
    def criteria_or_steps(self) -> L3JudgeConfig:
        if not self.criteria and not self.evaluation_steps:
            raise ValueError("Must provide either 'criteria' or 'evaluation_steps'")
        return self


class L4SystemConfig(BaseModel):
    tier: Literal["L4"] = "L4"
    scorer_name: str
    ttft_threshold_ms: float = 500.0
    p99_threshold_ms: float = 5000.0
    throughput_min_tps: float = 10.0
    sigma_threshold: float = 3.0
    window_size: int = Field(default=50, gt=0)
    threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class NumericalGuardConfig(BaseModel):
    enabled: bool = True
    sigma_threshold: float = Field(default=3.0, gt=0.0)
    enable_nan_check: bool = True
    enable_inf_check: bool = True
    window_size: int = Field(default=50, gt=0)


class RAGScorerConfig(BaseModel):
    tier: Literal["RAG"] = "RAG"
    scorer_name: str = "rag_triple_check"
    evaluation_model: str = "gpt-4o"
    faithfulness_threshold: float = 0.7
    relevancy_threshold: float = 0.7
    precision_threshold: float = 0.7
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


ScorerConfig = Annotated[
    L1RuleConfig | L2StatConfig | L3JudgeConfig | L4SystemConfig | RAGScorerConfig,
    Field(discriminator="tier"),
]


# --- Data Source Config ---


class DataSourceConfig(BaseModel):
    source_type: DataSourceType
    path: str | None = None
    dataset_name: str | None = None
    split: str = "test"
    field_mapping: dict[str, str] = {}
    limit: int | None = None


# --- Test Case & Results ---


class TestCaseData(BaseModel):
    id: str | None = None
    input: str
    actual_output: str | None = None
    expected_output: str | None = None
    context: list[str] | None = None
    retrieval_context: list[str] | None = None
    metadata: dict[str, Any] = {}


class ScorerResult(BaseModel):
    scorer_name: str
    tier: ScorerTier
    score: float
    passed: bool
    reason: str | None = None
    metadata: dict[str, Any] = {}
    latency_ms: float = 0.0


class EvalResult(BaseModel):
    test_case_id: str
    input: str
    actual_output: str | None = None
    scorer_results: list[ScorerResult] = []
    aggregate_score: float = 0.0
    passed: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provider_metadata: dict[str, Any] = {}


class EvalRunSummary(BaseModel):
    run_id: str
    config_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    results: list[EvalResult]
    scorer_summaries: dict[str, dict[str, float]] = {}
    start_time: datetime
    end_time: datetime
    duration_seconds: float


# --- Top-level Eval Config ---


class EvalConfig(BaseModel):
    name: str
    description: str = ""
    provider: ProviderConfig | None = None
    scorers: list[ScorerConfig] = Field(..., min_length=1)
    data_source: DataSourceConfig
    numerical_guard: NumericalGuardConfig = NumericalGuardConfig()
    wandb_project: str | None = None
    wandb_tags: list[str] = []
    concurrency: int = Field(default=5, gt=0)
    fail_fast: bool = False
