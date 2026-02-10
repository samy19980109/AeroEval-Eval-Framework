"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from aero_eval.models import (
    EvalConfig,
    DataSourceConfig,
    DataSourceType,
    L1RuleConfig,
    L2StatConfig,
    L3JudgeConfig,
    L4SystemConfig,
    NumericalGuardConfig,
    ProviderConfig,
    ProviderType,
    RAGScorerConfig,
    ScorerTier,
    TestCaseData,
    ScorerResult,
    EvalResult,
)


class TestProviderConfig:
    def test_defaults(self):
        cfg = ProviderConfig(
            provider_type=ProviderType.OPENAI, model_name="gpt-4o"
        )
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 1024
        assert cfg.enable_logprobs is False

    def test_temperature_range(self):
        with pytest.raises(ValidationError):
            ProviderConfig(
                provider_type=ProviderType.OPENAI,
                model_name="gpt-4o",
                temperature=3.0,
            )

    def test_max_tokens_positive(self):
        with pytest.raises(ValidationError):
            ProviderConfig(
                provider_type=ProviderType.OPENAI,
                model_name="gpt-4o",
                max_tokens=0,
            )


class TestScorerConfigs:
    def test_l1_defaults(self):
        cfg = L1RuleConfig(scorer_name="keyword")
        assert cfg.tier == "L1"
        assert cfg.threshold == 1.0
        assert cfg.case_sensitive is True

    def test_l2_defaults(self):
        cfg = L2StatConfig(scorer_name="bertscore")
        assert cfg.tier == "L2"
        assert cfg.threshold == 0.7

    def test_l3_requires_criteria_or_steps(self):
        with pytest.raises(ValidationError, match="criteria.*evaluation_steps"):
            L3JudgeConfig(scorer_name="geval")

    def test_l3_with_criteria(self):
        cfg = L3JudgeConfig(
            scorer_name="geval", criteria="Is the answer correct?"
        )
        assert cfg.criteria == "Is the answer correct?"

    def test_l3_with_steps(self):
        cfg = L3JudgeConfig(
            scorer_name="geval",
            evaluation_steps=["Check correctness", "Check tone"],
        )
        assert len(cfg.evaluation_steps) == 2

    def test_l4_defaults(self):
        cfg = L4SystemConfig(scorer_name="ttft")
        assert cfg.sigma_threshold == 3.0
        assert cfg.threshold == 0.8

    def test_rag_defaults(self):
        cfg = RAGScorerConfig()
        assert cfg.scorer_name == "rag_triple_check"
        assert cfg.threshold == 0.7

    def test_threshold_range(self):
        with pytest.raises(ValidationError):
            L1RuleConfig(scorer_name="keyword", threshold=1.5)


class TestTestCaseData:
    def test_minimal(self):
        case = TestCaseData(input="hello")
        assert case.input == "hello"
        assert case.actual_output is None

    def test_full(self):
        case = TestCaseData(
            id="t-1",
            input="q",
            actual_output="a",
            expected_output="e",
            context=["c1"],
            retrieval_context=["rc1"],
            metadata={"key": "val"},
        )
        assert case.context == ["c1"]


class TestEvalConfig:
    def test_from_dict(self):
        data = {
            "name": "test",
            "data_source": {"source_type": "jsonl", "path": "test.jsonl"},
            "scorers": [{"tier": "L1", "scorer_name": "length", "min_length": 1}],
        }
        cfg = EvalConfig.model_validate(data)
        assert cfg.name == "test"
        assert len(cfg.scorers) == 1

    def test_requires_scorers(self):
        with pytest.raises(ValidationError):
            EvalConfig.model_validate({
                "name": "test",
                "data_source": {"source_type": "jsonl", "path": "test.jsonl"},
                "scorers": [],
            })

    def test_serialization_roundtrip(self):
        cfg = EvalConfig(
            name="roundtrip",
            data_source=DataSourceConfig(
                source_type=DataSourceType.JSONL, path="test.jsonl"
            ),
            scorers=[L1RuleConfig(scorer_name="length")],
        )
        data = cfg.model_dump()
        restored = EvalConfig.model_validate(data)
        assert restored.name == cfg.name
        assert len(restored.scorers) == 1

    def test_discriminated_union(self):
        data = {
            "name": "test",
            "data_source": {"source_type": "jsonl", "path": "test.jsonl"},
            "scorers": [
                {"tier": "L1", "scorer_name": "regex", "pattern": "\\d+"},
                {"tier": "L2", "scorer_name": "bertscore"},
            ],
        }
        cfg = EvalConfig.model_validate(data)
        assert isinstance(cfg.scorers[0], L1RuleConfig)
        assert isinstance(cfg.scorers[1], L2StatConfig)


class TestScorerResult:
    def test_defaults(self):
        r = ScorerResult(
            scorer_name="test", tier=ScorerTier.L1, score=0.8, passed=True
        )
        assert r.latency_ms == 0.0
        assert r.metadata == {}
