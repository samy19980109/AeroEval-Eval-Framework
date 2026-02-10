"""Shared fixtures for Aero-Eval tests."""

import pytest

from aero_eval.models import (
    DataSourceConfig,
    DataSourceType,
    EvalConfig,
    L1RuleConfig,
    L2StatConfig,
    TestCaseData,
)


@pytest.fixture
def sample_test_case() -> TestCaseData:
    return TestCaseData(
        id="test-001",
        input="What is 2+2?",
        actual_output="The answer is 4.",
        expected_output="4",
    )


@pytest.fixture
def sample_rag_test_case() -> TestCaseData:
    return TestCaseData(
        id="rag-001",
        input="What are the side effects of aspirin?",
        actual_output="Aspirin can cause stomach irritation.",
        expected_output="Common side effects include stomach upset.",
        retrieval_context=[
            "Aspirin may cause gastrointestinal upset and bleeding.",
        ],
    )


@pytest.fixture
def sample_l1_keyword_config() -> L1RuleConfig:
    return L1RuleConfig(
        scorer_name="keyword",
        expected_keywords=["4", "answer"],
        threshold=0.5,
    )


@pytest.fixture
def sample_l1_regex_config() -> L1RuleConfig:
    return L1RuleConfig(
        scorer_name="regex",
        pattern=r"\d+",
        threshold=1.0,
    )


@pytest.fixture
def sample_l2_rouge_config() -> L2StatConfig:
    return L2StatConfig(
        scorer_name="rouge",
        rouge_types=["rouge1", "rougeL"],
        threshold=0.3,
    )


@pytest.fixture
def golden_qa_path(tmp_path):
    """Create a temporary JSONL file for testing."""
    import json

    data = [
        {
            "id": "t-001",
            "input": "What is 2+2?",
            "expected_output": "4",
            "actual_output": "The answer is 4.",
        },
        {
            "id": "t-002",
            "input": "Capital of France?",
            "expected_output": "Paris",
            "actual_output": "Paris is the capital.",
        },
    ]
    path = tmp_path / "test_data.jsonl"
    with open(path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def minimal_eval_config(golden_qa_path) -> EvalConfig:
    return EvalConfig(
        name="test-eval",
        data_source=DataSourceConfig(
            source_type=DataSourceType.JSONL,
            path=str(golden_qa_path),
        ),
        scorers=[
            L1RuleConfig(
                scorer_name="length",
                min_length=1,
                max_length=1000,
                threshold=1.0,
            ),
        ],
    )
