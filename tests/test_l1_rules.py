"""Tests for L1 rule-based scorers."""

import pytest
from deepeval.test_case import LLMTestCase

from aero_eval.models import L1RuleConfig
from aero_eval.scorers.l1_rules import (
    JsonSchemaScorer,
    KeywordScorer,
    LengthScorer,
    RegexScorer,
)


def _make_case(actual_output: str) -> LLMTestCase:
    return LLMTestCase(input="test", actual_output=actual_output)


class TestRegexScorer:
    def test_match(self):
        config = L1RuleConfig(scorer_name="regex", pattern=r"\d+", threshold=1.0)
        scorer = RegexScorer(config)
        score = scorer.measure(_make_case("The answer is 42"))
        assert score == 1.0
        assert scorer.is_successful()

    def test_no_match(self):
        config = L1RuleConfig(scorer_name="regex", pattern=r"\d+", threshold=1.0)
        scorer = RegexScorer(config)
        score = scorer.measure(_make_case("No numbers here"))
        assert score == 0.0
        assert not scorer.is_successful()

    def test_case_insensitive(self):
        config = L1RuleConfig(
            scorer_name="regex",
            pattern=r"hello",
            case_sensitive=False,
            threshold=1.0,
        )
        scorer = RegexScorer(config)
        score = scorer.measure(_make_case("HELLO world"))
        assert score == 1.0

    def test_empty_output(self):
        config = L1RuleConfig(scorer_name="regex", pattern=r"\w+", threshold=1.0)
        scorer = RegexScorer(config)
        score = scorer.measure(_make_case(""))
        assert score == 0.0


class TestJsonSchemaScorer:
    def test_valid_json(self):
        config = L1RuleConfig(scorer_name="json_schema", threshold=1.0)
        scorer = JsonSchemaScorer(config)
        score = scorer.measure(_make_case('{"key": "value"}'))
        assert score == 1.0

    def test_invalid_json(self):
        config = L1RuleConfig(scorer_name="json_schema", threshold=1.0)
        scorer = JsonSchemaScorer(config)
        score = scorer.measure(_make_case("not json"))
        assert score == 0.0

    def test_schema_validation_pass(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        config = L1RuleConfig(
            scorer_name="json_schema", json_schema=schema, threshold=1.0
        )
        scorer = JsonSchemaScorer(config)
        score = scorer.measure(_make_case('{"name": "Alice", "age": 30}'))
        assert score == 1.0

    def test_schema_validation_fail(self):
        schema = {"type": "object", "required": ["name"]}
        config = L1RuleConfig(
            scorer_name="json_schema", json_schema=schema, threshold=1.0
        )
        scorer = JsonSchemaScorer(config)
        score = scorer.measure(_make_case('{"age": 30}'))
        assert score == 0.0


class TestKeywordScorer:
    def test_all_keywords_present(self):
        config = L1RuleConfig(
            scorer_name="keyword",
            expected_keywords=["python", "data"],
            case_sensitive=False,
            threshold=0.5,
        )
        scorer = KeywordScorer(config)
        score = scorer.measure(_make_case("Python is great for data science"))
        assert score == 1.0

    def test_partial_keywords(self):
        config = L1RuleConfig(
            scorer_name="keyword",
            expected_keywords=["python", "java", "rust"],
            threshold=0.3,
        )
        scorer = KeywordScorer(config)
        score = scorer.measure(_make_case("I love python"))
        assert abs(score - 1 / 3) < 0.01

    def test_forbidden_keywords(self):
        config = L1RuleConfig(
            scorer_name="keyword",
            expected_keywords=["answer"],
            forbidden_keywords=["error", "unknown"],
            threshold=0.5,
        )
        scorer = KeywordScorer(config)
        score = scorer.measure(_make_case("The answer has an error"))
        # expected_ratio = 1.0, forbidden_penalty = 0.5
        # score = 1.0 * (1.0 - 0.5) = 0.5
        assert abs(score - 0.5) < 0.01

    def test_case_insensitive(self):
        config = L1RuleConfig(
            scorer_name="keyword",
            expected_keywords=["HELLO"],
            case_sensitive=False,
            threshold=0.5,
        )
        scorer = KeywordScorer(config)
        score = scorer.measure(_make_case("hello world"))
        assert score == 1.0

    def test_no_keywords(self):
        config = L1RuleConfig(scorer_name="keyword", threshold=0.5)
        scorer = KeywordScorer(config)
        score = scorer.measure(_make_case("anything"))
        assert score == 1.0


class TestLengthScorer:
    def test_within_bounds(self):
        config = L1RuleConfig(
            scorer_name="length", min_length=1, max_length=100, threshold=1.0
        )
        scorer = LengthScorer(config)
        score = scorer.measure(_make_case("Hello world"))
        assert score == 1.0

    def test_too_short(self):
        config = L1RuleConfig(
            scorer_name="length", min_length=100, threshold=1.0
        )
        scorer = LengthScorer(config)
        score = scorer.measure(_make_case("Hi"))
        assert score == 0.0

    def test_too_long(self):
        config = L1RuleConfig(
            scorer_name="length", max_length=5, threshold=1.0
        )
        scorer = LengthScorer(config)
        score = scorer.measure(_make_case("This is too long"))
        assert score == 0.0

    def test_no_bounds(self):
        config = L1RuleConfig(scorer_name="length", threshold=1.0)
        scorer = LengthScorer(config)
        score = scorer.measure(_make_case(""))
        assert score == 1.0

    def test_empty_output(self):
        config = L1RuleConfig(
            scorer_name="length", min_length=1, threshold=1.0
        )
        scorer = LengthScorer(config)
        score = scorer.measure(_make_case(""))
        assert score == 0.0
