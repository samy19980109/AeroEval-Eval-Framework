"""L1 Scorers: Deterministic rule-based validation."""

from __future__ import annotations

import json
import re

from deepeval.test_case import LLMTestCase

from aero_eval.models import L1RuleConfig, ScorerTier
from aero_eval.registry import ScorerRegistry
from aero_eval.scorers.base import BaseScorer


@ScorerRegistry.register("regex", ScorerTier.L1)
class RegexScorer(BaseScorer):
    """Checks if actual_output matches a regex pattern."""

    tier = ScorerTier.L1

    def __init__(self, config: L1RuleConfig):
        super().__init__(config)
        flags = 0 if config.case_sensitive else re.IGNORECASE
        self._pattern = re.compile(config.pattern or "", flags=flags)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        match = self._pattern.search(text)
        self.score = 1.0 if match else 0.0
        self.reason = f"Pattern {'matched' if match else 'not matched'}: {self.config.pattern}"
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "L1::Regex"


@ScorerRegistry.register("json_schema", ScorerTier.L1)
class JsonSchemaScorer(BaseScorer):
    """Validates that actual_output is valid JSON matching a schema."""

    tier = ScorerTier.L1

    def __init__(self, config: L1RuleConfig):
        super().__init__(config)
        self._schema = config.json_schema

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        try:
            parsed = json.loads(text)
            if self._schema:
                import jsonschema

                jsonschema.validate(parsed, self._schema)
            self.score = 1.0
            self.reason = "Valid JSON matching schema"
        except json.JSONDecodeError as e:
            self.score = 0.0
            self.reason = f"Invalid JSON: {str(e)[:200]}"
        except Exception as e:
            self.score = 0.0
            self.reason = f"Schema validation failed: {str(e)[:200]}"
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "L1::JsonSchema"


@ScorerRegistry.register("keyword", ScorerTier.L1)
class KeywordScorer(BaseScorer):
    """Checks presence of expected keywords and absence of forbidden keywords."""

    tier = ScorerTier.L1

    def __init__(self, config: L1RuleConfig):
        super().__init__(config)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        if not self.config.case_sensitive:
            text = text.lower()

        expected = self.config.expected_keywords
        if expected:
            matches = sum(
                1
                for kw in expected
                if (kw.lower() if not self.config.case_sensitive else kw) in text
            )
            expected_ratio = matches / len(expected)
        else:
            expected_ratio = 1.0

        forbidden = self.config.forbidden_keywords
        if forbidden:
            violations = sum(
                1
                for kw in forbidden
                if (kw.lower() if not self.config.case_sensitive else kw) in text
            )
            forbidden_penalty = violations / len(forbidden)
        else:
            forbidden_penalty = 0.0

        self.score = expected_ratio * (1.0 - forbidden_penalty)
        self.reason = (
            f"Expected: {expected_ratio:.0%} matched, "
            f"Forbidden: {forbidden_penalty:.0%} found"
        )
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "L1::Keyword"


@ScorerRegistry.register("length", ScorerTier.L1)
class LengthScorer(BaseScorer):
    """Validates output length is within bounds."""

    tier = ScorerTier.L1

    def __init__(self, config: L1RuleConfig):
        super().__init__(config)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        text = test_case.actual_output or ""
        length = len(text)
        min_ok = self.config.min_length is None or length >= self.config.min_length
        max_ok = self.config.max_length is None or length <= self.config.max_length
        self.score = 1.0 if (min_ok and max_ok) else 0.0
        self.reason = (
            f"Length {length} chars "
            f"(min={self.config.min_length}, max={self.config.max_length})"
        )
        self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "L1::Length"
