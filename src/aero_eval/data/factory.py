"""Data ingestion layer for Aero-Eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepeval.test_case import LLMTestCase

from aero_eval.models import DataSourceConfig, DataSourceType, TestCaseData


class DataFactory:
    """Unified data loading and conversion for evaluation datasets."""

    @staticmethod
    def load(config: DataSourceConfig) -> list[TestCaseData]:
        """Dispatch to the correct loader based on source_type."""
        if config.source_type == DataSourceType.JSONL:
            if not config.path:
                raise ValueError("JSONL source requires 'path'")
            return DataFactory.from_jsonl(
                Path(config.path),
                field_mapping=config.field_mapping,
                limit=config.limit,
            )
        elif config.source_type == DataSourceType.HUGGINGFACE:
            if not config.dataset_name:
                raise ValueError("HuggingFace source requires 'dataset_name'")
            return DataFactory.from_huggingface(
                config.dataset_name,
                split=config.split,
                field_mapping=config.field_mapping,
                limit=config.limit,
            )
        elif config.source_type == DataSourceType.DICT:
            raise ValueError("DICT source must be loaded via from_dicts()")
        else:
            raise ValueError(f"Unknown source type: {config.source_type}")

    @staticmethod
    def from_jsonl(
        path: Path,
        field_mapping: dict[str, str] | None = None,
        limit: int | None = None,
    ) -> list[TestCaseData]:
        """Load test cases from a JSONL file."""
        field_mapping = field_mapping or {}
        cases: list[TestCaseData] = []

        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                mapped = _apply_field_mapping(record, field_mapping)
                cases.append(TestCaseData.model_validate(mapped))

        return cases

    @staticmethod
    def from_huggingface(
        dataset_name: str,
        split: str = "test",
        field_mapping: dict[str, str] | None = None,
        limit: int | None = None,
    ) -> list[TestCaseData]:
        """Load test cases from a HuggingFace dataset."""
        from datasets import load_dataset

        field_mapping = field_mapping or {}
        ds = load_dataset(dataset_name, split=split)

        cases: list[TestCaseData] = []
        for i, record in enumerate(ds):
            if limit is not None and i >= limit:
                break
            mapped = _apply_field_mapping(dict(record), field_mapping)
            cases.append(TestCaseData.model_validate(mapped))

        return cases

    @staticmethod
    def from_dicts(records: list[dict[str, Any]]) -> list[TestCaseData]:
        """Convert raw dicts to TestCaseData models."""
        return [TestCaseData.model_validate(r) for r in records]

    @staticmethod
    def to_deepeval(cases: list[TestCaseData]) -> list[LLMTestCase]:
        """Convert TestCaseData to DeepEval LLMTestCase objects."""
        result: list[LLMTestCase] = []
        for case in cases:
            result.append(
                LLMTestCase(
                    input=case.input,
                    actual_output=case.actual_output or "",
                    expected_output=case.expected_output,
                    context=case.context,
                    retrieval_context=case.retrieval_context,
                )
            )
        return result


def _apply_field_mapping(
    record: dict[str, Any], mapping: dict[str, str]
) -> dict[str, Any]:
    """Apply field mapping: mapping keys are target field names, values are source field names."""
    if not mapping:
        return record
    result = dict(record)
    for target_field, source_field in mapping.items():
        if source_field in record:
            result[target_field] = record[source_field]
    return result
