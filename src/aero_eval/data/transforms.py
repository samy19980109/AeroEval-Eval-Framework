"""Data transformation utilities for Aero-Eval."""

from __future__ import annotations

import uuid

from aero_eval.models import TestCaseData


def normalize_whitespace(cases: list[TestCaseData]) -> list[TestCaseData]:
    """Strip and normalize whitespace in all text fields."""
    result: list[TestCaseData] = []
    for case in cases:
        data = case.model_dump()
        for field in ("input", "actual_output", "expected_output"):
            if data.get(field) and isinstance(data[field], str):
                data[field] = " ".join(data[field].split())
        result.append(TestCaseData.model_validate(data))
    return result


def truncate_outputs(
    cases: list[TestCaseData], max_chars: int
) -> list[TestCaseData]:
    """Truncate actual_output to max_chars."""
    result: list[TestCaseData] = []
    for case in cases:
        if case.actual_output and len(case.actual_output) > max_chars:
            data = case.model_dump()
            data["actual_output"] = case.actual_output[:max_chars]
            result.append(TestCaseData.model_validate(data))
        else:
            result.append(case)
    return result


def assign_ids(cases: list[TestCaseData]) -> list[TestCaseData]:
    """Assign sequential IDs to cases that lack them."""
    result: list[TestCaseData] = []
    for i, case in enumerate(cases):
        if case.id is None:
            data = case.model_dump()
            data["id"] = f"case-{i:04d}"
            result.append(TestCaseData.model_validate(data))
        else:
            result.append(case)
    return result
