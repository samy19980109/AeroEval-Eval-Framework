"""Tests for data factory."""

import json

import pytest

from aero_eval.data.factory import DataFactory
from aero_eval.models import DataSourceConfig, DataSourceType, TestCaseData


class TestFromJsonl:
    def test_loads_records(self, golden_qa_path):
        cases = DataFactory.from_jsonl(golden_qa_path)
        assert len(cases) == 2
        assert cases[0].id == "t-001"
        assert cases[0].input == "What is 2+2?"

    def test_limit(self, golden_qa_path):
        cases = DataFactory.from_jsonl(golden_qa_path, limit=1)
        assert len(cases) == 1

    def test_field_mapping(self, tmp_path):
        data = [{"question": "Hi?", "answer": "Hello!", "gold": "Hello."}]
        path = tmp_path / "mapped.jsonl"
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")

        mapping = {
            "input": "question",
            "actual_output": "answer",
            "expected_output": "gold",
        }
        cases = DataFactory.from_jsonl(path, field_mapping=mapping)
        assert len(cases) == 1
        assert cases[0].input == "Hi?"
        assert cases[0].actual_output == "Hello!"

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        cases = DataFactory.from_jsonl(path)
        assert cases == []


class TestFromDicts:
    def test_basic(self):
        records = [
            {"input": "q1", "actual_output": "a1"},
            {"input": "q2", "actual_output": "a2"},
        ]
        cases = DataFactory.from_dicts(records)
        assert len(cases) == 2
        assert cases[0].input == "q1"

    def test_missing_input_fails(self):
        with pytest.raises(Exception):
            DataFactory.from_dicts([{"actual_output": "a1"}])


class TestToDeepeval:
    def test_conversion(self):
        cases = [
            TestCaseData(
                input="q",
                actual_output="a",
                expected_output="e",
                context=["c1"],
                retrieval_context=["rc1"],
            )
        ]
        de_cases = DataFactory.to_deepeval(cases)
        assert len(de_cases) == 1
        assert de_cases[0].input == "q"
        assert de_cases[0].actual_output == "a"
        assert de_cases[0].expected_output == "e"
        assert de_cases[0].context == ["c1"]
        assert de_cases[0].retrieval_context == ["rc1"]

    def test_none_actual_output(self):
        cases = [TestCaseData(input="q")]
        de_cases = DataFactory.to_deepeval(cases)
        assert de_cases[0].actual_output == ""


class TestLoad:
    def test_jsonl_dispatch(self, golden_qa_path):
        config = DataSourceConfig(
            source_type=DataSourceType.JSONL,
            path=str(golden_qa_path),
        )
        cases = DataFactory.load(config)
        assert len(cases) == 2

    def test_jsonl_missing_path(self):
        config = DataSourceConfig(source_type=DataSourceType.JSONL)
        with pytest.raises(ValueError, match="path"):
            DataFactory.load(config)

    def test_dict_raises(self):
        config = DataSourceConfig(source_type=DataSourceType.DICT)
        with pytest.raises(ValueError, match="from_dicts"):
            DataFactory.load(config)
