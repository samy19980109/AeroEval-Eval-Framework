"""Tests for the CLI."""

import json

import pytest
import yaml
from typer.testing import CliRunner

from aero_eval.cli import app

runner = CliRunner()


class TestValidateCommand:
    def test_valid_config(self, tmp_path, golden_qa_path):
        config = {
            "name": "test-eval",
            "data_source": {
                "source_type": "jsonl",
                "path": str(golden_qa_path),
            },
            "scorers": [{"tier": "L1", "scorer_name": "length", "min_length": 1}],
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config))

        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_invalid_config(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("name: test\n")

        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code == 1


class TestListScorersCommand:
    def test_list_all(self):
        result = runner.invoke(app, ["list-scorers"])
        assert result.exit_code == 0
        assert "regex" in result.output.lower() or "Regex" in result.output

    def test_list_by_tier(self):
        result = runner.invoke(app, ["list-scorers", "--tier", "L1"])
        assert result.exit_code == 0


class TestInspectDataCommand:
    def test_inspect(self, golden_qa_path):
        result = runner.invoke(
            app, ["inspect-data", str(golden_qa_path), "--limit", "1"]
        )
        assert result.exit_code == 0
        assert "Case 1" in result.output
        assert "1 record" in result.output


class TestRunCommand:
    def test_run_with_output(self, tmp_path, golden_qa_path):
        config = {
            "name": "cli-test",
            "data_source": {
                "source_type": "jsonl",
                "path": str(golden_qa_path),
            },
            "scorers": [
                {"tier": "L1", "scorer_name": "length", "min_length": 1, "threshold": 1.0}
            ],
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))
        output_path = tmp_path / "results.json"

        result = runner.invoke(
            app, ["run", str(config_path), "--output", str(output_path)]
        )
        assert result.exit_code == 0
        assert output_path.exists()

        results = json.loads(output_path.read_text())
        assert results["total_cases"] == 2
        assert results["passed_cases"] == 2
