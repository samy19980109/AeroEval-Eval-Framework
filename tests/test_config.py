"""Tests for configuration loading."""

import os

import pytest
import yaml

from aero_eval.config import load_config, load_yaml, resolve_env_vars, validate_config


class TestLoadYaml:
    def test_valid_yaml(self, tmp_path):
        path = tmp_path / "test.yaml"
        path.write_text("name: test\nvalue: 42\n")
        result = load_yaml(path)
        assert result == {"name": "test", "value": 42}

    def test_empty_yaml(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        result = load_yaml(path)
        assert result == {}


class TestResolveEnvVars:
    def test_string_substitution(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret123")
        result = resolve_env_vars("api_key=${MY_KEY}")
        assert result == "api_key=secret123"

    def test_nested_dict(self, monkeypatch):
        monkeypatch.setenv("DB_HOST", "localhost")
        data = {"connection": {"host": "${DB_HOST}", "port": 5432}}
        result = resolve_env_vars(data)
        assert result["connection"]["host"] == "localhost"
        assert result["connection"]["port"] == 5432

    def test_list(self, monkeypatch):
        monkeypatch.setenv("TAG1", "prod")
        data = ["${TAG1}", "static"]
        result = resolve_env_vars(data)
        assert result == ["prod", "static"]

    def test_missing_env_var(self):
        with pytest.raises(ValueError, match="not set"):
            resolve_env_vars("${DEFINITELY_NOT_SET_AERO_EVAL_TEST}")

    def test_no_substitution_needed(self):
        result = resolve_env_vars("no vars here")
        assert result == "no vars here"

    def test_non_string_passthrough(self):
        assert resolve_env_vars(42) == 42
        assert resolve_env_vars(True) is True
        assert resolve_env_vars(None) is None


class TestLoadConfig:
    def test_valid_config(self, tmp_path):
        config = {
            "name": "test-eval",
            "data_source": {"source_type": "jsonl", "path": "test.jsonl"},
            "scorers": [{"tier": "L1", "scorer_name": "length", "min_length": 1}],
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config))
        result = load_config(path)
        assert result.name == "test-eval"

    def test_invalid_config(self, tmp_path):
        config = {"name": "bad", "data_source": {"source_type": "jsonl"}, "scorers": []}
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(config))
        with pytest.raises(Exception):
            load_config(path)


class TestValidateConfig:
    def test_missing_data_file(self, tmp_path):
        config = {
            "name": "test",
            "data_source": {
                "source_type": "jsonl",
                "path": "/nonexistent/file.jsonl",
            },
            "scorers": [{"tier": "L1", "scorer_name": "length"}],
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config))
        warnings = validate_config(path)
        assert any("not found" in w for w in warnings)

    def test_no_scorers_warning(self, tmp_path):
        config = {"name": "test", "data_source": {"source_type": "jsonl"}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config))
        warnings = validate_config(path)
        assert any("No scorers" in w for w in warnings)
