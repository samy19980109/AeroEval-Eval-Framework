"""Configuration loading and validation for Aero-Eval."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from aero_eval.models import EvalConfig

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_env_vars(data: Any) -> Any:
    """Recursively replace ${ENV_VAR} placeholders with environment values."""
    if isinstance(data, str):
        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                raise ValueError(f"Environment variable '{var_name}' is not set")
            return value
        return _ENV_VAR_PATTERN.sub(_replace, data)
    elif isinstance(data, dict):
        return {k: resolve_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_env_vars(item) for item in data]
    return data


def load_config(path: Path) -> EvalConfig:
    """Load a YAML config file, resolve env vars, and validate into EvalConfig."""
    raw = load_yaml(path)
    resolved = resolve_env_vars(raw)
    return EvalConfig.model_validate(resolved)


def validate_config(path: Path) -> list[str]:
    """Validate a config file and return a list of warnings."""
    warnings: list[str] = []
    raw = load_yaml(path)

    if not raw.get("scorers"):
        warnings.append("No scorers configured")

    data_source = raw.get("data_source", {})
    source_type = data_source.get("source_type")
    if source_type == "jsonl" and data_source.get("path"):
        data_path = Path(data_source["path"])
        if not data_path.exists():
            warnings.append(f"Data file not found: {data_path}")

    if raw.get("wandb_project") and not os.environ.get("WANDB_API_KEY"):
        warnings.append("W&B project configured but WANDB_API_KEY not set")

    return warnings
