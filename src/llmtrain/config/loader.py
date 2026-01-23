"""YAML configuration loading and validation utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, NoReturn

import yaml
from pydantic import ValidationError

from llmtrain.config.schemas import RunConfig


def _exit_config_error(message: str, details: str | None = None) -> NoReturn:
    print(f"Config error: {message}", file=sys.stderr)
    if details:
        print(details, file=sys.stderr)
    raise SystemExit(2)


def resolve_config_path(config_path: str) -> tuple[str, Path]:
    """Return the provided config path and its absolute resolved path."""
    if not config_path or not config_path.strip():
        _exit_config_error("config path must be a non-empty string")

    raw_path = config_path
    path = Path(config_path).expanduser()
    resolved = path if path.is_absolute() else (Path.cwd() / path)
    return raw_path, resolved.resolve()


def load_yaml_config(config_path: Path) -> Any:
    """Load YAML safely, exiting with code 2 on parse errors."""
    try:
        with config_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        _exit_config_error(f"YAML parse error in {config_path}: {exc}")
    except OSError as exc:
        _exit_config_error(f"unable to read config file {config_path}: {exc}")

    return {} if data is None else data


def load_and_validate_config(config_path: str) -> tuple[RunConfig, str, Path]:
    """Load YAML config, validate with Pydantic, and return config + paths."""
    raw_path, resolved_path = resolve_config_path(config_path)
    raw_config = load_yaml_config(resolved_path)

    if not isinstance(raw_config, dict):
        _exit_config_error(f"top-level config must be a mapping: {resolved_path}")

    try:
        resolved = RunConfig.model_validate(raw_config)
    except ValidationError as exc:
        _exit_config_error(
            f"validation failed for {resolved_path}",
            details=str(exc),
        )

    return resolved, raw_path, resolved_path
