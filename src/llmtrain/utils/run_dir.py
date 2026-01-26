"""Run directory creation and config persistence utilities."""

from __future__ import annotations

import shutil
from contextlib import suppress
from pathlib import Path

import yaml

from llmtrain.config.schemas import RunConfig


def create_run_directory(root_dir: str | Path, run_id: str) -> Path:
    """Create the run directory and logs subdirectory."""
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)

    run_path = root_path / run_id
    run_path.mkdir(exist_ok=False)

    try:
        (run_path / "logs").mkdir(exist_ok=False)
    except Exception:
        # Best-effort cleanup to avoid partial directories.
        with suppress(OSError):
            shutil.rmtree(run_path)
        raise

    return run_path


def write_resolved_config(run_dir: str | Path, config: RunConfig) -> Path:
    """Write resolved config to run_dir/config.yaml in canonical order."""
    run_path = Path(run_dir)
    output_path = run_path / "config.yaml"
    tmp_path = run_path / "config.yaml.tmp"

    payload = config.model_dump()

    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    tmp_path.replace(output_path)
    return output_path
