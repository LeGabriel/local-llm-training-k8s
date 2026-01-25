"""Run metadata generation and persistence utilities."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _get_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"

    sha = result.stdout.strip()
    return sha or None


def _read_env_var(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def generate_meta(
    *,
    run_id: str,
    run_name: str,
    config_path: str | None,
    resolved_config_path: str | None,
) -> dict[str, Any]:
    """Generate run metadata payload."""
    ddp_env = {
        "RANK": _read_env_var("RANK"),
        "WORLD_SIZE": _read_env_var("WORLD_SIZE"),
        "LOCAL_RANK": _read_env_var("LOCAL_RANK"),
        "MASTER_ADDR": _read_env_var("MASTER_ADDR"),
        "MASTER_PORT": _read_env_var("MASTER_PORT"),
    }

    return {
        "meta_version": 1,
        "run_id": run_id,
        "run_name": run_name,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_sha": _get_git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "config_path": config_path,
        "resolved_config_path": resolved_config_path,
        "ddp_env": ddp_env,
        "hostname": platform.node(),
        "pid": os.getpid(),
    }


def write_meta_json(run_dir: str | Path, meta: dict[str, Any]) -> Path:
    """Write metadata to run_dir/meta.json."""
    run_path = Path(run_dir)
    output_path = run_path / "meta.json"
    tmp_path = run_path / "meta.json.tmp"

    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    tmp_path.replace(output_path)
    return output_path
