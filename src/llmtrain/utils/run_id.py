"""Run ID generation utilities."""

from __future__ import annotations

import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

_SLUG_DISALLOWED_RE = re.compile(r"[^a-z0-9\-_]+")


def _get_short_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"

    sha = result.stdout.strip()
    return sha or "nogit"


def slugify_run_name(name: str) -> str:
    """Return a filesystem-friendly slug for run names."""
    slug = name.strip().lower()
    slug = _SLUG_DISALLOWED_RE.sub("_", slug)
    slug = slug.strip("-_")
    slug = re.sub(r"[-_]{2,}", "_", slug)
    if not slug:
        return "run"
    return slug[:40]


def _append_collision_suffix(run_id: str, root_dir: Path) -> str:
    if not (root_dir / run_id).exists():
        return run_id

    for suffix in range(1, 100):
        candidate = f"{run_id}__{suffix:02d}"
        if not (root_dir / candidate).exists():
            return candidate

    raise RuntimeError("Run ID collision limit reached (tried __01 through __99).")


def generate_run_id(run_name: str, root_dir: str | Path | None = None) -> str:
    """Generate a run ID with optional collision handling."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    short_sha = _get_short_git_sha()
    slug = slugify_run_name(run_name)
    run_id = f"{timestamp}_{short_sha}_{slug}"

    if root_dir is None:
        return run_id

    root_path = Path(root_dir)
    return _append_collision_suffix(run_id, root_path)
