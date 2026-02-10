"""Tracking protocol and default no-op implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol


class Tracker(Protocol):
    """Tracker contract for experiment logging."""

    def start_run(self, run_name: str | None = None) -> None:
        """Start a new tracking run."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log training configuration parameters."""

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Log numeric metrics at an optional step."""

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        """Log an artifact file (e.g. config, metadata)."""

    def end_run(self) -> None:
        """Finish the tracking run."""


class NullTracker:
    """No-op tracker used when tracking is disabled or unavailable."""

    def start_run(self, run_name: str | None = None) -> None:
        _ = run_name

    def log_params(self, params: Mapping[str, Any]) -> None:
        _ = params

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        _ = metrics
        _ = step

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        _ = path
        _ = artifact_path

    def end_run(self) -> None:
        return None
