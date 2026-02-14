"""MLflow-backed tracker implementation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _flatten_params(params: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping into dot-separated keys."""
    flattened: dict[str, Any] = {}
    for key, value in params.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_params(value, prefix=flat_key))
            continue
        if isinstance(value, (list, tuple, set)):
            flattened[flat_key] = json.dumps(list(value), default=str)
            continue
        if value is None:
            flattened[flat_key] = "None"
            continue
        if isinstance(value, (str, int, float, bool)):
            flattened[flat_key] = value
            continue
        flattened[flat_key] = str(value)
    return flattened


class MLflowTracker:
    """Tracker adapter around the ``mlflow`` Python client."""

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment: str,
        run_name: str | None = None,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment = experiment
        self._run_name = run_name
        try:
            import mlflow  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MLflowTracker requires the optional 'mlflow' dependency. "
                "Install it with `pip install -e '.[mlflow]'`."
            ) from exc
        self._mlflow = mlflow

    def start_run(self, run_name: str | None = None, *, run_id: str | None = None) -> None:
        self._mlflow.set_tracking_uri(self._tracking_uri)
        self._mlflow.set_experiment(self._experiment)
        if run_id is not None:
            # Join an existing run (used by non-rank-0 DDP workers).
            self._mlflow.start_run(run_id=run_id)
        else:
            self._mlflow.start_run(run_name=run_name or self._run_name)

    @property
    def active_run_id(self) -> str | None:
        """Return the run-id of the currently active MLflow run, if any."""
        run = self._mlflow.active_run()
        return run.info.run_id if run is not None else None

    def log_params(self, params: Mapping[str, Any]) -> None:
        flattened = _flatten_params(params)
        if flattened:
            self._mlflow.log_params(flattened)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        if not metrics:
            return
        payload = {name: float(value) for name, value in metrics.items()}
        if step is None:
            self._mlflow.log_metrics(payload)
            return
        self._mlflow.log_metrics(payload, step=step)

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def end_run(self) -> None:
        self._mlflow.end_run()
