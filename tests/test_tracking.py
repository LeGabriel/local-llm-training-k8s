from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest.mock import Mock

import pytest

from llmtrain.tracking import MLflowTracker, NullTracker
from llmtrain.tracking.mlflow import _flatten_params


def test_null_tracker_methods_are_callable_without_error() -> None:
    tracker = NullTracker()

    tracker.start_run(run_name="smoke-run")
    tracker.log_params({"batch_size": 8, "use_amp": False})
    tracker.log_metrics({"train/loss": 1.23, "train/lr": 1e-3}, step=1)
    tracker.log_artifact(Path("config.yaml"), artifact_path="configs")
    tracker.end_run()


def test_mlflow_tracker_lifecycle_and_call_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow: Any = ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = Mock()
    fake_mlflow.set_experiment = Mock()
    fake_mlflow.start_run = Mock()
    fake_mlflow.log_params = Mock()
    fake_mlflow.log_metrics = Mock()
    fake_mlflow.log_artifact = Mock()
    fake_mlflow.end_run = Mock()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    tracker = MLflowTracker(
        tracking_uri="file:./mlruns",
        experiment="llm-train-k8s",
        run_name="unit-run",
    )

    tracker.start_run()
    tracker.log_params(
        {
            "trainer": {"max_steps": 10, "schedule": {"warmup": 5}},
            "tags": ["smoke", "tracking"],
            "nullable": None,
        }
    )
    tracker.log_metrics({"train/loss": 1.0}, step=3)
    tracker.log_metrics({"val/loss": 0.9})
    tracker.log_artifact(Path("config.yaml"), artifact_path="configs")
    tracker.end_run()

    cast(Mock, fake_mlflow.set_tracking_uri).assert_called_once_with("file:./mlruns")
    cast(Mock, fake_mlflow.set_experiment).assert_called_once_with("llm-train-k8s")
    cast(Mock, fake_mlflow.start_run).assert_called_once_with(run_name="unit-run")
    cast(Mock, fake_mlflow.log_params).assert_called_once_with(
        {
            "trainer.max_steps": 10,
            "trainer.schedule.warmup": 5,
            "tags": '["smoke", "tracking"]',
            "nullable": "None",
        }
    )
    cast(Mock, fake_mlflow.log_metrics).assert_any_call({"train/loss": 1.0}, step=3)
    cast(Mock, fake_mlflow.log_metrics).assert_any_call({"val/loss": 0.9})
    cast(Mock, fake_mlflow.log_artifact).assert_called_once_with(
        "config.yaml", artifact_path="configs"
    )
    cast(Mock, fake_mlflow.end_run).assert_called_once_with()


def test_flatten_params_uses_dot_keys_for_nested_values() -> None:
    flattened = _flatten_params(
        {
            "run": {"name": "demo"},
            "trainer": {"lr": 0.001, "accum": {"steps": 4}},
            "flags": {"enabled": True},
        }
    )
    assert flattened == {
        "run.name": "demo",
        "trainer.lr": 0.001,
        "trainer.accum.steps": 4,
        "flags.enabled": True,
    }


def test_mlflow_tracker_integration_local_sqlite_backend(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_root = tmp_path / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    experiment_name = "v0_7_2_integration"

    # Ensure artifacts are stored in the temp directory instead of ./mlruns.
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())

    artifact_file = tmp_path / "artifact.txt"
    artifact_file.write_text("mlflow-artifact", encoding="utf-8")

    tracker = MLflowTracker(
        tracking_uri=tracking_uri,
        experiment=experiment_name,
        run_name="integration-run",
    )
    tracker.start_run()

    active_run = mlflow.active_run()
    assert active_run is not None
    run_id = active_run.info.run_id

    tracker.log_params({"trainer": {"max_steps": 7}})
    tracker.log_metrics({"train/loss": 0.42}, step=5)
    tracker.log_artifact(artifact_file, artifact_path="artifacts")
    tracker.end_run()

    run = client.get_run(run_id)
    assert run.data.params["trainer.max_steps"] == "7"
    assert run.data.metrics["train/loss"] == 0.42

    metric_history = client.get_metric_history(run_id, "train/loss")
    assert metric_history
    assert metric_history[-1].step == 5

    artifact_paths = [item.path for item in client.list_artifacts(run_id, path="artifacts")]
    assert "artifacts/artifact.txt" in artifact_paths


def test_mlflow_dependency_is_optional_not_core() -> None:
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    core_deps = pyproject["project"]["dependencies"]
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert all(not dep.startswith("mlflow") for dep in core_deps)
    assert "mlflow" in optional_deps
    assert any(dep.startswith("mlflow") for dep in optional_deps["mlflow"])
