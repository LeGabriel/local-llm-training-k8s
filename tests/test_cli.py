from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import yaml

import llmtrain
from llmtrain import cli as cli_module
from llmtrain.config.schemas import RunConfig


def test_imports() -> None:
    assert llmtrain.__version__


def test_cli_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "llmtrain", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _minimal_config(root_dir: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run": {"name": "cli-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {},
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {
            "root_dir": str(root_dir),
            "save_config_copy": True,
            "save_meta_json": True,
        },
    }


def test_validate_command_success(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _minimal_config(tmp_path / "runs"))

    result = subprocess.run(
        [sys.executable, "-m", "llmtrain", "validate", "--config", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "succeeded" in result.stdout.lower()


def test_validate_command_failure_json(tmp_path: Path) -> None:
    payload = _minimal_config(tmp_path / "runs")
    payload["model"] = {"name": "tiny-model", "d_model": 384, "n_heads": 7}
    config_path = _write_config(tmp_path, payload)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmtrain",
            "validate",
            "--config",
            str(config_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"


def test_print_config_command_json(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _minimal_config(tmp_path / "runs"))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmtrain",
            "print-config",
            "--config",
            str(config_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["run"]["name"] == "cli-test"
    assert payload["trainer"]["max_steps"] == 1000


def test_train_dry_run_creates_outputs(tmp_path: Path) -> None:
    root_dir = tmp_path / "runs"
    payload = _minimal_config(root_dir)
    payload["trainer"] = {"max_steps": 5, "warmup_steps": 0, "micro_batch_size": 1}
    config_path = _write_config(tmp_path, payload)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmtrain",
            "train",
            "--config",
            str(config_path),
            "--run-id",
            "unit-test-run",
            "--dry-run",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    summary = json.loads(result.stdout)
    assert summary["run_id"] == "unit-test-run"
    assert summary["resolved_model_adapter"] == "dummy_gpt"
    assert summary["resolved_data_module"] == "dummy_text"
    assert summary["dry_run_steps_executed"] == 5

    run_dir = root_dir / "unit-test-run"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "meta.json").exists()


def test_train_full_run_emits_training_summary(tmp_path: Path) -> None:
    """Full training (no --dry-run) emits a training summary with sane metrics.

    Note: we do *not* assert monotonic loss decrease here because the CLI path runs a
    stochastic training loop in a subprocess (model init, RNG, etc.). The "loss decreases"
    behavior is covered by `tests/test_trainer.py`.
    """
    root_dir = tmp_path / "runs"
    config_payload = _minimal_config(root_dir)
    config_payload["trainer"] = {
        "max_steps": 10,
        "warmup_steps": 0,
        "log_every_steps": 5,
        "micro_batch_size": 1,
        "grad_accum_steps": 1,
    }
    config_path = _write_config(tmp_path, config_payload)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmtrain",
            "train",
            "--config",
            str(config_path),
            "--run-id",
            "full-train-test",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    summary = json.loads(result.stdout)
    assert summary["run_id"] == "full-train-test"
    assert "training" in summary
    training = summary["training"]
    assert training["final_step"] == 10
    assert training["total_time"] > 0.0
    assert training["first_step_loss"] is not None
    assert math.isfinite(training["first_step_loss"])
    assert math.isfinite(training["final_loss"])
    assert "val_metrics" in training
    assert math.isfinite(training["val_metrics"]["val/loss"])


def test_train_full_run_text_output(tmp_path: Path) -> None:
    """Full training without --json emits human-readable text with Training line."""
    root_dir = tmp_path / "runs"
    config_payload = _minimal_config(root_dir)
    config_payload["trainer"] = {
        "max_steps": 5,
        "warmup_steps": 0,
        "log_every_steps": 5,
        "micro_batch_size": 1,
        "grad_accum_steps": 1,
    }
    config_path = _write_config(tmp_path, config_payload)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmtrain",
            "train",
            "--config",
            str(config_path),
            "--run-id",
            "full-train-text",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Training:" in result.stdout
    assert "final_step=" in result.stdout
    assert "final_loss=" in result.stdout
    assert "Validation:" in result.stdout
    assert "val/loss=" in result.stdout


# ---------------------------------------------------------------------------
# Resume tests
# ---------------------------------------------------------------------------


def test_train_resume_by_run_id(tmp_path: Path) -> None:
    """Resume via --resume <run_id> continues training to a higher max_steps."""
    root_dir = tmp_path / "runs"
    config_payload = _minimal_config(root_dir)
    config_payload["trainer"] = {
        "max_steps": 10,
        "warmup_steps": 0,
        "log_every_steps": 5,
        "save_every_steps": 10,
        "micro_batch_size": 1,
        "grad_accum_steps": 1,
    }
    trainer_payload = cast(dict[str, Any], config_payload["trainer"])
    config_path = _write_config(tmp_path, config_payload)

    # First run: train 10 steps.
    result1 = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmtrain",
            "train",
            "--config",
            str(config_path),
            "--run-id",
            "resume-run-id-test",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result1.returncode == 0, f"stderr: {result1.stderr}"
    summary1 = json.loads(result1.stdout)
    assert summary1["training"]["final_step"] == 10

    # Write a new config with higher max_steps.
    trainer_payload["max_steps"] = 20
    config_path2 = _write_config(tmp_path / "config2", config_payload)

    # Second run: resume from the first run's run_id.
    result2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmtrain",
            "train",
            "--config",
            str(config_path2),
            "--run-id",
            "resume-run-id-test-2",
            "--resume",
            str(root_dir / "resume-run-id-test" / "checkpoints"),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result2.returncode == 0, f"stderr: {result2.stderr}"
    summary2 = json.loads(result2.stdout)
    assert summary2["training"]["final_step"] == 20
    assert summary2["training"]["resumed_from_step"] == 10
    assert summary2["resumed_from"] is not None


def test_train_resume_by_path(tmp_path: Path) -> None:
    """Resume via --resume <path/to/step_*.pt> continues training."""
    root_dir = tmp_path / "runs"
    config_payload = _minimal_config(root_dir)
    config_payload["trainer"] = {
        "max_steps": 10,
        "warmup_steps": 0,
        "log_every_steps": 5,
        "save_every_steps": 10,
        "micro_batch_size": 1,
        "grad_accum_steps": 1,
    }
    trainer_payload = cast(dict[str, Any], config_payload["trainer"])
    config_path = _write_config(tmp_path, config_payload)

    # First run: train 10 steps.
    result1 = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmtrain",
            "train",
            "--config",
            str(config_path),
            "--run-id",
            "resume-path-test",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result1.returncode == 0, f"stderr: {result1.stderr}"

    # Find the checkpoint file.
    ckpt_path = root_dir / "resume-path-test" / "checkpoints" / "step_000010.pt"
    assert ckpt_path.exists(), f"Expected checkpoint at {ckpt_path}"

    # Write a new config with higher max_steps.
    trainer_payload["max_steps"] = 20
    config_path2 = _write_config(tmp_path / "config2", config_payload)

    # Second run: resume from the direct .pt path.
    result2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmtrain",
            "train",
            "--config",
            str(config_path2),
            "--run-id",
            "resume-path-test-2",
            "--resume",
            str(ckpt_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result2.returncode == 0, f"stderr: {result2.stderr}"
    summary2 = json.loads(result2.stdout)
    assert summary2["training"]["final_step"] == 20
    assert summary2["training"]["resumed_from_step"] == 10
    assert summary2["resumed_from"] == str(ckpt_path)


def _minimal_run_config(root_dir: Path) -> RunConfig:
    payload = _minimal_config(root_dir)
    payload["trainer"] = {
        "max_steps": 1,
        "warmup_steps": 0,
        "log_every_steps": 1,
        "eval_every_steps": 1,
        "micro_batch_size": 1,
        "grad_accum_steps": 1,
    }
    payload["mlflow"] = {
        "enabled": True,
        "tracking_uri": "sqlite:///./mlflow.db",
        "experiment": "cli-test",
        "run_name": None,
    }
    return RunConfig.model_validate(payload)


def test_handle_train_wires_tracker_lifecycle_and_artifacts(
    monkeypatch: Any, tmp_path: Path
) -> None:
    config = _minimal_run_config(tmp_path / "runs")
    args = argparse.Namespace(
        config=str(tmp_path / "config.yaml"),
        run_id="cli-wiring-run",
        dry_run=False,
        json=False,
        verbose=0,
        resume=None,
    )
    tracker = Mock()

    def fake_load_and_validate_config(_: str) -> tuple[RunConfig, str, str]:
        return (config, "raw.yaml", "resolved.yaml")

    def fake_create_run_directory(_: str, run_id: str) -> Path:
        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def fake_write_resolved_config(run_dir: Path, _: RunConfig) -> None:
        (run_dir / "config.yaml").write_text("schema_version: 1\n", encoding="utf-8")

    def fake_write_meta_json(run_dir: Path, _: dict[str, Any]) -> None:
        (run_dir / "meta.json").write_text('{"ok": true}\n', encoding="utf-8")

    monkeypatch.setattr(cli_module, "load_and_validate_config", fake_load_and_validate_config)
    monkeypatch.setattr(cli_module, "create_run_directory", fake_create_run_directory)
    monkeypatch.setattr(cli_module, "write_resolved_config", fake_write_resolved_config)
    monkeypatch.setattr(cli_module, "write_meta_json", fake_write_meta_json)
    monkeypatch.setattr(cli_module, "_create_tracker", lambda *_: tracker)
    monkeypatch.setattr(cli_module, "initialize_registries", lambda: None)
    monkeypatch.setattr(cli_module, "get_model_adapter", lambda _: object)
    monkeypatch.setattr(cli_module, "get_data_module", lambda _: object)
    monkeypatch.setattr(cli_module, "_configure_logger", lambda *_, **__: Mock())
    monkeypatch.setattr(cli_module, "generate_meta", lambda **_: {"meta": "ok"})
    monkeypatch.setattr(cli_module, "format_run_summary", lambda **_: "ok")

    trainer_result = Mock()
    trainer_result.final_step = 1
    trainer_instance = Mock()
    trainer_instance.fit.return_value = trainer_result
    trainer_ctor = Mock(return_value=trainer_instance)
    monkeypatch.setattr(cli_module, "Trainer", trainer_ctor)

    rc = cli_module._handle_train(args)
    assert rc == 0
    tracker.start_run.assert_called_once_with(run_name="cli-wiring-run")
    tracker.log_artifact.assert_any_call(
        tmp_path / "runs" / "cli-wiring-run" / "config.yaml",
        artifact_path="artifacts",
    )
    tracker.log_artifact.assert_any_call(
        tmp_path / "runs" / "cli-wiring-run" / "meta.json",
        artifact_path="artifacts",
    )
    tracker.end_run.assert_called_once_with()
    trainer_ctor.assert_called_once_with(
        config,
        run_dir=tmp_path / "runs" / "cli-wiring-run",
        tracker=tracker,
    )


def test_cli_train_with_mlflow_end_to_end(tmp_path: Path) -> None:
    import mlflow

    root_dir = tmp_path / "runs"
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_root = tmp_path / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    experiment_name = "cli-e2e-mlflow"

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())

    payload = _minimal_config(root_dir)
    payload["run"] = {"name": "cli-mlflow-e2e"}
    payload["trainer"] = {
        "max_steps": 4,
        "warmup_steps": 0,
        "log_every_steps": 2,
        "eval_every_steps": 2,
        "micro_batch_size": 1,
        "grad_accum_steps": 1,
    }
    payload["mlflow"] = {
        "enabled": True,
        "tracking_uri": tracking_uri,
        "experiment": experiment_name,
        "run_name": "cli-e2e-run",
    }
    payload["output"] = {
        "root_dir": str(root_dir),
        "save_config_copy": True,
        "save_meta_json": True,
    }
    config_path = _write_config(tmp_path, payload)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmtrain",
            "train",
            "--config",
            str(config_path),
            "--run-id",
            "cli-mlflow-e2e",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    summary = json.loads(result.stdout)
    assert summary["run_id"] == "cli-mlflow-e2e"

    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'cli-e2e-run'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    assert runs, "Expected at least one MLflow run created by CLI training"
    run = runs[0]

    assert run.data.params["run.name"] == "cli-mlflow-e2e"
    assert "train/loss" in run.data.metrics
    assert "train/lr" in run.data.metrics
    assert "val/loss" in run.data.metrics

    artifact_paths = [
        item.path for item in client.list_artifacts(run.info.run_id, path="artifacts")
    ]
    assert "artifacts/config.yaml" in artifact_paths
    assert "artifacts/meta.json" in artifact_paths
