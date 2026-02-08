from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

import llmtrain


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
