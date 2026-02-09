from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

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
