from __future__ import annotations

import json
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
        "model": {"name": "tiny-model"},
        "data": {"name": "toy-data"},
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
    config_path = _write_config(tmp_path, _minimal_config(root_dir))

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

    run_dir = root_dir / "unit-test-run"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "meta.json").exists()
