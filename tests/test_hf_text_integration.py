from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_preset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in preset {path}")
    return cast(dict[str, Any], payload)


def _write_config(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.slow
def test_train_gpt_wikitext_real_data_finite_and_improving_loss(tmp_path: Path) -> None:
    root_dir = _repo_root()
    preset_path = root_dir / "configs" / "presets" / "gpt_wikitext.yaml"
    payload = _load_preset(preset_path)

    payload["output"] = {
        "root_dir": str(tmp_path / "runs"),
        "save_config_copy": True,
        "save_meta_json": True,
    }
    payload["data"] = {
        **cast(dict[str, Any], payload["data"]),
        "cache_dir": str(tmp_path / "cache"),
    }
    payload["mlflow"] = {
        **cast(dict[str, Any], payload["mlflow"]),
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "run_name": "test-gpt-wikitext-slow",
    }

    config_path = _write_config(tmp_path / "gpt_wikitext_test.yaml", payload)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmtrain",
            "train",
            "--config",
            str(config_path),
            "--run-id",
            "gpt-wikitext-integration",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    summary = json.loads(result.stdout)
    training = summary["training"]

    first_step_loss = training["first_step_loss"]
    final_loss = training["final_loss"]
    assert first_step_loss is not None
    assert math.isfinite(first_step_loss)
    assert math.isfinite(final_loss)
    assert final_loss < first_step_loss
