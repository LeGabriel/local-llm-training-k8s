from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llmtrain.config.loader import ConfigLoadError, load_and_validate_config


def _minimal_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run": {"name": "test-run"},
        "model": {"name": "tiny-model"},
        "data": {"name": "toy-data"},
        "trainer": {},
        "ddp": {},
        "mlflow": {},
        "logging": {},
        "output": {},
    }


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_load_and_validate_materializes_defaults(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _minimal_config())

    config, raw_path, resolved_path = load_and_validate_config(str(config_path))

    assert raw_path == str(config_path)
    assert resolved_path == config_path.resolve()
    assert config.run.seed == 1337
    assert config.model.block_size == 256
    assert config.model.extra == {}
    assert config.data.cache_dir == ".cache/datasets"
    assert config.data.extra == {}
    assert config.trainer.extra == {}
    assert config.logging.json_output is True
    assert config.output.root_dir == "runs"


def test_load_and_validate_rejects_invalid_config(tmp_path: Path) -> None:
    payload = _minimal_config()
    payload["model"] = {"name": "tiny-model", "d_model": 384, "n_heads": 7}
    payload["trainer"] = {"max_steps": 10, "warmup_steps": 20}
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigLoadError):
        load_and_validate_config(str(config_path))


def test_load_and_validate_rejects_extra_fields(tmp_path: Path) -> None:
    payload = _minimal_config()
    payload["run"] = {"name": "test-run", "extra": "nope"}
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigLoadError):
        load_and_validate_config(str(config_path))


def test_load_and_validate_accepts_plugin_extras(tmp_path: Path) -> None:
    payload = _minimal_config()
    payload["model"] = {"name": "tiny-model", "extra": {"adapter": "dummy"}}
    payload["data"] = {"name": "toy-data", "extra": {"dataset": "synthetic"}}
    payload["trainer"] = {"extra": {"gradient_clip": 0.9}}
    config_path = _write_config(tmp_path, payload)

    config, _, _ = load_and_validate_config(str(config_path))

    assert config.model.extra["adapter"] == "dummy"
    assert config.data.extra["dataset"] == "synthetic"
    assert config.trainer.extra["gradient_clip"] == 0.9
