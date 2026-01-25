from __future__ import annotations

import json
from pathlib import Path

import yaml

from llmtrain.config.loader import load_and_validate_config
from llmtrain.utils.metadata import generate_meta, write_meta_json
from llmtrain.utils.run_dir import create_run_directory, write_resolved_config


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _minimal_config(root_dir: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run": {"name": "integration-test"},
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


def test_full_run_flow_persists_outputs(tmp_path: Path) -> None:
    root_dir = tmp_path / "runs"
    config_path = _write_config(tmp_path, _minimal_config(root_dir))

    config, raw_path, resolved_path = load_and_validate_config(str(config_path))

    run_id = "integration-run"
    run_dir = create_run_directory(config.output.root_dir, run_id)

    config_out = write_resolved_config(run_dir, config)
    meta = generate_meta(
        run_id=run_id,
        run_name=config.run.name,
        config_path=raw_path,
        resolved_config_path=str(resolved_path),
    )
    meta_out = write_meta_json(run_dir, meta)

    assert run_dir == root_dir / run_id
    assert (run_dir / "logs").exists()
    assert config_out.exists()
    assert meta_out.exists()

    persisted_config = yaml.safe_load(config_out.read_text(encoding="utf-8"))
    assert persisted_config["run"]["name"] == "integration-test"
    assert persisted_config["output"]["root_dir"] == str(root_dir)

    persisted_meta = json.loads(meta_out.read_text(encoding="utf-8"))
    assert persisted_meta["run_id"] == run_id
    assert persisted_meta["run_name"] == "integration-test"
    assert persisted_meta["config_path"] == str(config_path)
    assert persisted_meta["resolved_config_path"] == str(config_path.resolve())
