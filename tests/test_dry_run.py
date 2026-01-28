from __future__ import annotations

from llmtrain.config.schemas import RunConfig
from llmtrain.training.dry_run import DEFAULT_DRY_RUN_STEPS, run_dry_run


def _minimal_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run": {"name": "dry-run-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {"max_steps": 5, "warmup_steps": 0},
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }


def test_dry_run_executes_default_steps() -> None:
    cfg = RunConfig.model_validate(_minimal_payload())
    result = run_dry_run(cfg)
    assert result.steps_executed == DEFAULT_DRY_RUN_STEPS
    assert result.resolved_model_adapter == "dummy_gpt"
    assert result.resolved_data_module == "dummy_text"


def test_dry_run_respects_extra_step_override() -> None:
    payload = _minimal_payload()
    payload["trainer"] = {"max_steps": 5, "warmup_steps": 0, "extra": {"dry_run_steps": 1}}
    cfg = RunConfig.model_validate(payload)
    result = run_dry_run(cfg)
    assert result.steps_executed == 1
