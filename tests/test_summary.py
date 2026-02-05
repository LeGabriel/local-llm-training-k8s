from __future__ import annotations

from llmtrain.config.schemas import RunConfig
from llmtrain.utils.summary import format_run_summary


def _minimal_config() -> RunConfig:
    payload = {
        "schema_version": 1,
        "run": {"name": "summary-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {},
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }
    return RunConfig.model_validate(payload)


def test_format_run_summary_includes_v03_fields() -> None:
    config = _minimal_config()

    json_summary = format_run_summary(
        config=config,
        run_id="run-123",
        run_dir="runs/run-123",
        json_output=True,
        resolved_model_adapter="dummy_gpt",
        resolved_data_module="dummy_text",
        dry_run_steps_executed=5,
    )

    assert isinstance(json_summary, dict)
    assert json_summary["resolved_model_adapter"] == "dummy_gpt"
    assert json_summary["resolved_data_module"] == "dummy_text"
    assert json_summary["dry_run_steps_executed"] == 5

    text_summary = format_run_summary(
        config=config,
        run_id="run-123",
        run_dir="runs/run-123",
        json_output=False,
        resolved_model_adapter="dummy_gpt",
        resolved_data_module="dummy_text",
        dry_run_steps_executed=5,
    )

    assert isinstance(text_summary, str)
    assert "Dry run:" in text_summary
    assert "resolved_model_adapter=dummy_gpt" in text_summary
    assert "resolved_data_module=dummy_text" in text_summary
    assert "steps_executed=5" in text_summary
