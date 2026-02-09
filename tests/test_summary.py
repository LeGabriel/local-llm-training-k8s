from __future__ import annotations

from llmtrain.config.schemas import RunConfig
from llmtrain.training.trainer import TrainResult
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


def test_format_run_summary_includes_train_result_json() -> None:
    config = _minimal_config()
    result = TrainResult(
        final_step=10,
        final_loss=1.234,
        final_val_loss=None,
        total_time=5.67,
        peak_memory=0.0,
        first_step_loss=3.456,
    )

    json_summary = format_run_summary(
        config=config,
        run_id="run-train",
        run_dir="runs/run-train",
        json_output=True,
        train_result=result,
    )

    assert isinstance(json_summary, dict)
    assert "training" in json_summary
    training = json_summary["training"]
    assert training["final_step"] == 10
    assert training["final_loss"] == 1.234
    assert training["first_step_loss"] == 3.456
    assert training["total_time"] == 5.67
    assert training["peak_memory"] == 0.0


def test_format_run_summary_includes_train_result_text() -> None:
    config = _minimal_config()
    result = TrainResult(
        final_step=10,
        final_loss=1.234,
        final_val_loss=None,
        total_time=5.67,
        peak_memory=0.0,
        first_step_loss=3.456,
    )

    text_summary = format_run_summary(
        config=config,
        run_id="run-train",
        run_dir="runs/run-train",
        json_output=False,
        train_result=result,
    )

    assert isinstance(text_summary, str)
    assert "Training:" in text_summary
    assert "final_step=10" in text_summary
    assert "final_loss=1.2340" in text_summary
    assert "total_time=5.67s" in text_summary


# ---------------------------------------------------------------------------
# resumed_from tests
# ---------------------------------------------------------------------------


def test_format_run_summary_includes_resumed_from_json() -> None:
    config = _minimal_config()
    result = TrainResult(
        final_step=20,
        final_loss=0.5,
        final_val_loss=None,
        total_time=10.0,
        peak_memory=0.0,
        first_step_loss=2.0,
        resumed_from_step=10,
    )

    json_summary = format_run_summary(
        config=config,
        run_id="run-resume",
        run_dir="runs/run-resume",
        json_output=True,
        train_result=result,
        resumed_from="some-run-id",
    )

    assert isinstance(json_summary, dict)
    assert json_summary["resumed_from"] == "some-run-id"
    assert "training" in json_summary
    assert json_summary["training"]["resumed_from_step"] == 10


def test_format_run_summary_includes_resumed_from_text() -> None:
    config = _minimal_config()
    result = TrainResult(
        final_step=20,
        final_loss=0.5,
        final_val_loss=None,
        total_time=10.0,
        peak_memory=0.0,
        first_step_loss=2.0,
        resumed_from_step=10,
    )

    text_summary = format_run_summary(
        config=config,
        run_id="run-resume",
        run_dir="runs/run-resume",
        json_output=False,
        train_result=result,
        resumed_from="some-run-id",
    )

    assert isinstance(text_summary, str)
    assert "Resumed from: some-run-id" in text_summary
    assert "resumed_from_step=10" in text_summary


def test_format_run_summary_omits_resumed_from_when_none() -> None:
    config = _minimal_config()
    result = TrainResult(
        final_step=10,
        final_loss=1.0,
        final_val_loss=None,
        total_time=5.0,
        peak_memory=0.0,
        first_step_loss=3.0,
    )

    json_summary = format_run_summary(
        config=config,
        run_id="run-no-resume",
        run_dir="runs/run-no-resume",
        json_output=True,
        train_result=result,
    )

    assert isinstance(json_summary, dict)
    assert "resumed_from" not in json_summary
    assert "resumed_from_step" not in json_summary["training"]

    text_summary = format_run_summary(
        config=config,
        run_id="run-no-resume",
        run_dir="runs/run-no-resume",
        json_output=False,
        train_result=result,
    )

    assert isinstance(text_summary, str)
    assert "Resumed from:" not in text_summary
    assert "resumed_from_step" not in text_summary
