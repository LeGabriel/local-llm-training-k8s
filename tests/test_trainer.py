"""Tests for the single-process Trainer and TrainResult."""

from __future__ import annotations

import logging
import math

import pytest

from llmtrain.config.schemas import RunConfig
from llmtrain.training import Trainer, TrainResult


def _minimal_config() -> RunConfig:
    payload = {
        "schema_version": 1,
        "run": {"name": "trainer-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {
            "max_steps": 5,
            "warmup_steps": 0,
            "micro_batch_size": 1,
            "grad_accum_steps": 1,
        },
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }
    return RunConfig.model_validate(payload)


def test_trainer_fit_returns_train_result_with_finite_loss() -> None:
    cfg = _minimal_config()
    trainer = Trainer(cfg)
    result = trainer.fit()
    assert isinstance(result, TrainResult)
    assert result.final_step == cfg.trainer.max_steps
    assert math.isfinite(result.final_loss)
    assert result.total_time >= 0.0
    assert result.peak_memory == 0.0


def test_trainer_fit_full_loop_step_count_and_loss_decreases() -> None:
    payload = {
        "schema_version": 1,
        "run": {"name": "trainer-loop-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {
            "max_steps": 10,
            "micro_batch_size": 1,
            "grad_accum_steps": 2,
            "warmup_steps": 0,
        },
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }
    cfg = RunConfig.model_validate(payload)
    trainer = Trainer(cfg)
    result = trainer.fit()
    assert result.final_step == 10
    assert result.first_step_loss is not None
    assert result.final_loss < result.first_step_loss


def test_trainer_evaluate_returns_finite_metrics() -> None:
    cfg = _minimal_config()
    trainer = Trainer(cfg)

    metrics = trainer._evaluate()
    assert metrics is not None
    assert metrics
    assert all(math.isfinite(value) for value in metrics.values())


def _scheduler_config(
    max_steps: int = 10,
    warmup_steps: int = 4,
    lr: float = 1.0,
) -> RunConfig:
    payload = {
        "schema_version": 1,
        "run": {"name": "scheduler-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "lr": lr,
            "micro_batch_size": 1,
            "grad_accum_steps": 1,
        },
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }
    return RunConfig.model_validate(payload)


def test_lr_scheduler_step_zero() -> None:
    """LR at step 0 (before any scheduler.step()) is 0 during warmup."""
    cfg = _scheduler_config(max_steps=10, warmup_steps=4, lr=1.0)
    trainer = Trainer(cfg)
    lr: float = float(trainer.scheduler.get_last_lr()[0])
    assert lr == 0.0


def test_lr_scheduler_mid_warmup() -> None:
    """LR at step 2 with warmup_steps=4 is 0.5 (linear warmup)."""
    cfg = _scheduler_config(max_steps=10, warmup_steps=4, lr=1.0)
    trainer = Trainer(cfg)
    trainer.fit(max_steps_override=2)  # run 2 steps so scheduler step=2
    lr: float = float(trainer.scheduler.get_last_lr()[0])
    assert abs(lr - 0.5) < 1e-6


def test_lr_scheduler_end_warmup() -> None:
    """LR at end of warmup (step = warmup_steps) is 1.0."""
    cfg = _scheduler_config(max_steps=10, warmup_steps=4, lr=1.0)
    trainer = Trainer(cfg)
    trainer.fit(max_steps_override=4)  # run 4 steps so scheduler step=4, decay progress=0
    lr: float = float(trainer.scheduler.get_last_lr()[0])
    assert abs(lr - 1.0) < 1e-6


def test_lr_scheduler_final_step() -> None:
    """LR at final step (step = max_steps) is 0 (cosine decay)."""
    cfg = _scheduler_config(max_steps=10, warmup_steps=4, lr=1.0)
    trainer = Trainer(cfg)
    trainer.fit()
    lr: float = float(trainer.scheduler.get_last_lr()[0])
    assert lr == 0.0


# ---------------------------------------------------------------------------
# Metric logging tests
# ---------------------------------------------------------------------------


def test_metric_logging_emits_structured_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Structured log is emitted every log_every_steps with required fields."""
    payload = {
        "schema_version": 1,
        "run": {"name": "metric-log-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {
            "max_steps": 6,
            "log_every_steps": 2,
            "warmup_steps": 0,
            "micro_batch_size": 1,
            "grad_accum_steps": 1,
        },
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }
    cfg = RunConfig.model_validate(payload)
    trainer = Trainer(cfg)

    with caplog.at_level(logging.INFO, logger="llmtrain.training.trainer"):
        trainer.fit()

    # Filter only the structured step logs (exclude eval logs).
    step_logs = [
        r.message for r in caplog.records if "step=" in r.message and "val_step=" not in r.message
    ]

    # max_steps=6, log_every=2 → logs at steps 2, 4, 6.
    assert len(step_logs) == 3

    for msg in step_logs:
        assert "loss=" in msg
        assert "lr=" in msg
        assert "tokens_per_sec=" in msg
        assert "step_time=" in msg


def test_metric_logging_last_step_always_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Final step is always logged even if not a multiple of log_every_steps."""
    payload = {
        "schema_version": 1,
        "run": {"name": "metric-log-final-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {
            "max_steps": 5,
            "log_every_steps": 3,
            "warmup_steps": 0,
            "micro_batch_size": 1,
            "grad_accum_steps": 1,
        },
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }
    cfg = RunConfig.model_validate(payload)
    trainer = Trainer(cfg)

    with caplog.at_level(logging.INFO, logger="llmtrain.training.trainer"):
        trainer.fit()

    step_logs = [
        r.message for r in caplog.records if "step=" in r.message and "val_step=" not in r.message
    ]

    # Expect logs at step 3 (3 % 3 == 0) and step 5 (final step).
    assert len(step_logs) == 2
    assert "step=5/5" in step_logs[-1]


def test_metric_logging_tokens_per_sec_positive(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """tokens_per_sec is a positive number in emitted logs."""
    payload = {
        "schema_version": 1,
        "run": {"name": "metric-log-tps-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {
            "max_steps": 4,
            "log_every_steps": 4,
            "warmup_steps": 0,
            "micro_batch_size": 1,
            "grad_accum_steps": 1,
        },
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }
    cfg = RunConfig.model_validate(payload)
    trainer = Trainer(cfg)

    with caplog.at_level(logging.INFO, logger="llmtrain.training.trainer"):
        trainer.fit()

    step_logs = [
        r.message for r in caplog.records if "step=" in r.message and "val_step=" not in r.message
    ]
    assert len(step_logs) == 1

    # Extract tokens_per_sec value and verify it is positive.
    msg = step_logs[0]
    # Format: "... tokens_per_sec=1234.5 ..."
    for part in msg.split():
        if part.startswith("tokens_per_sec="):
            tps = float(part.split("=")[1])
            assert tps > 0.0
            break
    else:
        pytest.fail("tokens_per_sec field not found in log message")


def test_eval_logging_emits_at_expected_steps(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Eval log lines are emitted at eval_every_steps and final step."""
    payload = {
        "schema_version": 1,
        "run": {"name": "eval-log-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {
            "max_steps": 5,
            "eval_every_steps": 2,
            "log_every_steps": 10,
            "warmup_steps": 0,
            "micro_batch_size": 1,
            "grad_accum_steps": 1,
        },
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }
    cfg = RunConfig.model_validate(payload)
    trainer = Trainer(cfg)

    with caplog.at_level(logging.INFO, logger="llmtrain.training.trainer"):
        trainer.fit()

    eval_logs = [r.message for r in caplog.records if "val_step=" in r.message]
    assert len(eval_logs) == 3
    assert "val_step=2/5" in eval_logs[0]
    assert "val_step=4/5" in eval_logs[1]
    assert "val_step=5/5" in eval_logs[2]
    assert "val/loss=" in eval_logs[0]
