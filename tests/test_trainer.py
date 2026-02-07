"""Tests for the single-process Trainer and TrainResult."""

from __future__ import annotations

import math

from llmtrain.config.schemas import RunConfig
from llmtrain.training import Trainer, TrainResult


def _minimal_config() -> RunConfig:
    payload = {
        "schema_version": 1,
        "run": {"name": "trainer-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {"max_steps": 5, "warmup_steps": 0},
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
