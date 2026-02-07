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
    assert result.final_step == 1
    assert math.isfinite(result.final_loss)
    assert result.total_time >= 0.0
    assert result.peak_memory == 0.0
