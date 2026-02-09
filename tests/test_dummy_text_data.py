"""Tests for the dummy text data module."""

from __future__ import annotations

import torch

from llmtrain.config.schemas import RunConfig
from llmtrain.data.dummy_text import DummyTextDataModule


def _config() -> RunConfig:
    payload = {
        "schema_version": 1,
        "run": {"name": "dummy-text-data-test"},
        "model": {"name": "dummy_gpt", "block_size": 8},
        "data": {"name": "dummy_text"},
        "trainer": {
            "max_steps": 5,
            "micro_batch_size": 2,
            "grad_accum_steps": 1,
            "warmup_steps": 0,
        },
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }
    return RunConfig.model_validate(payload)


def test_dummy_text_val_dataloader_shapes() -> None:
    cfg = _config()
    data = DummyTextDataModule()
    data.setup(cfg)

    val_loader = data.val_dataloader()
    assert val_loader is not None

    batch = next(iter(val_loader))
    assert set(batch.keys()) == {"input_ids", "labels", "attention_mask"}
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)
    assert batch["input_ids"].dtype == torch.long
