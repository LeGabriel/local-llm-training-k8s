"""Tests for the Hugging Face text data module skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from llmtrain.config.schemas import RunConfig
from llmtrain.data.hf_text import HFTextDataModule
from llmtrain.registry import initialize_registries
from llmtrain.registry.data import get_data_module


def _config(cache_dir: Path) -> RunConfig:
    payload = {
        "schema_version": 1,
        "run": {"name": "hf-text-data-test"},
        "model": {"name": "dummy_gpt", "block_size": 8},
        "data": {
            "name": "hf_text",
            "cache_dir": str(cache_dir),
            "dataset_name": "wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "train_split": "train",
            "val_split": "validation",
            "text_column": "text",
        },
        "trainer": {
            "max_steps": 1,
            "micro_batch_size": 1,
            "grad_accum_steps": 1,
            "warmup_steps": 0,
        },
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }
    return RunConfig.model_validate(payload)


def _is_transient_hf_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "couldn't reach",
            "timed out",
            "temporary failure",
            "connection",
            "network",
            "max retries exceeded",
        )
    )


def test_hf_text_setup_loads_wikitext2_raw_v1(tmp_path: Path) -> None:
    initialize_registries()
    data_cls = get_data_module("hf_text")
    assert data_cls is HFTextDataModule
    data = cast(HFTextDataModule, data_cls())
    cfg = _config(tmp_path / "hf_cache")

    try:
        data.setup(cfg)
    except Exception as exc:
        if _is_transient_hf_error(exc):
            pytest.skip(f"Hugging Face dataset fetch unavailable in this environment: {exc}")
        raise

    assert data._train_dataset is not None
    assert data._val_dataset is not None
    assert len(data._train_dataset) > 0
    assert len(data._val_dataset) > 0
