"""Tests for the Hugging Face text data module skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from torch.utils.data.distributed import DistributedSampler

from llmtrain.config.schemas import RunConfig
from llmtrain.data.hf_text import HFTextDataModule
from llmtrain.models.gpt import GPTAdapter
from llmtrain.registry import initialize_registries
from llmtrain.registry.data import get_data_module


def _config(
    cache_dir: Path, *, ddp_world_size: int | None = None, ddp_rank: int | None = None
) -> RunConfig:
    payload = {
        "schema_version": 1,
        "run": {"name": "hf-text-data-test"},
        "model": {"name": "dummy_gpt", "block_size": 8},
        "data": {
            "name": "hf_text",
            "cache_dir": str(cache_dir),
            "num_workers": 0,
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
        "ddp": {
            "world_size": ddp_world_size,
            "rank": ddp_rank,
        },
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

    tokenizer = GPTAdapter().build_tokenizer(cfg)
    assert tokenizer is not None

    try:
        data.setup(cfg, tokenizer=tokenizer)
    except Exception as exc:
        if _is_transient_hf_error(exc):
            pytest.skip(f"Hugging Face dataset fetch unavailable in this environment: {exc}")
        raise

    assert data._train_dataset is not None
    assert data._val_dataset is not None
    assert len(data._train_dataset) > 0
    assert len(data._val_dataset) > 0


class _ToyTokenizer:
    def encode(self, text: str) -> list[int]:
        # Keep ids positive and deterministic.
        return [((ord(ch) % 31) + 1) for ch in text if not ch.isspace()]


def test_hf_text_setup_preprocesses_shapes_and_dtypes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datasets import Dataset  # type: ignore[import-untyped]

    initialize_registries()
    data = HFTextDataModule()
    cfg = _config(tmp_path / "hf_cache")
    tokenizer = _ToyTokenizer()

    dataset = Dataset.from_dict(
        {
            "text": ["alpha beta", "gamma delta epsilon", "zeta eta theta"],
            "metadata": [1, 2, 3],
        }
    )

    def _fake_load_dataset(*args: object, **kwargs: object) -> Dataset:
        del args, kwargs
        return dataset

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    data.setup(cfg, tokenizer=tokenizer)

    assert data._train_dataset is not None
    assert data._val_dataset is not None
    assert len(data._train_dataset) > 0
    assert len(data._val_dataset) > 0

    train_sample = data._train_dataset[0]
    assert set(train_sample.keys()) == {"input_ids", "labels", "attention_mask"}
    assert len(train_sample["input_ids"]) == cfg.model.block_size
    assert len(train_sample["labels"]) == cfg.model.block_size
    assert len(train_sample["attention_mask"]) == cfg.model.block_size
    assert all(isinstance(token_id, int) for token_id in train_sample["input_ids"])
    assert all(isinstance(token_id, int) for token_id in train_sample["labels"])
    assert all(mask_value == 1 for mask_value in train_sample["attention_mask"])

    # Structural next-token check: labels are a one-step right shift of inputs.
    assert train_sample["labels"][:-1] == train_sample["input_ids"][1:]

    batch = {
        "input_ids": torch.tensor([train_sample["input_ids"]], dtype=torch.long),
        "labels": torch.tensor([train_sample["labels"]], dtype=torch.long),
        "attention_mask": torch.tensor([train_sample["attention_mask"]], dtype=torch.long),
    }
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["input_ids"].shape == (1, cfg.model.block_size)
    assert batch["labels"].shape == (1, cfg.model.block_size)
    assert batch["attention_mask"].shape == (1, cfg.model.block_size)


def test_hf_text_setup_reuses_cached_processed_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datasets import Dataset  # type: ignore[import-untyped]

    initialize_registries()
    cfg = _config(tmp_path / "hf_cache")
    tokenizer = _ToyTokenizer()

    source_dataset = Dataset.from_dict({"text": ["one two", "three four five"]})

    def _fake_load_dataset(*args: object, **kwargs: object) -> Dataset:
        del args, kwargs
        return source_dataset

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    data = HFTextDataModule()
    data.setup(cfg, tokenizer=tokenizer)
    assert data._train_dataset is not None
    first_train_len = len(data._train_dataset)
    assert first_train_len > 0

    def _unexpected_load_dataset(*args: object, **kwargs: object) -> Dataset:
        del args, kwargs
        raise AssertionError("load_dataset should not be called when processed cache exists.")

    monkeypatch.setattr("datasets.load_dataset", _unexpected_load_dataset)

    cached_data = HFTextDataModule()
    cached_data.setup(cfg, tokenizer=tokenizer)
    assert cached_data._train_dataset is not None
    assert len(cached_data._train_dataset) == first_train_len


def test_hf_text_dataloaders_return_expected_batch_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datasets import Dataset  # type: ignore[import-untyped]

    cfg = _config(tmp_path / "hf_cache")
    tokenizer = _ToyTokenizer()
    data = HFTextDataModule()

    dataset = Dataset.from_dict(
        {
            "text": [
                "alpha beta gamma delta epsilon zeta eta theta iota kappa",
                "lambda mu nu xi omicron pi rho sigma tau upsilon",
                "phi chi psi omega alpha beta gamma delta epsilon zeta",
            ]
        }
    )

    def _fake_load_dataset(*args: object, **kwargs: object) -> Dataset:
        del args, kwargs
        return dataset

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)
    data.setup(cfg, tokenizer=tokenizer)

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    assert val_loader is not None

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert set(train_batch.keys()) == {"input_ids", "labels", "attention_mask"}
    assert train_batch["input_ids"].shape == (cfg.trainer.micro_batch_size, cfg.model.block_size)
    assert train_batch["labels"].shape == (cfg.trainer.micro_batch_size, cfg.model.block_size)
    assert train_batch["attention_mask"].shape == (
        cfg.trainer.micro_batch_size,
        cfg.model.block_size,
    )
    assert train_batch["input_ids"].dtype == torch.long
    assert train_batch["labels"].dtype == torch.long
    assert train_batch["attention_mask"].dtype == torch.long

    assert val_batch["input_ids"].shape == (cfg.trainer.micro_batch_size, cfg.model.block_size)
    assert val_batch["labels"].shape == (cfg.trainer.micro_batch_size, cfg.model.block_size)
    assert val_batch["attention_mask"].shape == (cfg.trainer.micro_batch_size, cfg.model.block_size)


def test_hf_text_dataloaders_use_ddp_sampler_from_config_hints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datasets import Dataset  # type: ignore[import-untyped]

    cfg = _config(tmp_path / "hf_cache", ddp_world_size=4, ddp_rank=2)
    tokenizer = _ToyTokenizer()
    data = HFTextDataModule()

    dataset = Dataset.from_dict(
        {
            "text": [
                "alpha beta gamma delta epsilon zeta eta theta iota kappa",
                "lambda mu nu xi omicron pi rho sigma tau upsilon",
                "phi chi psi omega alpha beta gamma delta epsilon zeta",
                "eta theta iota kappa lambda mu nu xi omicron pi",
            ]
        }
    )

    def _fake_load_dataset(*args: object, **kwargs: object) -> Dataset:
        del args, kwargs
        return dataset

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)
    monkeypatch.setattr("llmtrain.data.hf_text.dist.is_available", lambda: True)
    monkeypatch.setattr("llmtrain.data.hf_text.dist.is_initialized", lambda: False)

    data.setup(cfg, tokenizer=tokenizer)
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    assert val_loader is not None

    assert isinstance(train_loader.sampler, DistributedSampler)
    assert isinstance(val_loader.sampler, DistributedSampler)
    assert train_loader.sampler.num_replicas == 4
    assert train_loader.sampler.rank == 2
    assert val_loader.sampler.num_replicas == 4
    assert val_loader.sampler.rank == 2
