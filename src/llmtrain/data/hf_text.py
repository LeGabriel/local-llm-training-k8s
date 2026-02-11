from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from llmtrain.config.schemas import RunConfig
from llmtrain.data.base import DataModule
from llmtrain.registry.data import register_data_module


@register_data_module("hf_text")
class HFTextDataModule(DataModule):
    """Hugging Face text dataset module (setup-only skeleton)."""

    def __init__(self) -> None:
        self._cfg: RunConfig | None = None
        self._train_dataset: Any | None = None
        self._val_dataset: Any | None = None

    def setup(self, cfg: RunConfig, tokenizer: Any | None = None) -> None:
        del tokenizer
        if cfg.data.dataset_name is None:
            raise ValueError("hf_text requires data.dataset_name to be configured.")

        from datasets import load_dataset  # type: ignore[import-untyped]

        self._cfg = cfg
        self._train_dataset = load_dataset(
            cfg.data.dataset_name,
            cfg.data.dataset_config,
            split=cfg.data.train_split,
            cache_dir=cfg.data.cache_dir,
        )
        self._val_dataset = load_dataset(
            cfg.data.dataset_name,
            cfg.data.dataset_config,
            split=cfg.data.val_split,
            cache_dir=cfg.data.cache_dir,
        )

    def train_dataloader(self) -> DataLoader:
        raise RuntimeError("HFTextDataModule dataloaders are not implemented yet.")

    def val_dataloader(self) -> DataLoader | None:
        raise RuntimeError("HFTextDataModule dataloaders are not implemented yet.")
