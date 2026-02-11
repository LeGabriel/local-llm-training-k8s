from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from llmtrain.config.schemas import RunConfig
from llmtrain.data.base import DataModule
from llmtrain.registry.data import register_data_module


@register_data_module("hf_text")
class HFTextDataModule(DataModule):
    """Hugging Face text dataset module with cached tokenized chunks."""

    def __init__(self) -> None:
        self._cfg: RunConfig | None = None
        self._train_dataset: Any | None = None
        self._val_dataset: Any | None = None

    @staticmethod
    def _collate_batch(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([row["input_ids"] for row in batch], dtype=torch.long),
            "labels": torch.tensor([row["labels"] for row in batch], dtype=torch.long),
            "attention_mask": torch.tensor(
                [row["attention_mask"] for row in batch],
                dtype=torch.long,
            ),
        }

    def setup(self, cfg: RunConfig, tokenizer: Any | None = None) -> None:
        if cfg.data.dataset_name is None:
            raise ValueError("hf_text requires data.dataset_name to be configured.")
        text_column = cfg.data.text_column
        if text_column is None:
            raise ValueError("hf_text requires data.text_column to be configured.")
        if tokenizer is None:
            raise ValueError("hf_text requires a tokenizer instance.")
        if not hasattr(tokenizer, "encode"):
            raise ValueError("hf_text tokenizer must provide an encode(text) method.")

        from datasets import Dataset, load_dataset, load_from_disk  # type: ignore[import-untyped]

        self._cfg = cfg
        self._train_dataset = self._prepare_split(
            split=cfg.data.train_split,
            cfg=cfg,
            tokenizer=tokenizer,
            text_column=text_column,
            load_dataset=load_dataset,
            load_from_disk=load_from_disk,
            dataset_cls=Dataset,
        )
        self._val_dataset = self._prepare_split(
            split=cfg.data.val_split,
            cfg=cfg,
            tokenizer=tokenizer,
            text_column=text_column,
            load_dataset=load_dataset,
            load_from_disk=load_from_disk,
            dataset_cls=Dataset,
        )

    def _prepare_split(
        self,
        *,
        split: str,
        cfg: RunConfig,
        tokenizer: Any,
        text_column: str,
        load_dataset: Any,
        load_from_disk: Any,
        dataset_cls: Any,
    ) -> Any:
        cache_path = self._processed_cache_path(cfg, split)
        if cache_path.exists():
            return load_from_disk(str(cache_path))

        raw_dataset = load_dataset(
            cfg.data.dataset_name,
            cfg.data.dataset_config,
            split=split,
            cache_dir=cfg.data.cache_dir,
        )
        processed_dataset = self._tokenize_and_chunk(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            text_column=text_column,
            block_size=cfg.model.block_size,
            dataset_cls=dataset_cls,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        processed_dataset.save_to_disk(str(cache_path))
        return processed_dataset

    def _processed_cache_path(self, cfg: RunConfig, split: str) -> Path:
        dataset_name = cfg.data.dataset_name or "unknown"
        dataset_config = cfg.data.dataset_config or "default"
        safe_name = dataset_name.replace("/", "__")
        safe_config = dataset_config.replace("/", "__")
        return (
            Path(cfg.data.cache_dir)
            / "processed"
            / f"{safe_name}__{safe_config}__b{cfg.model.block_size}__{split}"
        )

    def _tokenize_and_chunk(
        self,
        *,
        raw_dataset: Any,
        tokenizer: Any,
        text_column: str,
        block_size: int,
        dataset_cls: Any,
    ) -> Any:
        chunk_size = block_size + 1
        buffer: list[int] = []
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []

        for row in raw_dataset:
            text_value = row.get(text_column)
            if text_value is None:
                continue
            encoded = tokenizer.encode(str(text_value))
            if not isinstance(encoded, list):
                raise ValueError("Tokenizer encode output must be a list of token ids.")
            buffer.extend(int(token_id) for token_id in encoded)
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                del buffer[:chunk_size]
                input_ids.append(chunk[:-1])
                labels.append(chunk[1:])
                attention_mask.append([1] * block_size)

        return dataset_cls.from_dict(
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
        )

    def train_dataloader(self) -> DataLoader:
        if self._cfg is None or self._train_dataset is None:
            raise RuntimeError("setup must be called before train_dataloader")
        micro_batch_size = self._cfg.trainer.micro_batch_size or 1
        num_workers = self._cfg.data.num_workers
        use_ddp = False
        world_size = self._cfg.ddp.world_size or 1
        rank = self._cfg.ddp.rank or 0
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            use_ddp = world_size > 1
        elif world_size > 1:
            use_ddp = True
        sampler: DistributedSampler | None = None
        if use_ddp:
            sampler = DistributedSampler(
                self._train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=not self._cfg.run.deterministic,
                seed=self._cfg.run.seed,
            )
        return DataLoader(
            self._train_dataset,
            batch_size=micro_batch_size,
            num_workers=num_workers,
            shuffle=sampler is None and not self._cfg.run.deterministic,
            sampler=sampler,
            collate_fn=self._collate_batch,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._cfg is None:
            raise RuntimeError("setup must be called before val_dataloader")
        if self._val_dataset is None:
            return None
        micro_batch_size = self._cfg.trainer.micro_batch_size or 1
        num_workers = self._cfg.data.num_workers
        use_ddp = False
        world_size = self._cfg.ddp.world_size or 1
        rank = self._cfg.ddp.rank or 0
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            use_ddp = world_size > 1
        elif world_size > 1:
            use_ddp = True
        sampler: DistributedSampler | None = None
        if use_ddp:
            sampler = DistributedSampler(
                self._val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=self._cfg.run.seed,
            )
        return DataLoader(
            self._val_dataset,
            batch_size=micro_batch_size,
            num_workers=num_workers,
            shuffle=False,
            sampler=sampler,
            collate_fn=self._collate_batch,
        )
