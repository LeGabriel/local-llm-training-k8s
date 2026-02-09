from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from llmtrain.config.schemas import RunConfig
from llmtrain.data.base import DataModule
from llmtrain.registry.data import register_data_module


class _DummyTextDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        num_examples: int,
        seq_len: int,
        vocab_size: int,
        deterministic: bool,
        seed: int,
    ) -> None:
        self._num_examples = num_examples
        self._seq_len = seq_len
        self._vocab_size = vocab_size
        self._deterministic = deterministic
        self._seed = seed

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = None
        if self._deterministic:
            generator = torch.Generator()
            generator.manual_seed(self._seed + index)
        input_ids = torch.randint(
            0,
            self._vocab_size,
            (self._seq_len,),
            dtype=torch.long,
            generator=generator,
        )
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


@register_data_module("dummy_text")
class DummyTextDataModule(DataModule):
    """Synthetic text data for dry-run smoke tests."""

    def __init__(self) -> None:
        self._cfg: RunConfig | None = None
        self._train_dataset: Dataset[dict[str, torch.Tensor]] | None = None
        self._val_dataset: Dataset[dict[str, torch.Tensor]] | None = None

    def setup(self, cfg: RunConfig, tokenizer: Any | None = None) -> None:
        del tokenizer
        self._cfg = cfg
        vocab_size = cfg.model.vocab_size or 128
        # Keep synthetic batches tiny so unit tests are fast and stable.
        seq_len = max(1, min(cfg.model.block_size, 8))
        max_steps = cfg.trainer.max_steps or 1
        micro_batch_size = cfg.trainer.micro_batch_size or 1
        requested = max_steps * micro_batch_size
        num_examples = max(1, min(requested, 128))
        self._train_dataset = _DummyTextDataset(
            num_examples=num_examples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            deterministic=cfg.run.deterministic,
            seed=cfg.run.seed,
        )
        val_examples = max(1, min(num_examples // 5, 32))
        self._val_dataset = _DummyTextDataset(
            num_examples=val_examples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            deterministic=cfg.run.deterministic,
            seed=cfg.run.seed + 1000,
        )

    def train_dataloader(self) -> DataLoader:
        if self._cfg is None or self._train_dataset is None:
            raise RuntimeError("setup must be called before train_dataloader")
        micro_batch_size = self._cfg.trainer.micro_batch_size or 1
        # This is a synthetic dataset; multiprocessing adds massive overhead on macOS
        # (worker spawn + importing torch) and provides no benefit.
        num_workers = 0
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
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._cfg is None:
            raise RuntimeError("setup must be called before val_dataloader")
        if self._val_dataset is None:
            return None
        micro_batch_size = self._cfg.trainer.micro_batch_size or 1
        num_workers = 0
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
        )
