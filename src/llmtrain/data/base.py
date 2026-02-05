from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import DataLoader

from llmtrain.config.schemas import RunConfig


class DataModule(ABC):
    """Abstract base class for data modules."""

    @abstractmethod
    def setup(self, cfg: RunConfig, tokenizer: Any | None = None) -> None:
        """Prepare datasets and internal state needed for dataloaders."""

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""

    @abstractmethod
    def val_dataloader(self) -> DataLoader | None:
        """Return the validation dataloader, or None if not provided."""
