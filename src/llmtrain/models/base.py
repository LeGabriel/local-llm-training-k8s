from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from llmtrain.config.schemas import RunConfig


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    @abstractmethod
    def build_model(self, cfg: RunConfig) -> nn.Module:
        """Return the model instance constructed for the given run config."""

    @abstractmethod
    def build_tokenizer(self, cfg: RunConfig) -> Any | None:
        """Return the tokenizer for the run config, or None for no tokenizer."""

    @abstractmethod
    def compute_loss(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute a scalar loss and metric values for a single batch."""
