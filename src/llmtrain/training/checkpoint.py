"""Checkpoint management: save, prune, and (later) load/restore."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch

from llmtrain.config.schemas import RunConfig

logger = logging.getLogger(__name__)


class CheckpointPayload(TypedDict):
    """Typed dict describing what lives inside a checkpoint file."""

    step: int
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any]
    rng_states: dict[str, Any]
    config: dict[str, Any]


class CheckpointManager:
    """Manages writing and pruning of training checkpoints on disk."""

    def __init__(self, checkpoint_dir: Path, keep_last_k: int = 3) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._keep_last_k = keep_last_k
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        config: RunConfig,
    ) -> Path:
        """Persist a full checkpoint to disk and prune old ones.

        Returns the path to the newly saved checkpoint file.
        """
        rng_states: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["cuda"] = torch.cuda.get_rng_state_all()

        payload: CheckpointPayload = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_states": rng_states,
            "config": config.model_dump(),
        }

        path = self._checkpoint_dir / f"step_{step:06d}.pt"
        torch.save(payload, path)
        logger.info("checkpoint: saved step %d to %s", step, path)

        self._prune_old()
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_old(self) -> None:
        """Delete oldest checkpoints beyond *keep_last_k*."""
        existing = sorted(
            self._checkpoint_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        to_remove = existing[: max(0, len(existing) - self._keep_last_k)]
        for path in to_remove:
            path.unlink()
            logger.debug("checkpoint: pruned %s", path)
