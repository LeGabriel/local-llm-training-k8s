"""Single-process training loop: optimizer steps, gradient accumulation, LR schedule."""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from llmtrain.config.schemas import RunConfig
from llmtrain.registry import initialize_registries
from llmtrain.registry.data import get_data_module
from llmtrain.registry.models import get_model_adapter
from llmtrain.training.checkpoint import CheckpointPayload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainResult:
    """Result of a training run (single process)."""

    final_step: int
    final_loss: float
    total_time: float
    peak_memory: float
    first_step_loss: float | None = None


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class Trainer:
    """Single-process trainer: one device, step-based loop with gradient accumulation."""

    def __init__(self, cfg: RunConfig) -> None:
        self._cfg = cfg
        initialize_registries()

        adapter_cls = get_model_adapter(cfg.model.name)
        data_cls = get_data_module(cfg.data.name)
        self._adapter = adapter_cls()
        data_module = data_cls()

        self._model = self._adapter.build_model(cfg)
        tokenizer = self._adapter.build_tokenizer(cfg)
        data_module.setup(cfg, tokenizer=tokenizer)
        self._train_loader = data_module.train_dataloader()

        self._device = torch.device("cpu")
        self._model = self._model.to(self._device)

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=cfg.trainer.lr,
            weight_decay=cfg.trainer.weight_decay,
        )
        self._scheduler = self._build_scheduler(self._optimizer)

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Build LambdaLR: linear warmup 0→warmup_steps, then cosine decay to 0 by max_steps."""
        warmup_steps = self._cfg.trainer.warmup_steps
        max_steps = self._cfg.trainer.max_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step / warmup_steps) if warmup_steps > 0 else 1.0
            if step >= max_steps:
                return 0.0
            if max_steps <= warmup_steps:
                return 1.0
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Scheduler instance (for tests that assert LR curve)."""
        return self._scheduler

    def restore(self, payload: CheckpointPayload) -> int:
        """Restore model, optimizer, scheduler, and RNG states from a checkpoint payload.

        Returns the step number stored in the checkpoint (i.e. the step to
        resume *from*).
        """
        self._model.load_state_dict(payload["model_state_dict"])
        self._optimizer.load_state_dict(payload["optimizer_state_dict"])
        self._scheduler.load_state_dict(payload["scheduler_state_dict"])

        rng = payload["rng_states"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.random.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

        step = payload["step"]
        logger.info("trainer: restored state from step %d", step)
        return step

    def fit(self, *, max_steps_override: int | None = None) -> TrainResult:
        """Run up to max_steps optimizer steps with gradient accumulation; return the result."""
        self._model.train()
        max_steps = (
            max_steps_override if max_steps_override is not None else self._cfg.trainer.max_steps
        )
        grad_accum_steps = self._cfg.trainer.grad_accum_steps
        log_every = self._cfg.trainer.log_every_steps

        start_time = time.perf_counter()
        iterator = iter(self._train_loader)
        first_step_loss: float | None = None
        step_loss = 0.0

        # Metric logging state: running accumulators for the current log interval.
        interval_loss_sum = 0.0
        interval_steps = 0
        interval_tokens = 0
        interval_start = time.perf_counter()

        for step in range(1, max_steps + 1):
            self._optimizer.zero_grad()
            accumulated_loss = 0.0
            step_tokens = 0

            for _ in range(grad_accum_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(self._train_loader)
                    batch = next(iterator)

                batch = _move_batch(batch, self._device)
                step_tokens += batch["input_ids"].numel()
                loss, metrics = self._adapter.compute_loss(self._model, batch)
                scaled = loss / grad_accum_steps
                scaled.backward()
                accumulated_loss += float(metrics.get("loss", loss.item()))

            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self._cfg.trainer.max_grad_norm,
            )
            self._optimizer.step()
            self._scheduler.step()

            step_loss = accumulated_loss / grad_accum_steps
            if step == 1:
                first_step_loss = step_loss

            # Accumulate interval metrics.
            interval_loss_sum += step_loss
            interval_steps += 1
            interval_tokens += step_tokens

            # Emit structured log every log_every steps and always at the final step.
            if step % log_every == 0 or step == max_steps:
                interval_time = time.perf_counter() - interval_start
                avg_loss = interval_loss_sum / interval_steps
                avg_step_time = interval_time / interval_steps
                tokens_per_sec = interval_tokens / interval_time if interval_time > 0 else 0.0
                current_lr = float(self._scheduler.get_last_lr()[0])
                logger.info(
                    "step=%d/%d  loss=%.4f  lr=%.6e  tokens_per_sec=%.1f  step_time=%.4fs",
                    step,
                    max_steps,
                    avg_loss,
                    current_lr,
                    tokens_per_sec,
                    avg_step_time,
                )
                # Reset interval accumulators.
                interval_loss_sum = 0.0
                interval_steps = 0
                interval_tokens = 0
                interval_start = time.perf_counter()

        total_time = time.perf_counter() - start_time
        return TrainResult(
            final_step=max_steps,
            final_loss=step_loss,
            total_time=total_time,
            peak_memory=0.0,
            first_step_loss=first_step_loss,
        )
