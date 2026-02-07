"""Single-process training loop: optimizer steps, gradient accumulation, LR schedule."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from llmtrain.config.schemas import RunConfig
from llmtrain.registry import initialize_registries
from llmtrain.registry.data import get_data_module
from llmtrain.registry.models import get_model_adapter


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

    def fit(self) -> TrainResult:
        """Run up to max_steps optimizer steps with gradient accumulation; return the result."""
        self._model.train()
        max_steps = self._cfg.trainer.max_steps
        grad_accum_steps = self._cfg.trainer.grad_accum_steps

        start_time = time.perf_counter()
        iterator = iter(self._train_loader)
        first_step_loss: float | None = None
        step_loss = 0.0

        for step in range(1, max_steps + 1):
            self._optimizer.zero_grad()
            accumulated_loss = 0.0

            for _ in range(grad_accum_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(self._train_loader)
                    batch = next(iterator)

                batch = _move_batch(batch, self._device)
                loss, metrics = self._adapter.compute_loss(self._model, batch)
                scaled = loss / grad_accum_steps
                scaled.backward()
                accumulated_loss += float(metrics.get("loss", loss.item()))

            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self._cfg.trainer.max_grad_norm,
            )
            self._optimizer.step()

            step_loss = accumulated_loss / grad_accum_steps
            if step == 1:
                first_step_loss = step_loss

        total_time = time.perf_counter() - start_time
        return TrainResult(
            final_step=max_steps,
            final_loss=step_loss,
            total_time=total_time,
            peak_memory=0.0,
            first_step_loss=first_step_loss,
        )
