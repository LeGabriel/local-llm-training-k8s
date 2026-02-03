from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch

from llmtrain.config.schemas import RunConfig
from llmtrain.registry import initialize_registries
from llmtrain.registry.data import get_data_module
from llmtrain.registry.models import get_model_adapter

DEFAULT_DRY_RUN_STEPS = 5


@dataclass(frozen=True)
class DryRunResult:
    resolved_model_adapter: str
    resolved_data_module: str
    steps_executed: int


def _resolve_dry_run_steps(cfg: RunConfig) -> int:
    steps = DEFAULT_DRY_RUN_STEPS
    if steps < 1:
        steps = DEFAULT_DRY_RUN_STEPS
    return min(steps, cfg.trainer.max_steps)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_dry_run(cfg: RunConfig, *, logger: logging.Logger | None = None) -> DryRunResult:
    logger = logger or logging.getLogger(__name__)
    initialize_registries()

    adapter_cls = get_model_adapter(cfg.model.name)
    data_cls = get_data_module(cfg.data.name)
    adapter = adapter_cls()
    data_module = data_cls()

    model = adapter.build_model(cfg)
    tokenizer = adapter.build_tokenizer(cfg)
    data_module.setup(cfg, tokenizer=tokenizer)
    train_loader = data_module.train_dataloader()

    device = torch.device(cfg.run.device)
    model = model.to(device)
    model.train()

    steps = _resolve_dry_run_steps(cfg)
    iterator = iter(train_loader)
    steps_executed = 0
    with torch.no_grad():
        for step in range(1, steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batch = _move_batch(batch, device)
            start_time = time.perf_counter()
            loss, metrics = adapter.compute_loss(model, batch)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            steps_executed += 1
            loss_value = metrics.get("loss", float(loss.item()))
            logger.info(
                "Dry-run step %s/%s loss=%.4f step_time_ms=%.2f",
                step,
                steps,
                loss_value,
                duration_ms,
            )

    return DryRunResult(
        resolved_model_adapter=cfg.model.name,
        resolved_data_module=cfg.data.name,
        steps_executed=steps_executed,
    )
