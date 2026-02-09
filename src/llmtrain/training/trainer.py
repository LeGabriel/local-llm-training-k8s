"""Single-process training loop: optimizer steps, gradient accumulation, LR schedule."""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from llmtrain.config.schemas import RunConfig
from llmtrain.registry import initialize_registries
from llmtrain.registry.data import get_data_module
from llmtrain.registry.models import get_model_adapter
from llmtrain.training.checkpoint import CheckpointManager, CheckpointPayload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainResult:
    """Result of a training run (single process)."""

    final_step: int
    final_loss: float
    final_val_loss: float | None
    total_time: float
    peak_memory: float
    first_step_loss: float | None = None
    resumed_from_step: int | None = None


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

    def __init__(self, cfg: RunConfig, *, run_dir: Path | None = None) -> None:
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
        self._val_loader = data_module.val_dataloader()

        self._device = torch.device("cpu")
        self._model = self._model.to(self._device)

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=cfg.trainer.lr,
            weight_decay=cfg.trainer.weight_decay,
        )
        self._scheduler = self._build_scheduler(self._optimizer)
        self._ckpt_mgr: CheckpointManager | None = None
        if run_dir is not None:
            keep_last_k = int(cfg.trainer.extra.get("keep_last_k", 3))
            self._ckpt_mgr = CheckpointManager(run_dir / "checkpoints", keep_last_k=keep_last_k)

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

    def _resolve_resume_path(self, resume_from: str | Path) -> Path:
        """Resolve a resume spec into a concrete checkpoint path."""
        candidate = Path(resume_from)
        if candidate.exists():
            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                mgr = CheckpointManager(candidate, keep_last_k=1)
                latest = mgr.latest_checkpoint()
                if latest is None:
                    raise FileNotFoundError(f"No checkpoints found in {candidate}")
                return latest
            raise FileNotFoundError(f"Resume path {candidate} is not a file or directory")

        if candidate.suffix == ".pt":
            raise FileNotFoundError(f"Checkpoint file {candidate} does not exist")

        run_id = str(resume_from)
        ckpt_dir = Path(self._cfg.output.root_dir) / run_id / "checkpoints"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist")

        mgr = CheckpointManager(ckpt_dir, keep_last_k=1)
        latest = mgr.latest_checkpoint()
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        return latest

    def _evaluate(self) -> dict[str, float] | None:
        """Run a validation loop and return averaged metrics."""
        if self._val_loader is None:
            return None

        was_training = self._model.training
        self._model.eval()

        metrics_sum: dict[str, float] = {}
        batches = 0

        with torch.no_grad():
            for batch in self._val_loader:
                batch = _move_batch(batch, self._device)
                loss, metrics = self._adapter.compute_loss(self._model, batch)
                batch_metrics = dict(metrics)
                batch_metrics.setdefault("loss", float(loss.item()))
                for key, value in batch_metrics.items():
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value)
                batches += 1

        if was_training:
            self._model.train()

        if batches == 0:
            return {}

        return {f"val/{key}": value / batches for key, value in metrics_sum.items()}

    def fit(
        self,
        *,
        max_steps_override: int | None = None,
        resume_from: str | Path | None = None,
    ) -> TrainResult:
        """Run up to max_steps optimizer steps with gradient accumulation; return the result."""
        self._model.train()
        max_steps = (
            max_steps_override if max_steps_override is not None else self._cfg.trainer.max_steps
        )
        grad_accum_steps = self._cfg.trainer.grad_accum_steps
        log_every = self._cfg.trainer.log_every_steps
        eval_every = self._cfg.trainer.eval_every_steps
        save_every = self._cfg.trainer.save_every_steps

        iterator = iter(self._train_loader)
        start_step = 1
        resume_step = 0
        resumed_from_step: int | None = None
        if resume_from is not None:
            resume_path = self._resolve_resume_path(resume_from)
            mgr = CheckpointManager(resume_path.parent, keep_last_k=1)
            payload = mgr.load(resume_path)
            if payload["config"] != self._cfg.model_dump():
                logger.warning(
                    "checkpoint: config mismatch detected; using current config for resume"
                )
            resume_step = self.restore(payload)
            resumed_from_step = resume_step
            start_step = resume_step + 1
            if start_step > max_steps:
                logger.info(
                    "trainer: resume step %d >= max_steps %d; no further steps",
                    resume_step,
                    max_steps,
                )

        start_time = time.perf_counter()
        if resume_step > 0:
            batches_to_skip = resume_step * grad_accum_steps
            for _ in range(batches_to_skip):
                try:
                    next(iterator)
                except StopIteration:
                    iterator = iter(self._train_loader)
                    next(iterator)
        first_step_loss: float | None = None
        final_val_loss: float | None = None
        step_loss = 0.0

        # Metric logging state: running accumulators for the current log interval.
        interval_loss_sum = 0.0
        interval_steps = 0
        interval_tokens = 0
        interval_start = time.perf_counter()

        for step in range(start_step, max_steps + 1):
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

            if self._ckpt_mgr is not None and (step % save_every == 0 or step == max_steps):
                self._ckpt_mgr.save(
                    step,
                    self._model,
                    self._optimizer,
                    self._scheduler,
                    self._cfg,
                )

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

            if step % eval_every == 0 or step == max_steps:
                eval_metrics = self._evaluate()
                if eval_metrics:
                    metrics_parts = "  ".join(
                        f"{key}={value:.4f}" for key, value in sorted(eval_metrics.items())
                    )
                    logger.info(
                        "val_step=%d/%d  %s",
                        step,
                        max_steps,
                        metrics_parts,
                    )
                    val_loss = eval_metrics.get("val/loss")
                    if val_loss is not None:
                        final_val_loss = val_loss

        total_time = time.perf_counter() - start_time
        return TrainResult(
            final_step=max_steps,
            final_loss=step_loss,
            final_val_loss=final_val_loss,
            total_time=total_time,
            peak_memory=0.0,
            first_step_loss=first_step_loss,
            resumed_from_step=resumed_from_step,
        )
