"""Training loop: optimizer steps, gradient accumulation, LR schedule, optional DDP."""

from __future__ import annotations

import contextlib
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from llmtrain.config.schemas import RunConfig
from llmtrain.distributed import DDPState
from llmtrain.registry import initialize_registries
from llmtrain.registry.data import get_data_module
from llmtrain.registry.models import get_model_adapter
from llmtrain.tracking import NullTracker, Tracker
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
    val_metrics: dict[str, float] | None = None
    first_step_loss: float | None = None
    resumed_from_step: int | None = None
    parameter_count: int | None = None
    trainable_parameter_count: int | None = None


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class Trainer:
    """Trainer: one device, step-based loop with gradient accumulation and optional DDP."""

    def __init__(
        self,
        cfg: RunConfig,
        *,
        run_dir: Path | None = None,
        tracker: Tracker | None = None,
        ddp_state: DDPState | None = None,
    ) -> None:
        self._cfg = cfg
        self._tracker: Tracker = tracker or NullTracker()
        self._ddp_state = ddp_state
        initialize_registries()

        adapter_cls = get_model_adapter(cfg.model.name)
        data_cls = get_data_module(cfg.data.name)
        self._adapter = adapter_cls()
        data_module = data_cls()

        self._model: torch.nn.Module = self._adapter.build_model(cfg)
        tokenizer = self._adapter.build_tokenizer(cfg)
        data_module.setup(cfg, tokenizer=tokenizer)
        self._train_loader = data_module.train_dataloader()
        self._val_loader = data_module.val_dataloader()

        self._device = torch.device("cpu")
        self._model = self._model.to(self._device)

        # Wrap with DistributedDataParallel when DDP is active with >1 process.
        if ddp_state is not None and ddp_state.world_size > 1:
            self._model = DistributedDataParallel(
                self._model,
                find_unused_parameters=cfg.ddp.find_unused_parameters,
            )

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

    @property
    def _is_main(self) -> bool:
        """Whether this process is the main rank (or DDP is not active)."""
        return self._ddp_state is None or self._ddp_state.is_main

    @property
    def _raw_model(self) -> torch.nn.Module:
        """Return the unwrapped model (strips DDP wrapper if present)."""
        if isinstance(self._model, DistributedDataParallel):
            return self._model.module
        return self._model

    @property
    def _rank(self) -> int:
        """Return the rank of this process (0 when DDP is inactive)."""
        return self._ddp_state.rank if self._ddp_state is not None else 0

    @property
    def _world_size(self) -> int:
        """Return the world size (1 when DDP is inactive)."""
        return self._ddp_state.world_size if self._ddp_state is not None else 1

    @property
    def _is_ddp_active(self) -> bool:
        """Whether multi-process DDP is active (world_size > 1)."""
        return self._ddp_state is not None and self._ddp_state.world_size > 1

    def _reduce_metrics(self, **scalars: float) -> dict[str, float]:
        """All-reduce SUM the given scalar values across ranks.

        Returns the local values unchanged when DDP is not active.
        """
        if not self._is_ddp_active:
            return dict(scalars)
        keys = list(scalars.keys())
        tensor = torch.tensor([scalars[k] for k in keys], dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return {k: float(v) for k, v in zip(keys, tensor, strict=True)}

    def restore(self, payload: CheckpointPayload) -> int:
        """Restore model, optimizer, scheduler, and RNG states from a checkpoint payload.

        Returns the step number stored in the checkpoint (i.e. the step to
        resume *from*).
        """
        self._raw_model.load_state_dict(payload["model_state_dict"])
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

    def _evaluate(self) -> tuple[dict[str, float], dict[str, float]] | None:
        """Run a validation loop with token-weighted loss averaging.

        Returns ``(local_metrics, global_metrics)`` where *global_metrics*
        are all-reduced across ranks when DDP is active.  When DDP is
        inactive the two dicts are identical.  Returns ``None`` when no
        validation loader is configured.
        """
        if self._val_loader is None:
            return None

        was_training = self._model.training
        self._model.eval()

        loss_sum_local = 0.0
        tok_count_local = 0

        with torch.no_grad():
            for batch in self._val_loader:
                batch = _move_batch(batch, self._device)
                loss, _metrics = self._adapter.compute_loss(self._raw_model, batch)
                tokens = batch["input_ids"].numel()
                loss_sum_local += loss.item() * tokens
                tok_count_local += tokens

        if was_training:
            self._model.train()

        if tok_count_local == 0:
            return {}, {}

        local_val_loss = loss_sum_local / tok_count_local
        local_metrics = {"val/loss": local_val_loss}

        if self._is_ddp_active:
            reduced = self._reduce_metrics(
                loss_sum=loss_sum_local,
                tok_count=float(tok_count_local),
            )
            global_val_loss = (
                reduced["loss_sum"] / reduced["tok_count"] if reduced["tok_count"] > 0 else 0.0
            )
            global_metrics = {"val/loss": global_val_loss}
        else:
            global_metrics = dict(local_metrics)

        return local_metrics, global_metrics

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
        if self._is_main:
            self._tracker.log_params(self._cfg.model_dump())
        parameter_count = sum(param.numel() for param in self._raw_model.parameters())
        trainable_parameter_count = sum(
            param.numel() for param in self._raw_model.parameters() if param.requires_grad
        )

        start_time = time.perf_counter()
        # Skip batches to approximate the same data order on resume.
        # This is only safe in single-process mode; under DDP each rank's
        # DistributedSampler produces a different shard, so naively
        # skipping batches would desynchronise the ranks.
        if resume_step > 0 and not self._is_ddp_active:
            batches_to_skip = resume_step * grad_accum_steps
            for _ in range(batches_to_skip):
                try:
                    next(iterator)
                except StopIteration:
                    iterator = iter(self._train_loader)
                    next(iterator)
        first_step_loss: float | None = None
        final_val_loss: float | None = None
        final_val_metrics: dict[str, float] | None = None
        step_loss = 0.0
        total_tokens_local = 0
        tokens_total_global = 0  # maintained on rank 0 only

        # Metric logging state: running accumulators for the current log interval.
        interval_loss_sum = 0.0
        interval_steps = 0
        interval_tokens = 0
        interval_start = time.perf_counter()

        for step in range(start_step, max_steps + 1):
            self._optimizer.zero_grad()
            accumulated_loss = 0.0
            step_tokens = 0

            for micro_step in range(grad_accum_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(self._train_loader)
                    batch = next(iterator)

                batch = _move_batch(batch, self._device)
                step_tokens += batch["input_ids"].numel()

                # Skip all-reduce on non-final micro-batches when DDP is active.
                is_last_micro = micro_step == grad_accum_steps - 1
                sync_ctx: contextlib.AbstractContextManager[None] = (
                    self._model.no_sync()
                    if not is_last_micro and isinstance(self._model, DistributedDataParallel)
                    else contextlib.nullcontext()
                )

                with sync_ctx:
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
            total_tokens_local += step_tokens
            if step == 1:
                first_step_loss = step_loss

            if (
                self._ckpt_mgr is not None
                and (step % save_every == 0 or step == max_steps)
                and self._is_main
            ):
                self._ckpt_mgr.save(
                    step,
                    self._raw_model,
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

                if self._is_ddp_active:
                    # -- Per-rank metrics (every rank logs its own shard) --
                    rank = self._rank
                    self._tracker.log_metrics(
                        {
                            f"train/loss_rank_{rank}": avg_loss,
                            f"train/lr_rank_{rank}": current_lr,
                            f"train/tokens_per_sec_rank_{rank}": tokens_per_sec,
                            f"train/step_time_sec_rank_{rank}": avg_step_time,
                            f"train/tokens_total_rank_{rank}": float(total_tokens_local),
                        },
                        step=step,
                    )
                    # -- Global reduced metrics (rank 0 only) --
                    reduced = self._reduce_metrics(
                        loss_sum=interval_loss_sum,
                        steps=float(interval_steps),
                        tokens=float(interval_tokens),
                    )
                    if self._is_main:
                        global_loss = reduced["loss_sum"] / reduced["steps"]
                        global_tokens = reduced["tokens"]
                        tokens_total_global += int(global_tokens)
                        global_tps = global_tokens / interval_time if interval_time > 0 else 0.0
                        self._tracker.log_metrics(
                            {
                                "train/loss": global_loss,
                                "train/lr": current_lr,
                                "train/tokens_per_sec": global_tps,
                                "train/tokens_total": float(tokens_total_global),
                                "train/step_time_sec": avg_step_time,
                            },
                            step=step,
                        )
                else:
                    # Non-DDP: no rank suffixes, global == local.
                    if self._is_main:
                        self._tracker.log_metrics(
                            {
                                "train/loss": avg_loss,
                                "train/lr": current_lr,
                                "train/tokens_per_sec": tokens_per_sec,
                                "train/step_time_sec": avg_step_time,
                                "train/tokens_total": float(total_tokens_local),
                            },
                            step=step,
                        )

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
                eval_result = self._evaluate()
                if eval_result is not None:
                    local_val, global_val = eval_result

                    # Per-rank validation metrics under DDP.
                    if self._is_ddp_active:
                        rank = self._rank
                        rank_metrics = {f"{k}_rank_{rank}": v for k, v in local_val.items()}
                        self._tracker.log_metrics(rank_metrics, step=step)

                    # Global validation metrics (rank 0 under DDP, or main otherwise).
                    if self._is_main:
                        self._tracker.log_metrics(global_val, step=step)

                    # Update final result state.
                    effective = global_val if self._is_main else local_val
                    if effective:
                        final_val_metrics = effective
                        val_loss = effective.get("val/loss")
                        if val_loss is not None:
                            final_val_loss = val_loss
                        metrics_parts = "  ".join(
                            f"{key}={value:.4f}" for key, value in sorted(effective.items())
                        )
                        logger.info(
                            "val_step=%d/%d  %s",
                            step,
                            max_steps,
                            metrics_parts,
                        )

        total_time = time.perf_counter() - start_time
        return TrainResult(
            final_step=max_steps,
            final_loss=step_loss,
            final_val_loss=final_val_loss,
            total_time=total_time,
            peak_memory=0.0,
            val_metrics=final_val_metrics,
            first_step_loss=first_step_loss,
            resumed_from_step=resumed_from_step,
            parameter_count=parameter_count,
            trainable_parameter_count=trainable_parameter_count,
        )
