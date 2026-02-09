"""Run summary formatting utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from llmtrain.config.schemas import RunConfig
from llmtrain.training.trainer import TrainResult


def _ddp_env_snapshot() -> dict[str, str | None]:
    keys = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
    return {key: os.environ.get(key) or None for key in keys}


def format_run_summary(
    *,
    config: RunConfig,
    run_id: str,
    run_dir: str | Path,
    json_output: bool = False,
    resolved_model_adapter: str | None = None,
    resolved_data_module: str | None = None,
    dry_run_steps_executed: int | None = None,
    train_result: TrainResult | None = None,
    resumed_from: str | None = None,
) -> str | dict[str, Any]:
    """Return a planned run summary as either human text or JSON-ready data."""
    run_path = Path(run_dir)
    ddp_env = _ddp_env_snapshot()

    summary: dict[str, Any] = {
        "run_id": run_id,
        "output_dir": str(run_path),
        "model": {
            "name": config.model.name,
            "init": config.model.init,
            "block_size": config.model.block_size,
            "d_model": config.model.d_model,
            "n_layers": config.model.n_layers,
            "n_heads": config.model.n_heads,
            "d_ff": config.model.d_ff,
            "dropout": config.model.dropout,
            "tie_embeddings": config.model.tie_embeddings,
            "vocab_size": config.model.vocab_size,
        },
        "data": {
            "name": config.data.name,
            "dataset_name": config.data.dataset_name,
            "dataset_config": config.data.dataset_config,
            "text_column": config.data.text_column,
            "cache_dir": config.data.cache_dir,
            "num_workers": config.data.num_workers,
            "train_split": config.data.train_split,
            "val_split": config.data.val_split,
        },
        "trainer": {
            "max_steps": config.trainer.max_steps,
            "micro_batch_size": config.trainer.micro_batch_size,
            "grad_accum_steps": config.trainer.grad_accum_steps,
            "lr": config.trainer.lr,
            "weight_decay": config.trainer.weight_decay,
            "warmup_steps": config.trainer.warmup_steps,
            "max_grad_norm": config.trainer.max_grad_norm,
            "log_every_steps": config.trainer.log_every_steps,
            "eval_every_steps": config.trainer.eval_every_steps,
            "save_every_steps": config.trainer.save_every_steps,
        },
        "ddp": {
            "enabled": config.ddp.enabled,
            "backend": config.ddp.backend,
            "init_method": config.ddp.init_method,
            "timeout_sec": config.ddp.timeout_sec,
            "find_unused_parameters": config.ddp.find_unused_parameters,
            "rank": config.ddp.rank,
            "world_size": config.ddp.world_size,
            "local_rank": config.ddp.local_rank,
            "master_addr": config.ddp.master_addr,
            "master_port": config.ddp.master_port,
            "env": ddp_env,
        },
        "mlflow": {
            "enabled": config.mlflow.enabled,
            "tracking_uri": config.mlflow.tracking_uri,
            "experiment": config.mlflow.experiment,
            "run_name": config.mlflow.run_name,
            "log_models": config.mlflow.log_models,
        },
    }
    if resolved_model_adapter is not None:
        summary["resolved_model_adapter"] = resolved_model_adapter
    if resolved_data_module is not None:
        summary["resolved_data_module"] = resolved_data_module
    if dry_run_steps_executed is not None:
        summary["dry_run_steps_executed"] = dry_run_steps_executed
    if resumed_from is not None:
        summary["resumed_from"] = resumed_from
    if train_result is not None:
        training_dict: dict[str, Any] = {
            "final_step": train_result.final_step,
            "final_loss": train_result.final_loss,
            "first_step_loss": train_result.first_step_loss,
            "total_time": train_result.total_time,
            "peak_memory": train_result.peak_memory,
        }
        if train_result.final_val_loss is not None:
            training_dict["final_val_loss"] = train_result.final_val_loss
        if train_result.resumed_from_step is not None:
            training_dict["resumed_from_step"] = train_result.resumed_from_step
        summary["training"] = training_dict

    if json_output:
        return summary

    ddp_env_text = ", ".join(f"{key}={value or 'unset'}" for key, value in ddp_env.items())
    model_summary = (
        "Model: "
        f"name={config.model.name} init={config.model.init} block_size={config.model.block_size} "
        f"d_model={config.model.d_model} n_layers={config.model.n_layers} "
        f"n_heads={config.model.n_heads} "
        f"d_ff={config.model.d_ff} dropout={config.model.dropout} "
        f"tie_embeddings={config.model.tie_embeddings} "
        f"vocab_size={config.model.vocab_size}"
    )
    data_summary = (
        "Data: "
        f"name={config.data.name} dataset_name={config.data.dataset_name} "
        f"dataset_config={config.data.dataset_config} text_column={config.data.text_column} "
        f"cache_dir={config.data.cache_dir} num_workers={config.data.num_workers} "
        f"train_split={config.data.train_split} val_split={config.data.val_split}"
    )
    trainer_summary = (
        "Trainer: "
        f"max_steps={config.trainer.max_steps} micro_batch_size={config.trainer.micro_batch_size} "
        f"grad_accum_steps={config.trainer.grad_accum_steps} lr={config.trainer.lr} "
        f"weight_decay={config.trainer.weight_decay} warmup_steps={config.trainer.warmup_steps} "
        f"max_grad_norm={config.trainer.max_grad_norm} "
        f"log_every_steps={config.trainer.log_every_steps} "
        f"eval_every_steps={config.trainer.eval_every_steps} "
        f"save_every_steps={config.trainer.save_every_steps}"
    )
    ddp_summary = (
        "DDP: "
        f"enabled={config.ddp.enabled} backend={config.ddp.backend} "
        f"init_method={config.ddp.init_method} timeout_sec={config.ddp.timeout_sec} "
        f"find_unused_parameters={config.ddp.find_unused_parameters} env=[{ddp_env_text}]"
    )
    mlflow_summary = (
        "MLflow: "
        f"enabled={config.mlflow.enabled} tracking_uri={config.mlflow.tracking_uri} "
        f"experiment={config.mlflow.experiment} run_name={config.mlflow.run_name} "
        f"log_models={config.mlflow.log_models}"
    )
    dry_run_summary = None
    if (
        resolved_model_adapter is not None
        or resolved_data_module is not None
        or dry_run_steps_executed is not None
    ):
        dry_run_summary = (
            "Dry run: "
            f"resolved_model_adapter={resolved_model_adapter} "
            f"resolved_data_module={resolved_data_module} "
            f"steps_executed={dry_run_steps_executed}"
        )

    training_summary = None
    if train_result is not None:
        parts = [
            "Training: "
            f"final_step={train_result.final_step} "
            f"final_loss={train_result.final_loss:.4f} "
            f"total_time={train_result.total_time:.2f}s",
        ]
        if train_result.final_val_loss is not None:
            parts[0] += f" final_val_loss={train_result.final_val_loss:.4f}"
        if train_result.resumed_from_step is not None:
            parts[0] += f" resumed_from_step={train_result.resumed_from_step}"
        training_summary = parts[0]

    lines = [
        "Planned run:",
        f"  Run ID: {run_id}",
        f"  Output dir: {run_path}",
        f"  {model_summary}",
        f"  {data_summary}",
        f"  {trainer_summary}",
        f"  {ddp_summary}",
        f"  {mlflow_summary}",
    ]
    if resumed_from is not None:
        lines.append(f"  Resumed from: {resumed_from}")
    if dry_run_summary is not None:
        lines.append(f"  {dry_run_summary}")
    if training_summary is not None:
        lines.append(f"  {training_summary}")
    return "\n".join(lines)
