"""Run summary formatting utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from llmtrain.config.schemas import RunConfig


def _ddp_env_snapshot() -> dict[str, str | None]:
    keys = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
    return {key: os.environ.get(key) or None for key in keys}


def format_run_summary(
    *,
    config: RunConfig,
    run_id: str,
    run_dir: str | Path,
    json_output: bool = False,
) -> str | dict[str, Any]:
    """Return a planned run summary as either human text or JSON-ready data."""
    run_path = Path(run_dir)
    ddp_env = _ddp_env_snapshot()

    summary = {
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

    if json_output:
        return summary

    ddp_env_text = ", ".join(f"{key}={value or 'unset'}" for key, value in ddp_env.items())
    model_summary = (
        "Model: "
        f"name={config.model.name} init={config.model.init} block_size={config.model.block_size} "
        f"d_model={config.model.d_model} n_layers={config.model.n_layers} \
            n_heads={config.model.n_heads} "
        f"d_ff={config.model.d_ff} dropout={config.model.dropout} \
            tie_embeddings={config.model.tie_embeddings} "
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
        f"max_grad_norm={config.trainer.max_grad_norm} \
            log_every_steps={config.trainer.log_every_steps} "
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
    return "\n".join(lines)
