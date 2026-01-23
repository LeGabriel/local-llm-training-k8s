"""Pydantic schema models for configuration validation."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunSectionConfig(BaseModel):
    """Basic run-level configuration."""

    name: str
    seed: int = 1337
    device: Literal["cpu"] = "cpu"
    deterministic: bool = True
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")


class ModelConfig(BaseModel):
    """Model architecture and initialization details."""

    name: str
    init: Literal["random"] = "random"
    block_size: int = Field(..., ge=8)
    d_model: int = Field(..., ge=64)
    n_layers: int = Field(..., ge=1)
    n_heads: int = Field(..., ge=1)
    d_ff: int = Field(..., ge=1)
    dropout: float = Field(..., ge=0.0, lt=1.0)
    tie_embeddings: bool = True
    vocab_size: int | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_model_dimensions(self) -> Self:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff < self.d_model:
            raise ValueError("d_ff must be greater than or equal to d_model")
        return self


class DataConfig(BaseModel):
    """Dataset paths, splits, and optional HuggingFace overrides."""

    name: str
    cache_dir: str
    num_workers: int = Field(..., ge=0)
    train_split: str
    val_split: str
    dataset_name: str | None = None
    dataset_config: str | None = None
    text_column: str | None = None

    model_config = ConfigDict(extra="forbid")


class TrainerConfig(BaseModel):
    """Training loop pacing and logging controls."""

    max_steps: int = Field(..., ge=1)
    micro_batch_size: int = Field(..., ge=1)
    grad_accum_steps: int = Field(..., ge=1)
    lr: float = Field(..., gt=0.0)
    weight_decay: float = Field(..., ge=0.0)
    warmup_steps: int = Field(..., ge=0)
    max_grad_norm: float = Field(..., gt=0.0)
    log_every_steps: int = Field(..., ge=1)
    eval_every_steps: int = Field(..., ge=1)
    save_every_steps: int = Field(..., ge=1)

    model_config = ConfigDict(extra="forbid")


class DDPConfig(BaseModel):
    """Distributed data-parallel runtime hints and overrides."""

    enabled: bool
    backend: Literal["gloo"] = "gloo"
    init_method: Literal["env://"] = "env://"
    timeout_sec: int = Field(..., ge=1)
    find_unused_parameters: bool = False
    rank: int | None = None
    world_size: int | None = None
    local_rank: int | None = None
    master_addr: str | None = None
    master_port: int | None = None

    model_config = ConfigDict(extra="forbid")


class MLflowConfig(BaseModel):
    """MLflow tracking integration options."""

    enabled: bool
    tracking_uri: str
    experiment: str
    run_name: str | None = None
    log_models: bool = False

    model_config = ConfigDict(extra="forbid")


class LoggingConfig(BaseModel):
    """Structured logging settings for stdout/file output."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    json_output: bool = Field(True, alias="json")
    log_to_file: bool = True
    file_name: str = "train.log"

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class OutputConfig(BaseModel):
    """Output-related paths and persistence toggles."""

    root_dir: str = "runs"
    run_id: str | None = None
    save_config_copy: bool = True
    save_meta_json: bool = True

    model_config = ConfigDict(extra="forbid")


class RunConfig(BaseModel):
    """Top-level schema that ties every section into one executable run."""

    schema_version: int = 1
    run: RunSectionConfig
    model: ModelConfig
    data: DataConfig
    trainer: TrainerConfig
    ddp: DDPConfig
    mlflow: MLflowConfig
    logging: LoggingConfig
    output: OutputConfig

    model_config = ConfigDict(extra="forbid")
