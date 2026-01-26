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

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class ModelConfig(BaseModel):
    """Model architecture and initialization details."""

    name: str
    init: Literal["random"] = "random"
    block_size: int = Field(256, ge=8)
    d_model: int = Field(384, ge=64)
    n_layers: int = Field(6, ge=1)
    n_heads: int = Field(6, ge=1)
    d_ff: int = Field(1536, ge=64)
    dropout: float = Field(0.1, ge=0.0, lt=1.0)
    tie_embeddings: bool = True
    vocab_size: int | None = None

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

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
    cache_dir: str = ".cache/datasets"
    num_workers: int = Field(2, ge=0)
    train_split: str = "train"
    val_split: str = "validation"
    dataset_name: str | None = None
    dataset_config: str | None = None
    text_column: str | None = None

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class TrainerConfig(BaseModel):
    """Training loop pacing and logging controls."""

    max_steps: int = Field(1000, ge=1)
    micro_batch_size: int = Field(8, ge=1)
    grad_accum_steps: int = Field(4, ge=1)
    lr: float = Field(3e-4, gt=0.0)
    weight_decay: float = Field(0.1, ge=0.0)
    warmup_steps: int = Field(100, ge=0)
    max_grad_norm: float = Field(1.0, gt=0.0)
    log_every_steps: int = Field(10, ge=1)
    eval_every_steps: int = Field(100, ge=1)
    save_every_steps: int = Field(500, ge=1)

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    @model_validator(mode="after")
    def check_steps(self) -> Self:
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps cannot exceed max_steps")
        return self


class DDPConfig(BaseModel):
    """Distributed data-parallel runtime hints and overrides."""

    enabled: bool = False
    backend: Literal["gloo"] = "gloo"
    init_method: Literal["env://"] = "env://"
    timeout_sec: int = Field(1800, ge=1)
    find_unused_parameters: bool = False
    rank: int | None = None
    world_size: int | None = None
    local_rank: int | None = None
    master_addr: str | None = None
    master_port: int | None = None

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class MLflowConfig(BaseModel):
    """MLflow tracking integration options."""

    enabled: bool = True
    tracking_uri: str = "file:./mlruns"
    experiment: str = "llm-train-k8s"
    run_name: str | None = None
    log_models: bool = False

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class LoggingConfig(BaseModel):
    """Structured logging settings for stdout/file output."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_output: bool = True
    log_to_file: bool = True
    file_name: str = "train.log"

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class OutputConfig(BaseModel):
    """Output-related paths and persistence toggles."""

    root_dir: str = "runs"
    run_id: str | None = None
    save_config_copy: bool = True
    save_meta_json: bool = True

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class RunConfig(BaseModel):
    """Top-level schema that ties every section into one executable run."""

    schema_version: int = Field(1, ge=1)
    run: RunSectionConfig
    model: ModelConfig
    data: DataConfig
    trainer: TrainerConfig
    ddp: DDPConfig
    mlflow: MLflowConfig
    logging: LoggingConfig
    output: OutputConfig

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )
