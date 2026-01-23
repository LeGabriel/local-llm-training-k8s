"""Pydantic schema models for configuration validation."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunConfig(BaseModel):
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
    vocab_size: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_model_dimensions(self) -> Self:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff < self.d_model:
            raise ValueError("d_ff must be greater than or equal to d_model")
        return self
