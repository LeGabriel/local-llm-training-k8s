"""Distributed Data Parallel setup, state, and teardown utilities."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta

import torch.distributed as dist

from llmtrain.config.schemas import RunConfig

__all__ = ["DDPState", "setup_ddp", "teardown_ddp"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DDPState:
    """Runtime DDP state resolved from environment variables and config."""

    rank: int
    world_size: int
    local_rank: int
    is_main: bool

    def __post_init__(self) -> None:
        if self.is_main != (self.rank == 0):
            msg = "is_main must be True when rank == 0 and False otherwise"
            raise ValueError(msg)


def _env_int(name: str) -> int | None:
    """Return an env var as int, or *None* if unset.

    Raises :class:`RuntimeError` when the variable is present but cannot be
    parsed as an integer (e.g. empty string or non-numeric value).
    """
    val = os.environ.get(name)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        msg = f"Env var {name} must be an integer, got: {val!r}"
        raise RuntimeError(msg) from None


def _resolve_int(env_name: str, cfg_fallback: int | None, cfg_label: str) -> int:
    """Resolve an integer value from an env var, falling back to config."""
    value = _env_int(env_name)
    if value is not None:
        return value
    if cfg_fallback is not None:
        return cfg_fallback
    msg = f"DDP {cfg_label.split('.')[-1]} not found in env ({env_name}) or config ({cfg_label})"
    raise RuntimeError(msg)


def setup_ddp(cfg: RunConfig) -> DDPState:
    """Initialise the distributed process group and return runtime state.

    Reads ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``MASTER_ADDR``, and
    ``MASTER_PORT`` from the environment (set by ``torchrun``). Falls back
    to the corresponding :class:`~llmtrain.config.schemas.DDPConfig` fields
    when an env var is absent.

    If the process group is already initialised (e.g. a previous call did not
    teardown), the existing group is returned without re-initialising.
    """
    # ------------------------------------------------------------------
    # Idempotency: return current state when already initialised
    # ------------------------------------------------------------------
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        resolved_local_rank = _env_int("LOCAL_RANK")
        if resolved_local_rank is None:
            resolved_local_rank = cfg.ddp.local_rank
        if resolved_local_rank is None:
            resolved_local_rank = rank  # best-effort fallback
        state = DDPState(
            rank=rank,
            world_size=world_size,
            local_rank=resolved_local_rank,
            is_main=(rank == 0),
        )
        logger.warning(
            "DDP process group already initialised — returning existing state: %s",
            state,
        )
        return state

    # ------------------------------------------------------------------
    # Resolve rank / world_size / local_rank
    # ------------------------------------------------------------------
    _launched_by_torchrun = os.environ.get("RANK") is not None

    rank = _resolve_int("RANK", cfg.ddp.rank, "ddp.rank")
    world_size = _resolve_int("WORLD_SIZE", cfg.ddp.world_size, "ddp.world_size")
    local_rank = _resolve_int("LOCAL_RANK", cfg.ddp.local_rank, "ddp.local_rank")

    # ------------------------------------------------------------------
    # Resolve init_method
    # ------------------------------------------------------------------
    # When launched via torchrun the env vars are authoritative — force
    # init_method="env://" regardless of what the config says.  When
    # falling back to config fields, propagate master_addr / master_port
    # into the environment so that init_method="env://" can find them.
    if _launched_by_torchrun:
        init_method = "env://"
    else:
        init_method = cfg.ddp.init_method
        if os.environ.get("MASTER_ADDR") is None and cfg.ddp.master_addr is not None:
            os.environ["MASTER_ADDR"] = cfg.ddp.master_addr
        if os.environ.get("MASTER_PORT") is None and cfg.ddp.master_port is not None:
            os.environ["MASTER_PORT"] = str(cfg.ddp.master_port)

    logger.info(
        "Initialising DDP process group: rank=%d, world_size=%d, local_rank=%d, "
        "backend=%s, init_method=%s",
        rank,
        world_size,
        local_rank,
        cfg.ddp.backend,
        init_method,
    )

    dist.init_process_group(
        backend=cfg.ddp.backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=cfg.ddp.timeout_sec),
    )

    state = DDPState(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_main=(rank == 0),
    )

    logger.info("DDP process group initialised: %s", state)
    return state


def teardown_ddp() -> None:
    """Destroy the distributed process group if one is active."""
    if dist.is_initialized():
        logger.info("Destroying DDP process group")
        dist.destroy_process_group()
