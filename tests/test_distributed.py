"""Tests for the distributed (DDP) setup / teardown utilities."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from llmtrain.config.schemas import RunConfig
from llmtrain.distributed import DDPState, setup_ddp, teardown_ddp
from llmtrain.training.trainer import Trainer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _always_teardown_ddp() -> Iterator[None]:
    """Guarantee a clean process-group state between tests."""
    yield
    teardown_ddp()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _minimal_run_config(**ddp_overrides: object) -> RunConfig:
    """Build a minimal ``RunConfig`` with DDP overrides applied."""
    return RunConfig.model_validate(
        {
            "schema_version": 1,
            "run": {"name": "test-ddp"},
            "model": {"name": "dummy_gpt"},
            "data": {"name": "dummy_text"},
            "trainer": {},
            "ddp": {**{"enabled": True, "backend": "gloo"}, **ddp_overrides},
            "mlflow": {"enabled": False},
            "logging": {},
            "output": {},
        }
    )


# ---------------------------------------------------------------------------
# DDPState dataclass unit tests
# ---------------------------------------------------------------------------


class TestDDPState:
    """Pure unit tests for the frozen DDPState dataclass."""

    def test_is_main_true_when_rank_zero(self) -> None:
        state = DDPState(rank=0, world_size=2, local_rank=0, is_main=True)
        assert state.is_main is True

    def test_is_main_false_when_rank_nonzero(self) -> None:
        state = DDPState(rank=1, world_size=2, local_rank=1, is_main=False)
        assert state.is_main is False

    def test_is_main_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="is_main"):
            DDPState(rank=0, world_size=2, local_rank=0, is_main=False)

    def test_is_main_mismatch_nonzero_raises(self) -> None:
        with pytest.raises(ValueError, match="is_main"):
            DDPState(rank=1, world_size=2, local_rank=1, is_main=True)

    def test_frozen(self) -> None:
        state = DDPState(rank=0, world_size=1, local_rank=0, is_main=True)
        with pytest.raises(AttributeError):
            state.rank = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# setup_ddp / teardown_ddp tests (single-process process group)
# ---------------------------------------------------------------------------


class TestSetupTeardown:
    """Verify the init/destroy cycle with a single-process group."""

    def test_setup_and_teardown_with_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full init -> verify state -> destroy using env vars (torchrun path)."""
        port = _free_port()
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", str(port))

        cfg = _minimal_run_config()
        state = setup_ddp(cfg)

        assert state.rank == 0
        assert state.world_size == 1
        assert state.local_rank == 0
        assert state.is_main is True
        assert dist.is_initialized()

        teardown_ddp()
        assert not dist.is_initialized()

    def test_setup_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling setup_ddp() twice returns consistent state without error."""
        port = _free_port()
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", str(port))

        cfg = _minimal_run_config()
        state1 = setup_ddp(cfg)
        state2 = setup_ddp(cfg)  # must not raise

        assert state1 == state2
        assert dist.is_initialized()

    def test_teardown_is_safe_when_not_initialised(self) -> None:
        """teardown_ddp must be a no-op when no process group exists."""
        assert not dist.is_initialized()
        teardown_ddp()  # should not raise


# ---------------------------------------------------------------------------
# Config fallback tests
# ---------------------------------------------------------------------------


class TestConfigFallback:
    """Verify that setup_ddp falls back to DDPConfig fields when env vars
    are absent."""

    def test_fallback_to_config_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When env vars are absent, rank/world_size/local_rank come from
        config and MASTER_ADDR/MASTER_PORT are propagated into the env."""
        port = _free_port()

        # Ensure the env vars torchrun would set are absent
        for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            monkeypatch.delenv(var, raising=False)

        cfg = _minimal_run_config(
            rank=0,
            world_size=1,
            local_rank=0,
            master_addr="localhost",
            master_port=port,
        )

        state = setup_ddp(cfg)

        assert state.rank == 0
        assert state.world_size == 1
        assert state.local_rank == 0
        assert state.is_main is True
        assert dist.is_initialized()

        # Verify that config values were propagated into the environment
        # so that init_method="env://" can find them.
        assert os.environ["MASTER_ADDR"] == "localhost"
        assert os.environ["MASTER_PORT"] == str(port)

    def test_missing_rank_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When neither env var nor config provides rank, a RuntimeError
        must be raised."""
        for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            monkeypatch.delenv(var, raising=False)

        cfg = _minimal_run_config()  # rank/world_size/local_rank default to None

        with pytest.raises(RuntimeError, match="rank"):
            setup_ddp(cfg)

    def test_missing_world_size_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            monkeypatch.delenv(var, raising=False)

        cfg = _minimal_run_config(rank=0)

        with pytest.raises(RuntimeError, match="world_size"):
            setup_ddp(cfg)

    def test_missing_local_rank_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            monkeypatch.delenv(var, raising=False)

        cfg = _minimal_run_config(rank=0, world_size=1)

        with pytest.raises(RuntimeError, match="local_rank"):
            setup_ddp(cfg)

    def test_non_integer_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When an env var like RANK is set to a non-integer value,
        a clear RuntimeError must be raised."""
        monkeypatch.setenv("RANK", "not-a-number")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", "29500")

        cfg = _minimal_run_config()

        with pytest.raises(RuntimeError, match="must be an integer"):
            setup_ddp(cfg)


# ---------------------------------------------------------------------------
# DDP model wrapping tests (Trainer integration, 1.0.2)
# ---------------------------------------------------------------------------


class TestDDPModelWrapping:
    """Verify Trainer wraps the model with DistributedDataParallel when
    ddp_state indicates world_size > 1, and that _raw_model unwraps it."""

    def test_model_wrapped_when_world_size_gt_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ddp_state.world_size > 1, the model must be wrapped in DDP
        and model.module must be accessible."""
        port = _free_port()
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", str(port))

        cfg = _minimal_run_config()
        setup_ddp(cfg)

        # Create a DDPState with world_size=2 so the Trainer triggers wrapping,
        # even though the actual process group has world_size=1.
        ddp_state = DDPState(rank=0, world_size=2, local_rank=0, is_main=True)
        trainer = Trainer(cfg, ddp_state=ddp_state)

        assert isinstance(trainer._model, DistributedDataParallel)
        assert hasattr(trainer._model, "module")

    def test_model_not_wrapped_when_no_ddp_state(self) -> None:
        """Without ddp_state the model must remain a plain nn.Module
        (not wrapped in DDP)."""
        cfg = _minimal_run_config()
        trainer = Trainer(cfg, ddp_state=None)

        assert not isinstance(trainer._model, DistributedDataParallel)

    def test_model_not_wrapped_when_world_size_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With world_size=1 the model must not be wrapped (no point in DDP
        with a single process)."""
        port = _free_port()
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", str(port))

        cfg = _minimal_run_config()
        setup_ddp(cfg)

        ddp_state = DDPState(rank=0, world_size=1, local_rank=0, is_main=True)
        trainer = Trainer(cfg, ddp_state=ddp_state)

        assert not isinstance(trainer._model, DistributedDataParallel)

    def test_raw_model_returns_unwrapped_when_ddp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_raw_model must return the inner module when the model is wrapped
        in DistributedDataParallel."""
        port = _free_port()
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", str(port))

        cfg = _minimal_run_config()
        setup_ddp(cfg)

        ddp_state = DDPState(rank=0, world_size=2, local_rank=0, is_main=True)
        trainer = Trainer(cfg, ddp_state=ddp_state)

        raw = trainer._raw_model
        assert not isinstance(raw, DistributedDataParallel)
        assert isinstance(raw, torch.nn.Module)
        # The raw model must be the same object as the DDP wrapper's .module
        assert raw is trainer._model.module

    def test_raw_model_returns_model_when_no_ddp(self) -> None:
        """_raw_model must return self._model directly when DDP is not used."""
        cfg = _minimal_run_config()
        trainer = Trainer(cfg, ddp_state=None)

        raw = trainer._raw_model
        assert raw is trainer._model


# ---------------------------------------------------------------------------
# Rank-0-only I/O tests (1.0.3)
# ---------------------------------------------------------------------------


class TestRank0OnlyIO:
    """Verify that tracker calls and checkpoint saves are gated behind
    ``ddp_state.is_main``."""

    def test_tracker_not_called_on_non_main_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ``ddp_state.is_main`` is False, ``tracker.log_params`` and
        ``tracker.log_metrics`` must never be called during ``fit()``."""
        port = _free_port()
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", str(port))

        cfg = _minimal_run_config()
        setup_ddp(cfg)

        tracker = Mock()
        ddp_state = DDPState(rank=1, world_size=2, local_rank=1, is_main=False)
        trainer = Trainer(cfg, tracker=tracker, ddp_state=ddp_state)
        trainer.fit(max_steps_override=2)

        tracker.log_params.assert_not_called()
        tracker.log_metrics.assert_not_called()

    def test_tracker_called_on_main_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ``ddp_state.is_main`` is True, ``tracker.log_params`` and
        ``tracker.log_metrics`` must be called during ``fit()``."""
        port = _free_port()
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", str(port))

        cfg = _minimal_run_config()
        setup_ddp(cfg)

        tracker = Mock()
        ddp_state = DDPState(rank=0, world_size=2, local_rank=0, is_main=True)
        trainer = Trainer(cfg, tracker=tracker, ddp_state=ddp_state)
        trainer.fit(max_steps_override=2)

        tracker.log_params.assert_called_once()
        tracker.log_metrics.assert_called()

    def test_checkpoint_not_saved_on_non_main_rank(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When ``ddp_state.is_main`` is False, no checkpoint files must be
        written even when a ``run_dir`` is provided."""
        run_dir = tmp_path / "non_main_ckpt"
        run_dir.mkdir(parents=True)

        port = _free_port()
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", str(port))

        cfg = _minimal_run_config()
        setup_ddp(cfg)

        ddp_state = DDPState(rank=1, world_size=2, local_rank=1, is_main=False)
        trainer = Trainer(cfg, run_dir=run_dir, ddp_state=ddp_state)
        trainer.fit(max_steps_override=2)

        ckpt_dir = run_dir / "checkpoints"
        # The CheckpointManager creates the directory but should write no files.
        if ckpt_dir.exists():
            assert list(ckpt_dir.glob("step_*.pt")) == []

    def test_checkpoint_saved_on_main_rank(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When ``ddp_state.is_main`` is True, checkpoint files must be written."""
        run_dir = tmp_path / "main_ckpt"
        run_dir.mkdir(parents=True)

        port = _free_port()
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", str(port))

        cfg = _minimal_run_config()
        setup_ddp(cfg)

        ddp_state = DDPState(rank=0, world_size=2, local_rank=0, is_main=True)
        trainer = Trainer(cfg, run_dir=run_dir, ddp_state=ddp_state)
        trainer.fit(max_steps_override=2)

        ckpt_dir = run_dir / "checkpoints"
        assert ckpt_dir.exists()
        assert len(list(ckpt_dir.glob("step_*.pt"))) > 0
