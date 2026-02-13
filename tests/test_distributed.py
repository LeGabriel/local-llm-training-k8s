"""Tests for the distributed (DDP) setup / teardown utilities."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator

import pytest
import torch.distributed as dist

from llmtrain.config.schemas import RunConfig
from llmtrain.distributed import DDPState, setup_ddp, teardown_ddp

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
            "model": {"name": "dummy"},
            "data": {"name": "dummy"},
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
