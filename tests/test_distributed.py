"""Tests for the distributed (DDP) setup / teardown utilities."""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import yaml
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
        """When ``ddp_state.is_main`` is False, ``tracker.log_params`` must
        not be called, and only rank-suffixed metrics are logged (no global
        keys like ``train/loss`` or ``val/loss``)."""
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
        # Non-main ranks now log per-rank metrics but must NOT log global keys.
        tracker.log_metrics.assert_called()
        for call in tracker.log_metrics.call_args_list:
            metrics_dict = call[0][0]
            for key in metrics_dict:
                assert "_rank_" in key, f"Non-main rank logged global metric key: {key}"

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


# ---------------------------------------------------------------------------
# Slow multi-process integration test (1.0.5)
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_preset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in preset {path}")
    return cast(dict[str, Any], payload)


def _write_config(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Metric key naming tests (per-rank vs global)
# ---------------------------------------------------------------------------


class TestMetricKeyNaming:
    """Verify that metric keys use rank suffixes only under multi-process DDP
    and that non-DDP runs produce plain (unsuffixed) keys."""

    def test_no_rank_suffix_without_ddp(self) -> None:
        """Without DDP, metric keys must NOT contain rank suffixes."""
        cfg = _minimal_run_config()
        tracker = Mock()
        trainer = Trainer(cfg, tracker=tracker, ddp_state=None)
        trainer.fit(max_steps_override=2)

        for call in tracker.log_metrics.call_args_list:
            metrics_dict = call[0][0]
            for key in metrics_dict:
                assert "_rank_" not in key, f"Non-DDP run logged rank-suffixed key: {key}"

    def test_rank_suffix_and_global_keys_under_ddp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under DDP (rank 0), both rank-suffixed and global keys must appear."""
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

        all_keys: set[str] = set()
        for call in tracker.log_metrics.call_args_list:
            metrics_dict = call[0][0]
            all_keys.update(metrics_dict.keys())

        rank_keys = {k for k in all_keys if "_rank_0" in k}
        global_keys = {k for k in all_keys if "_rank_" not in k}

        assert rank_keys, f"Expected rank-suffixed keys, got: {all_keys}"
        assert global_keys, f"Expected global keys, got: {all_keys}"
        # Verify specific expected keys are present.
        assert "train/loss_rank_0" in rank_keys
        assert "train/loss" in global_keys

    def test_non_main_only_logs_rank_suffixed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-main rank must only log rank-suffixed metrics, never global."""
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
        for call in tracker.log_metrics.call_args_list:
            metrics_dict = call[0][0]
            for key in metrics_dict:
                assert "_rank_1" in key, f"Non-main rank logged global metric key: {key}"


# ---------------------------------------------------------------------------
# _reduce_metrics helper tests
# ---------------------------------------------------------------------------


class TestReduceMetrics:
    """Verify the _reduce_metrics helper returns local values when DDP is
    not active."""

    def test_returns_local_when_no_ddp(self) -> None:
        """Without any DDP state, _reduce_metrics must return local values."""
        cfg = _minimal_run_config()
        trainer = Trainer(cfg, ddp_state=None)

        result = trainer._reduce_metrics(a=1.0, b=2.5, c=3.0)

        assert result == {"a": 1.0, "b": 2.5, "c": 3.0}

    def test_returns_local_when_world_size_1(self) -> None:
        """With world_size=1, _reduce_metrics must return local values
        unchanged (DDP is not active)."""
        cfg = _minimal_run_config()
        ddp_state = DDPState(rank=0, world_size=1, local_rank=0, is_main=True)
        trainer = Trainer(cfg, ddp_state=ddp_state)

        result = trainer._reduce_metrics(loss_sum=10.0, steps=5.0)

        assert result == {"loss_sum": 10.0, "steps": 5.0}

    def test_reduce_with_single_process_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a single-process group and DDPState.world_size > 1 the
        all_reduce path is exercised but the result equals the input
        (only one process contributing)."""
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

        result = trainer._reduce_metrics(loss_sum=4.0, steps=2.0, tokens=100.0)

        # Single-process all_reduce SUM == identity.
        assert result["loss_sum"] == pytest.approx(4.0)
        assert result["steps"] == pytest.approx(2.0)
        assert result["tokens"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Slow multi-process integration test (1.0.5)
# ---------------------------------------------------------------------------


class TestDDPMultiProcessIntegration:
    """Slow integration tests that spawn real multi-process torchrun runs."""

    @pytest.mark.slow
    def test_ddp_two_process_torchrun(self, tmp_path: Path) -> None:
        """Spawn a 2-process DDP run via ``torchrun`` and verify correctness.

        Asserts:
        - Process exits with code 0.
        - JSON summary ``training.final_step`` matches ``max_steps``.
        - ``training.final_loss`` is finite.
        - ``training.parameter_count > 0``.
        - Only one set of checkpoint files exists (rank-0-only saving).
        """
        root_dir = _repo_root()
        preset_path = root_dir / "configs" / "presets" / "ddp_smoke.yaml"
        payload = _load_preset(preset_path)

        max_steps = int(payload["trainer"]["max_steps"])
        save_every = int(payload["trainer"]["save_every_steps"])

        # Redirect output into the temp directory.
        payload["output"] = {
            "root_dir": str(tmp_path / "runs"),
            "save_config_copy": True,
            "save_meta_json": True,
        }

        config_path = _write_config(tmp_path / "ddp_smoke_test.yaml", payload)
        run_id = "ddp-integration-test"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=2",
                "-m",
                "llmtrain",
                "train",
                "--config",
                str(config_path),
                "--run-id",
                run_id,
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

        assert result.returncode == 0, (
            f"torchrun exited with code {result.returncode}\n"
            f"--- stderr ---\n{result.stderr}\n"
            f"--- stdout ---\n{result.stdout}"
        )

        # Rank 0 emits exactly one JSON summary on stdout.
        summary = json.loads(result.stdout)
        training = summary["training"]

        assert training["final_step"] == max_steps
        assert math.isfinite(training["final_loss"])
        assert training["parameter_count"] > 0

        # Only rank 0 saves checkpoints — verify a single set of files.
        run_dir = tmp_path / "runs" / run_id
        ckpt_dir = run_dir / "checkpoints"
        ckpt_files = sorted(ckpt_dir.glob("step_*.pt"))

        expected_ckpts = max_steps // save_every
        assert len(ckpt_files) == expected_ckpts, (
            f"Expected {expected_ckpts} checkpoint(s), found {len(ckpt_files)}: "
            f"{[f.name for f in ckpt_files]}"
        )

        # Confirm rank-0-only: no second run directory or duplicate checkpoints.
        all_run_dirs = list((tmp_path / "runs").iterdir())
        assert len(all_run_dirs) == 1, (
            f"Expected exactly 1 run directory, found {len(all_run_dirs)}: "
            f"{[d.name for d in all_run_dirs]}"
        )
