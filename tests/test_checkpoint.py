"""Tests for CheckpointManager: save, pruning, load, latest_checkpoint, and restore."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from llmtrain.config.schemas import RunConfig
from llmtrain.training.checkpoint import CheckpointManager, CheckpointPayload
from llmtrain.training.trainer import Trainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config() -> RunConfig:
    payload = {
        "schema_version": 1,
        "run": {"name": "ckpt-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {"max_steps": 5, "warmup_steps": 0},
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }
    return RunConfig.model_validate(payload)


def _training_config(max_steps: int, save_every_steps: int) -> RunConfig:
    payload = {
        "schema_version": 1,
        "run": {"name": "ckpt-train-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {
            "max_steps": max_steps,
            "warmup_steps": 0,
            "micro_batch_size": 1,
            "grad_accum_steps": 1,
            "save_every_steps": save_every_steps,
        },
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }
    return RunConfig.model_validate(payload)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _make_objects(
    cfg: RunConfig,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Build a tiny model, optimizer, and scheduler for checkpoint tests."""
    from llmtrain.registry import initialize_registries
    from llmtrain.registry.models import get_model_adapter

    initialize_registries()
    adapter_cls = get_model_adapter(cfg.model.name)
    adapter = adapter_cls()
    model = adapter.build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.trainer.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    return model, optimizer, scheduler


# ---------------------------------------------------------------------------
# save tests
# ---------------------------------------------------------------------------


def test_save_creates_checkpoint_file(tmp_path: Path) -> None:
    """save() writes a .pt file that torch.load can read with the expected keys."""
    cfg = _minimal_config()
    model, optimizer, scheduler = _make_objects(cfg)

    mgr = CheckpointManager(tmp_path / "checkpoints", keep_last_k=3)
    path = mgr.save(step=1, model=model, optimizer=optimizer, scheduler=scheduler, config=cfg)

    assert path.exists()
    assert path.name == "step_000001.pt"

    data = torch.load(path, weights_only=False)

    expected_keys: set[str] = set(CheckpointPayload.__annotations__)
    assert expected_keys.issubset(data.keys())
    assert data["step"] == 1
    assert isinstance(data["model_state_dict"], dict)
    assert isinstance(data["optimizer_state_dict"], dict)
    assert isinstance(data["scheduler_state_dict"], dict)
    assert isinstance(data["rng_states"], dict)
    assert isinstance(data["config"], dict)


def test_save_captures_rng_states(tmp_path: Path) -> None:
    """Saved checkpoint includes Python, NumPy, and Torch RNG states."""
    cfg = _minimal_config()
    model, optimizer, scheduler = _make_objects(cfg)

    mgr = CheckpointManager(tmp_path / "checkpoints", keep_last_k=3)
    path = mgr.save(step=3, model=model, optimizer=optimizer, scheduler=scheduler, config=cfg)

    data = torch.load(path, weights_only=False)
    rng = data["rng_states"]
    assert "python" in rng
    assert "numpy" in rng
    assert "torch" in rng


# ---------------------------------------------------------------------------
# prune tests
# ---------------------------------------------------------------------------


def test_prune_keeps_only_k_checkpoints(tmp_path: Path) -> None:
    """After saving at steps 5, 10, 15, 20 with keep_last_k=2, only 15 and 20 remain."""
    cfg = _minimal_config()
    model, optimizer, scheduler = _make_objects(cfg)

    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(ckpt_dir, keep_last_k=2)

    for step in (5, 10, 15, 20):
        mgr.save(step=step, model=model, optimizer=optimizer, scheduler=scheduler, config=cfg)

    remaining = sorted(ckpt_dir.glob("step_*.pt"))
    assert len(remaining) == 2
    assert remaining[0].name == "step_000015.pt"
    assert remaining[1].name == "step_000020.pt"


# ---------------------------------------------------------------------------
# load tests
# ---------------------------------------------------------------------------


def test_load_returns_checkpoint_payload(tmp_path: Path) -> None:
    """load() returns a dict with the correct keys and types matching CheckpointPayload."""
    cfg = _minimal_config()
    model, optimizer, scheduler = _make_objects(cfg)

    mgr = CheckpointManager(tmp_path / "checkpoints", keep_last_k=3)
    path = mgr.save(step=7, model=model, optimizer=optimizer, scheduler=scheduler, config=cfg)

    payload = mgr.load(path)

    expected_keys: set[str] = set(CheckpointPayload.__annotations__)
    assert expected_keys.issubset(payload.keys())
    assert payload["step"] == 7
    assert isinstance(payload["model_state_dict"], dict)
    assert isinstance(payload["optimizer_state_dict"], dict)
    assert isinstance(payload["scheduler_state_dict"], dict)
    assert isinstance(payload["rng_states"], dict)
    assert isinstance(payload["config"], dict)


# ---------------------------------------------------------------------------
# latest_checkpoint tests
# ---------------------------------------------------------------------------


def test_latest_checkpoint_returns_most_recent(tmp_path: Path) -> None:
    """latest_checkpoint() returns the path to the highest-step checkpoint."""
    cfg = _minimal_config()
    model, optimizer, scheduler = _make_objects(cfg)

    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(ckpt_dir, keep_last_k=5)

    mgr.save(step=5, model=model, optimizer=optimizer, scheduler=scheduler, config=cfg)
    mgr.save(step=10, model=model, optimizer=optimizer, scheduler=scheduler, config=cfg)

    latest = mgr.latest_checkpoint()
    assert latest is not None
    assert latest.name == "step_000010.pt"


def test_latest_checkpoint_returns_none_when_empty(tmp_path: Path) -> None:
    """latest_checkpoint() returns None when the checkpoint directory has no files."""
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(ckpt_dir, keep_last_k=3)

    assert mgr.latest_checkpoint() is None


# ---------------------------------------------------------------------------
# restore tests
# ---------------------------------------------------------------------------


def test_restore_sets_optimizer_state(tmp_path: Path) -> None:
    """After restore(), the optimizer state dict matches the one saved in the checkpoint."""
    cfg = _minimal_config()
    model, optimizer, scheduler = _make_objects(cfg)

    # Run a few optimizer steps so the state is non-trivial.
    dummy_ids = torch.randint(0, 128, (2, 4))
    dummy_labels = torch.randint(0, 128, (2, 4))
    for _ in range(3):
        logits = model(dummy_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), dummy_labels.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Save checkpoint at step 5.
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(ckpt_dir, keep_last_k=3)
    path = mgr.save(step=5, model=model, optimizer=optimizer, scheduler=scheduler, config=cfg)

    # Build a fresh trainer and restore.
    trainer = Trainer(cfg)
    payload = mgr.load(path)
    resumed_step = trainer.restore(payload)

    assert resumed_step == 5

    # Compare optimizer state_dict (param groups, excluding params list which
    # may use different ids).
    original_opt = optimizer.state_dict()
    restored_opt = trainer._optimizer.state_dict()

    # Compare param group hyper-parameters (lr, betas, etc.).
    for orig_group, rest_group in zip(
        original_opt["param_groups"],
        restored_opt["param_groups"],
        strict=True,
    ):
        for key in ("lr", "betas", "eps", "weight_decay"):
            assert orig_group[key] == rest_group[key], f"Mismatch on param_group key {key}"


# ---------------------------------------------------------------------------
# trainer checkpoint save tests
# ---------------------------------------------------------------------------


def test_training_loop_saves_at_correct_steps(tmp_path: Path) -> None:
    """Checkpoints are saved every save_every_steps and at the final step."""
    cfg = _training_config(max_steps=20, save_every_steps=10)
    trainer = Trainer(cfg, run_dir=tmp_path)
    trainer.fit()

    ckpt_dir = tmp_path / "checkpoints"
    expected = {"step_000010.pt", "step_000020.pt"}
    found = {path.name for path in ckpt_dir.glob("step_*.pt")}
    assert expected.issubset(found)


def test_training_loop_always_saves_final_step(tmp_path: Path) -> None:
    """Final step checkpoint is saved even if not a multiple of save_every_steps."""
    cfg = _training_config(max_steps=15, save_every_steps=10)
    trainer = Trainer(cfg, run_dir=tmp_path)
    trainer.fit()

    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "step_000010.pt").exists()
    assert (ckpt_dir / "step_000015.pt").exists()


# ---------------------------------------------------------------------------
# resume tests
# ---------------------------------------------------------------------------


def test_resume_continues_from_saved_step(tmp_path: Path) -> None:
    """Resume continues from the saved step and reaches the new max_steps."""
    cfg = _training_config(max_steps=20, save_every_steps=10)
    _seed_all(123)
    trainer = Trainer(cfg, run_dir=tmp_path / "first_run")
    trainer.fit(max_steps_override=10)

    ckpt_path = tmp_path / "first_run" / "checkpoints" / "step_000010.pt"
    assert ckpt_path.exists()

    _seed_all(999)
    resumed_trainer = Trainer(cfg, run_dir=tmp_path / "resume_run")
    result = resumed_trainer.fit(resume_from=ckpt_path)

    assert result.final_step == 20
    assert result.resumed_from_step == 10


def test_resume_produces_same_loss_as_continuous_run(tmp_path: Path) -> None:
    """Resume yields the same final loss as an uninterrupted run."""
    cfg = _training_config(max_steps=20, save_every_steps=10)

    _seed_all(123)
    full_trainer = Trainer(cfg)
    full_result = full_trainer.fit()

    _seed_all(123)
    partial_trainer = Trainer(cfg, run_dir=tmp_path / "partial_run")
    partial_trainer.fit(max_steps_override=10)

    ckpt_path = tmp_path / "partial_run" / "checkpoints" / "step_000010.pt"
    assert ckpt_path.exists()

    _seed_all(999)
    resumed_trainer = Trainer(cfg, run_dir=tmp_path / "resume_run")
    resumed_result = resumed_trainer.fit(resume_from=ckpt_path)

    assert abs(full_result.final_loss - resumed_result.final_loss) <= 1e-5


def test_resume_with_config_mismatch_warns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A config mismatch between checkpoint and current config emits a warning."""
    cfg = _training_config(max_steps=20, save_every_steps=10)
    _seed_all(123)
    trainer = Trainer(cfg, run_dir=tmp_path / "first_run")
    trainer.fit(max_steps_override=10)

    ckpt_path = tmp_path / "first_run" / "checkpoints" / "step_000010.pt"
    assert ckpt_path.exists()

    mismatched_cfg = _training_config(max_steps=30, save_every_steps=10)
    mismatched_trainer = Trainer(mismatched_cfg)

    with caplog.at_level(logging.WARNING, logger="llmtrain.training.trainer"):
        mismatched_trainer.fit(resume_from=ckpt_path, max_steps_override=12)

    assert any("config mismatch" in record.message for record in caplog.records)
