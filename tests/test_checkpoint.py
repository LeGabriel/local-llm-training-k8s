"""Tests for CheckpointManager: save and pruning."""

from __future__ import annotations

from pathlib import Path

import torch

from llmtrain.config.schemas import RunConfig
from llmtrain.training.checkpoint import CheckpointManager, CheckpointPayload

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
