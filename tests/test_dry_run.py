from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from llmtrain.config.schemas import RunConfig
from llmtrain.data.base import DataModule
from llmtrain.registry import data as data_registry
from llmtrain.training.dry_run import DEFAULT_DRY_RUN_STEPS, run_dry_run


def _minimal_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run": {"name": "dry-run-test"},
        "model": {"name": "dummy_gpt"},
        "data": {"name": "dummy_text"},
        "trainer": {"max_steps": 5, "warmup_steps": 0},
        "ddp": {},
        "mlflow": {},
        "logging": {"log_to_file": False},
        "output": {"root_dir": "runs"},
    }


def test_dry_run_executes_default_steps() -> None:
    cfg = RunConfig.model_validate(_minimal_payload())
    result = run_dry_run(cfg)
    assert result.steps_executed == DEFAULT_DRY_RUN_STEPS
    assert result.resolved_model_adapter == "dummy_gpt"
    assert result.resolved_data_module == "dummy_text"


def test_dry_run_stops_on_short_dataloader() -> None:
    name = "short-text"
    if name not in data_registry.available_data_modules():

        @data_registry.register_data_module(name)
        class ShortDataModule(DataModule):
            def setup(self, cfg: RunConfig, tokenizer: object | None = None) -> None:
                del cfg
                del tokenizer

            def train_dataloader(self) -> DataLoader:
                class _SingleBatchDataset(Dataset[dict[str, torch.Tensor]]):
                    def __len__(self) -> int:
                        return 1

                    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                        del index
                        return {
                            "input_ids": torch.zeros(1, 1, dtype=torch.long),
                            "labels": torch.zeros(1, 1, dtype=torch.long),
                        }

                return DataLoader(_SingleBatchDataset(), batch_size=None)

            def val_dataloader(self) -> DataLoader | None:
                return None

    payload = _minimal_payload()
    payload["data"] = {"name": name}
    cfg = RunConfig.model_validate(payload)
    result = run_dry_run(cfg)
    assert result.steps_executed == 1
