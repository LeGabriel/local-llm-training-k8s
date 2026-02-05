from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from llmtrain.config.schemas import RunConfig
from llmtrain.data.base import DataModule
from llmtrain.models.base import ModelAdapter
from llmtrain.registry import data as data_registry
from llmtrain.registry import models as model_registry


def test_model_registry_register_and_lookup() -> None:
    name = "test-model-registry"

    @model_registry.register_model(name)
    class DemoModel(ModelAdapter):
        def build_model(self, cfg: RunConfig) -> torch.nn.Module:
            del cfg
            return torch.nn.Linear(1, 1)

        def build_tokenizer(self, cfg: RunConfig) -> None:
            del cfg
            return None

        def compute_loss(
            self, model: torch.nn.Module, batch: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, float]]:
            del model
            del batch
            loss = torch.tensor(0.0)
            return loss, {"loss": 0.0}

    assert model_registry.get_model_adapter(name) is DemoModel
    assert name in model_registry.available_model_adapters()


def test_model_registry_duplicate_name_raises() -> None:
    name = "test-model-registry-dup"

    @model_registry.register_model(name)
    class FirstModel(ModelAdapter):
        def build_model(self, cfg: RunConfig) -> torch.nn.Module:
            del cfg
            return torch.nn.Linear(1, 1)

        def build_tokenizer(self, cfg: RunConfig) -> None:
            del cfg
            return None

        def compute_loss(
            self, model: torch.nn.Module, batch: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, float]]:
            del model
            del batch
            loss = torch.tensor(0.0)
            return loss, {"loss": 0.0}

    with pytest.raises(model_registry.RegistryError):

        @model_registry.register_model(name)
        class SecondModel(ModelAdapter):
            def build_model(self, cfg: RunConfig) -> torch.nn.Module:
                del cfg
                return torch.nn.Linear(1, 1)

            def build_tokenizer(self, cfg: RunConfig) -> None:
                del cfg
                return None

            def compute_loss(
                self, model: torch.nn.Module, batch: dict[str, torch.Tensor]
            ) -> tuple[torch.Tensor, dict[str, float]]:
                del model
                del batch
                loss = torch.tensor(0.0)
                return loss, {"loss": 0.0}


def test_model_registry_unknown_name_mentions_available() -> None:
    registered = "test-model-registry-known"
    missing = "test-model-registry-missing"

    @model_registry.register_model(registered)
    class KnownModel(ModelAdapter):
        def build_model(self, cfg: RunConfig) -> torch.nn.Module:
            del cfg
            return torch.nn.Linear(1, 1)

        def build_tokenizer(self, cfg: RunConfig) -> None:
            del cfg
            return None

        def compute_loss(
            self, model: torch.nn.Module, batch: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, float]]:
            del model
            del batch
            loss = torch.tensor(0.0)
            return loss, {"loss": 0.0}

    with pytest.raises(model_registry.RegistryError) as excinfo:
        model_registry.get_model_adapter(missing)

    message = str(excinfo.value)
    assert registered in message
    assert missing in message
    assert "available" in message.lower()


def test_data_registry_register_and_lookup() -> None:
    name = "test-data-registry"

    @data_registry.register_data_module(name)
    class DemoData(DataModule):
        def setup(self, cfg: RunConfig, tokenizer: object | None = None) -> None:
            del cfg
            del tokenizer

        def train_dataloader(self) -> DataLoader:
            dataset = TensorDataset(torch.zeros(0, 1))
            return DataLoader(dataset)

        def val_dataloader(self) -> DataLoader | None:
            return None

    assert data_registry.get_data_module(name) is DemoData
    assert name in data_registry.available_data_modules()


def test_data_registry_duplicate_name_raises() -> None:
    name = "test-data-registry-dup"

    @data_registry.register_data_module(name)
    class FirstData(DataModule):
        def setup(self, cfg: RunConfig, tokenizer: object | None = None) -> None:
            del cfg
            del tokenizer

        def train_dataloader(self) -> DataLoader:
            dataset = TensorDataset(torch.zeros(0, 1))
            return DataLoader(dataset)

        def val_dataloader(self) -> DataLoader | None:
            return None

    with pytest.raises(data_registry.RegistryError):

        @data_registry.register_data_module(name)
        class SecondData(DataModule):
            def setup(self, cfg: RunConfig, tokenizer: object | None = None) -> None:
                del cfg
                del tokenizer

            def train_dataloader(self) -> DataLoader:
                dataset = TensorDataset(torch.zeros(0, 1))
                return DataLoader(dataset)

            def val_dataloader(self) -> DataLoader | None:
                return None


def test_data_registry_unknown_name_mentions_available() -> None:
    registered = "test-data-registry-known"
    missing = "test-data-registry-missing"

    @data_registry.register_data_module(registered)
    class KnownData(DataModule):
        def setup(self, cfg: RunConfig, tokenizer: object | None = None) -> None:
            del cfg
            del tokenizer

        def train_dataloader(self) -> DataLoader:
            dataset = TensorDataset(torch.zeros(0, 1))
            return DataLoader(dataset)

        def val_dataloader(self) -> DataLoader | None:
            return None

    with pytest.raises(data_registry.RegistryError) as excinfo:
        data_registry.get_data_module(missing)

    message = str(excinfo.value)
    assert registered in message
    assert missing in message
    assert "available" in message.lower()
