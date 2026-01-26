from __future__ import annotations

import pytest

from llmtrain.registry import data as data_registry
from llmtrain.registry import models as model_registry


def test_model_registry_register_and_lookup() -> None:
    name = "test-model-registry"

    @model_registry.register_model(name)
    class DemoModel:
        pass

    assert model_registry.get_model_adapter(name) is DemoModel
    assert name in model_registry.available_model_adapters()


def test_model_registry_duplicate_name_raises() -> None:
    name = "test-model-registry-dup"

    @model_registry.register_model(name)
    class FirstModel:
        pass

    with pytest.raises(model_registry.RegistryError):

        @model_registry.register_model(name)
        class SecondModel:
            pass


def test_model_registry_unknown_name_mentions_available() -> None:
    registered = "test-model-registry-known"
    missing = "test-model-registry-missing"

    @model_registry.register_model(registered)
    class KnownModel:
        pass

    with pytest.raises(model_registry.RegistryError) as excinfo:
        model_registry.get_model_adapter(missing)

    message = str(excinfo.value)
    assert registered in message
    assert missing in message
    assert "available" in message.lower()


def test_data_registry_register_and_lookup() -> None:
    name = "test-data-registry"

    @data_registry.register_data_module(name)
    class DemoData:
        pass

    assert data_registry.get_data_module(name) is DemoData
    assert name in data_registry.available_data_modules()


def test_data_registry_duplicate_name_raises() -> None:
    name = "test-data-registry-dup"

    @data_registry.register_data_module(name)
    class FirstData:
        pass

    with pytest.raises(data_registry.RegistryError):

        @data_registry.register_data_module(name)
        class SecondData:
            pass


def test_data_registry_unknown_name_mentions_available() -> None:
    registered = "test-data-registry-known"
    missing = "test-data-registry-missing"

    @data_registry.register_data_module(registered)
    class KnownData:
        pass

    with pytest.raises(data_registry.RegistryError) as excinfo:
        data_registry.get_data_module(missing)

    message = str(excinfo.value)
    assert registered in message
    assert missing in message
    assert "available" in message.lower()
