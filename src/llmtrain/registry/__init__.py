"""Model and component registries for llmtrain."""

from __future__ import annotations

from importlib import import_module

MODEL_REGISTRY_MODULES: tuple[str, ...] = (
    "llmtrain.models.dummy_gpt",
    "llmtrain.models.gpt",
)
DATA_REGISTRY_MODULES: tuple[str, ...] = (
    "llmtrain.data.dummy_text",
    "llmtrain.data.hf_text",
)


def initialize_registries() -> None:
    """Import known plugin modules to populate registries deterministically."""
    for module_name in MODEL_REGISTRY_MODULES + DATA_REGISTRY_MODULES:
        import_module(module_name)
