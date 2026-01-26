"""Model and component registries for llmtrain."""

from __future__ import annotations

from importlib import import_module

MODEL_REGISTRY_MODULES: tuple[str, ...] = ()
DATA_REGISTRY_MODULES: tuple[str, ...] = ()


def initialize_registries() -> None:
    """Import known plugin modules to populate registries deterministically."""
    for module_name in MODEL_REGISTRY_MODULES + DATA_REGISTRY_MODULES:
        import_module(module_name)
