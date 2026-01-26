"""Model adapter registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RegistryError(ValueError):
    """Raised when a registry operation fails."""


_MODEL_REGISTRY: dict[str, type[object]] = {}


def _normalize_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise RegistryError("Registry name must be non-empty.")
    return normalized


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    """Register a model adapter by name."""
    normalized = _normalize_name(name)

    def decorator(cls: type[T]) -> type[T]:
        if normalized in _MODEL_REGISTRY:
            available = ", ".join(sorted(_MODEL_REGISTRY)) or "none"
            raise RegistryError(
                f"Model adapter '{normalized}' is already registered. Available: {available}."
            )
        _MODEL_REGISTRY[normalized] = cls
        return cls

    return decorator


def get_model_adapter(name: str) -> type[object]:
    """Return the registered model adapter class for the given name."""
    normalized = _normalize_name(name)
    if normalized not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "none"
        raise RegistryError(f"Unknown model adapter '{normalized}'. Available: {available}.")
    return _MODEL_REGISTRY[normalized]


def available_model_adapters() -> list[str]:
    """Return sorted names of registered model adapters."""
    return sorted(_MODEL_REGISTRY)
