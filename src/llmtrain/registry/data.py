"""Data module registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RegistryError(ValueError):
    """Raised when a registry operation fails."""


_DATA_REGISTRY: dict[str, type[object]] = {}


def _normalize_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise RegistryError("Registry name must be non-empty.")
    return normalized


def register_data_module(name: str) -> Callable[[type[T]], type[T]]:
    """Register a data module by name."""
    normalized = _normalize_name(name)

    def decorator(cls: type[T]) -> type[T]:
        if normalized in _DATA_REGISTRY:
            available = ", ".join(sorted(_DATA_REGISTRY)) or "none"
            raise RegistryError(
                f"Data module '{normalized}' is already registered. Available: {available}."
            )
        _DATA_REGISTRY[normalized] = cls
        return cls

    return decorator


def get_data_module(name: str) -> type[object]:
    """Return the registered data module class for the given name."""
    normalized = _normalize_name(name)
    if normalized not in _DATA_REGISTRY:
        available = ", ".join(sorted(_DATA_REGISTRY)) or "none"
        raise RegistryError(f"Unknown data module '{normalized}'. Available: {available}.")
    return _DATA_REGISTRY[normalized]


def available_data_modules() -> list[str]:
    """Return sorted names of registered data modules."""
    return sorted(_DATA_REGISTRY)
