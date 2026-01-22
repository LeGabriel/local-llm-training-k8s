"""LLM training utilities for local Kubernetes clusters."""

from importlib import metadata

__all__ = ["__version__"]

try:
    __version__ = metadata.version("local-llm-training-k8s")
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"
