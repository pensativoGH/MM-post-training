"""Adapter exports for repo-owned dataset, trainer, and runtime integrations."""

from .dataset import (
    ADAPTER_REGISTRY,
    REGISTRY,
    entries,
    get_adapter,
    get_dataset_adapter,
    iter_adapters,
    iter_entries,
    lookup,
    resolve,
)

__all__ = [
    "ADAPTER_REGISTRY",
    "REGISTRY",
    "entries",
    "get_adapter",
    "get_dataset_adapter",
    "iter_adapters",
    "iter_entries",
    "lookup",
    "resolve",
]
