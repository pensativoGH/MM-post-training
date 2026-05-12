"""Dataset adapter registry."""

from .base import DatasetAdapter
from .chat_rl import ChatRLDatasetAdapter
from .chat_sft import ChatSFTDatasetAdapter
from .dreamdojo import DreamDojoDatasetAdapter
from .vjepa2 import VJEPA2DatasetAdapter
from .wan import WanDatasetAdapter


def _build_registry() -> dict[str, DatasetAdapter]:
    adapters: tuple[DatasetAdapter, ...] = (
        ChatSFTDatasetAdapter(),
        ChatRLDatasetAdapter(),
        VJEPA2DatasetAdapter(),
        WanDatasetAdapter(),
        DreamDojoDatasetAdapter(),
    )
    return {adapter.adapter_key: adapter for adapter in adapters}


ADAPTER_REGISTRY: dict[str, DatasetAdapter] = _build_registry()
REGISTRY = ADAPTER_REGISTRY
DATASET_ADAPTER_KEYS = tuple(sorted(ADAPTER_REGISTRY))


def get_dataset_adapter(adapter_key: str) -> DatasetAdapter:
    try:
        return ADAPTER_REGISTRY[adapter_key]
    except KeyError as exc:
        known = ", ".join(sorted(ADAPTER_REGISTRY)) or "none"
        raise LookupError(
            f"Unknown dataset adapter {adapter_key!r}. Registered adapters: {known}"
        ) from exc


def get_adapter(adapter_key: str) -> DatasetAdapter:
    return get_dataset_adapter(adapter_key)


def lookup(adapter_key: str) -> DatasetAdapter:
    return get_dataset_adapter(adapter_key)


def resolve(adapter_key: str) -> DatasetAdapter:
    return get_dataset_adapter(adapter_key)


def iter_adapters() -> tuple[DatasetAdapter, ...]:
    return tuple(ADAPTER_REGISTRY.values())


def iter_entries() -> tuple[DatasetAdapter, ...]:
    return iter_adapters()


def entries() -> tuple[DatasetAdapter, ...]:
    return iter_adapters()


def list_dataset_adapters() -> tuple[str, ...]:
    return DATASET_ADAPTER_KEYS


__all__ = [
    "ADAPTER_REGISTRY",
    "DATASET_ADAPTER_KEYS",
    "DatasetAdapter",
    "REGISTRY",
    "entries",
    "get_adapter",
    "get_dataset_adapter",
    "iter_adapters",
    "iter_entries",
    "list_dataset_adapters",
    "lookup",
    "resolve",
]
