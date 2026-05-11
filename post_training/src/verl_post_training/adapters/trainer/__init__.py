"""Trainer adapter registry."""

from __future__ import annotations

from ...launch.dispatch import DispatchCompatibilityError, DispatchPlan, resolve_dispatch
from ...launch.load_config import TaskConfig
from ...registry import TrainerBackend
from .base import TrainerAdapter
from .vjepa2 import VJEPA2TrainerAdapter


def _build_registry() -> dict[str, TrainerAdapter]:
    adapters: tuple[TrainerAdapter, ...] = (VJEPA2TrainerAdapter(),)
    return {adapter.adapter_key: adapter for adapter in adapters}


ADAPTER_REGISTRY: dict[str, TrainerAdapter] = _build_registry()
REGISTRY = ADAPTER_REGISTRY
TRAINER_ADAPTER_KEYS = tuple(sorted(ADAPTER_REGISTRY))


def get_trainer_adapter(adapter_key: str) -> TrainerAdapter:
    try:
        return ADAPTER_REGISTRY[adapter_key]
    except KeyError as exc:
        known = ", ".join(sorted(ADAPTER_REGISTRY)) or "none"
        raise LookupError(
            f"Unknown trainer adapter {adapter_key!r}. Registered adapters: {known}"
        ) from exc


def get_adapter(adapter_key: str) -> TrainerAdapter:
    return get_trainer_adapter(adapter_key)


def select_trainer_adapter(config: TaskConfig | DispatchPlan) -> TrainerAdapter:
    return resolve_trainer_adapter(config)


def trainer_adapter_for_plan(config: TaskConfig | DispatchPlan) -> TrainerAdapter:
    return resolve_trainer_adapter(config)


def build_trainer_adapter(config: TaskConfig | DispatchPlan) -> TrainerAdapter:
    return resolve_trainer_adapter(config)


def resolve_trainer_adapter(config: TaskConfig | DispatchPlan) -> TrainerAdapter:
    plan = config if isinstance(config, DispatchPlan) else resolve_dispatch(config)
    if plan.trainer_backend is None:
        raise DispatchCompatibilityError("Dispatch plan does not define a trainer_backend.")
    adapter = get_trainer_adapter(plan.trainer_backend.value)
    validate = getattr(adapter, "validate", None)
    if callable(validate):
        validate(plan)
    return adapter


def run_trainer(config: TaskConfig | DispatchPlan, **kwargs: object) -> dict[str, object]:
    plan = config if isinstance(config, DispatchPlan) else resolve_dispatch(config)
    adapter = resolve_trainer_adapter(plan)
    return adapter.run(plan, **kwargs)


def dispatch_trainer(config: TaskConfig | DispatchPlan, **kwargs: object) -> dict[str, object]:
    return run_trainer(config, **kwargs)


def lookup(adapter_key: str) -> TrainerAdapter:
    return get_trainer_adapter(adapter_key)


def resolve(adapter_key: str) -> TrainerAdapter:
    return get_trainer_adapter(adapter_key)


def iter_adapters() -> tuple[TrainerAdapter, ...]:
    return tuple(ADAPTER_REGISTRY.values())


def list_trainer_adapters() -> tuple[str, ...]:
    return TRAINER_ADAPTER_KEYS


__all__ = [
    "ADAPTER_REGISTRY",
    "DispatchCompatibilityError",
    "REGISTRY",
    "TRAINER_ADAPTER_KEYS",
    "TrainerAdapter",
    "VJEPA2TrainerAdapter",
    "build_trainer_adapter",
    "dispatch_trainer",
    "get_adapter",
    "get_trainer_adapter",
    "iter_adapters",
    "list_trainer_adapters",
    "lookup",
    "resolve",
    "resolve_trainer_adapter",
    "run_trainer",
    "select_trainer_adapter",
    "trainer_adapter_for_plan",
]
