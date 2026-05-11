"""Runtime adapter registry."""

from __future__ import annotations

from ...launch.dispatch import DispatchCompatibilityError, DispatchPlan, resolve_dispatch
from ...launch.load_config import TaskConfig
from ...registry import RuntimeBackend
from .base import RuntimeAdapter
from .encoder import EncoderRuntimeAdapter


def _build_registry() -> dict[str, RuntimeAdapter]:
    adapters: tuple[RuntimeAdapter, ...] = (EncoderRuntimeAdapter(),)
    return {adapter.adapter_key: adapter for adapter in adapters}


ADAPTER_REGISTRY: dict[str, RuntimeAdapter] = _build_registry()
REGISTRY = ADAPTER_REGISTRY
RUNTIME_ADAPTER_KEYS = tuple(sorted(ADAPTER_REGISTRY))


def get_runtime_adapter(adapter_key: str) -> RuntimeAdapter:
    try:
        return ADAPTER_REGISTRY[adapter_key]
    except KeyError as exc:
        known = ", ".join(sorted(ADAPTER_REGISTRY)) or "none"
        raise LookupError(
            f"Unknown runtime adapter {adapter_key!r}. Registered adapters: {known}"
        ) from exc


def get_adapter(adapter_key: str) -> RuntimeAdapter:
    return get_runtime_adapter(adapter_key)


def select_runtime_adapter(
    config: TaskConfig | DispatchPlan,
) -> RuntimeAdapter:
    return resolve_runtime_adapter(config)


def runtime_adapter_for_plan(
    config: TaskConfig | DispatchPlan,
) -> RuntimeAdapter:
    return resolve_runtime_adapter(config)


def build_runtime_adapter(
    config: TaskConfig | DispatchPlan,
) -> RuntimeAdapter:
    return resolve_runtime_adapter(config)


def resolve_runtime_adapter(
    config: TaskConfig | DispatchPlan,
) -> RuntimeAdapter:
    plan = config if isinstance(config, DispatchPlan) else resolve_dispatch(config)
    if plan.runtime_backend is None:
        raise DispatchCompatibilityError("Dispatch plan does not define a runtime_backend.")
    adapter = get_runtime_adapter(plan.runtime_backend.value)
    validate = getattr(adapter, "validate", None)
    if callable(validate):
        validate(plan)
    return adapter


def run_runtime(
    config: TaskConfig | DispatchPlan,
    **kwargs: object,
) -> dict[str, object]:
    plan = config if isinstance(config, DispatchPlan) else resolve_dispatch(config)
    adapter = resolve_runtime_adapter(plan)
    return adapter.run(plan, **kwargs)


def dispatch_runtime(
    config: TaskConfig | DispatchPlan,
    **kwargs: object,
) -> dict[str, object]:
    return run_runtime(config, **kwargs)


def lookup(adapter_key: str) -> RuntimeAdapter:
    return get_runtime_adapter(adapter_key)


def resolve(adapter_key: str) -> RuntimeAdapter:
    return get_runtime_adapter(adapter_key)


def iter_adapters() -> tuple[RuntimeAdapter, ...]:
    return tuple(ADAPTER_REGISTRY.values())


def list_runtime_adapters() -> tuple[str, ...]:
    return RUNTIME_ADAPTER_KEYS


__all__ = [
    "ADAPTER_REGISTRY",
    "DispatchCompatibilityError",
    "EncoderRuntimeAdapter",
    "REGISTRY",
    "RUNTIME_ADAPTER_KEYS",
    "RuntimeAdapter",
    "dispatch_runtime",
    "get_adapter",
    "get_runtime_adapter",
    "iter_adapters",
    "list_runtime_adapters",
    "lookup",
    "resolve",
    "select_runtime_adapter",
    "runtime_adapter_for_plan",
    "resolve_runtime_adapter",
    "run_runtime",
    "build_runtime_adapter",
]
