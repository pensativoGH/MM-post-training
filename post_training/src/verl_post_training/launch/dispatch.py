"""Resolve validated task configs into registry-backed dispatch decisions."""

from __future__ import annotations

from dataclasses import dataclass

from .load_config import TaskConfig
from ..registry import (
    ModelRegistryEntry,
    RuntimeBackend,
    TaskType,
    TrainerBackend,
    get_model_entry,
)


class DispatchCompatibilityError(ValueError):
    """Raised when a validated config is incompatible with the registry."""


@dataclass(frozen=True)
class DispatchPlan:
    """Resolved dispatch inputs for a backend adapter or launcher."""

    config: TaskConfig
    model_entry: ModelRegistryEntry
    task_type: TaskType
    trainer_backend: TrainerBackend | None
    runtime_backend: RuntimeBackend | None
    backend_config: dict[str, object]


def resolve_dispatch(config: TaskConfig) -> DispatchPlan:
    """Validate config compatibility against the model registry."""

    model_entry = get_model_entry(config.model_id)
    if config.task_type not in model_entry.supported_task_types:
        raise DispatchCompatibilityError(
            "Unsupported task/backend combination: "
            f"model_family={model_entry.model_family.value}, "
            f"task_type={config.task_type.value}, "
            f"model_id={config.model_id} does not support that task."
        )

    if config.trainer_backend is not None:
        _ensure_supported_backend(
            model_entry=model_entry,
            backend=config.trainer_backend,
            supported_backends=model_entry.trainer_backends,
            backend_field="trainer_backend",
        )
    if config.runtime_backend is not None:
        _ensure_supported_backend(
            model_entry=model_entry,
            backend=config.runtime_backend,
            supported_backends=model_entry.runtime_backends,
            backend_field="runtime_backend",
        )

    return DispatchPlan(
        config=config,
        model_entry=model_entry,
        task_type=config.task_type,
        trainer_backend=config.trainer_backend,
        runtime_backend=config.runtime_backend,
        backend_config=dict(config.backend_config),
    )


def dispatch_config(config: TaskConfig) -> DispatchPlan:
    """Compatibility alias for config-to-dispatch resolution."""

    return resolve_dispatch(config)


def get_runtime_adapter(config: TaskConfig | DispatchPlan):
    """Compatibility selector for callers that resolve runtime adapters via launch."""

    from ..adapters.runtime import resolve_runtime_adapter

    return resolve_runtime_adapter(config)


def select_runtime_adapter(config: TaskConfig | DispatchPlan):
    """Compatibility alias for runtime adapter selection."""

    return get_runtime_adapter(config)


def runtime_adapter_for_plan(config: TaskConfig | DispatchPlan):
    """Compatibility alias for runtime adapter selection."""

    return get_runtime_adapter(config)


def build_runtime_adapter(config: TaskConfig | DispatchPlan):
    """Compatibility alias for runtime adapter selection."""

    return get_runtime_adapter(config)


def _ensure_supported_backend(
    *,
    model_entry: ModelRegistryEntry,
    backend: TrainerBackend | RuntimeBackend,
    supported_backends: tuple[TrainerBackend, ...] | tuple[RuntimeBackend, ...],
    backend_field: str,
) -> None:
    if backend in supported_backends:
        return

    supported = ", ".join(item.value for item in supported_backends) or "none"
    raise DispatchCompatibilityError(
        "Unsupported task/backend combination: "
        f"model_family={model_entry.model_family.value}, "
        f"{backend_field}={backend.value}, "
        f"model_id={model_entry.model_id}. Supported {backend_field} values: {supported}"
    )
