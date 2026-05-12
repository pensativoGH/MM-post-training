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
    _ensure_backend_task_role(config, model_entry)
    if config.task_type not in model_entry.supported_task_types:
        backend_field, backend_value = _requested_backend(config)
        raise DispatchCompatibilityError(
            "Unsupported task/backend combination: "
            f"model_family={model_entry.model_family.value}, "
            f"task_type={config.task_type.value}, "
            f"{backend_field}={backend_value}, "
            f"model_id={config.model_id} does not support that task."
        )

    if config.trainer_backend is not None:
        _ensure_supported_backend(
            model_entry=model_entry,
            task_type=config.task_type,
            backend=config.trainer_backend,
            supported_backends=model_entry.trainer_backends,
            backend_field="trainer_backend",
        )
    if config.runtime_backend is not None:
        _ensure_supported_backend(
            model_entry=model_entry,
            task_type=config.task_type,
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


dispatch = resolve_dispatch
resolve = resolve_dispatch
plan = resolve_dispatch
build_plan = resolve_dispatch


def get_trainer_adapter(config: TaskConfig | DispatchPlan):
    """Compatibility selector for callers that resolve trainer adapters via launch."""

    plan = config if isinstance(config, DispatchPlan) else resolve_dispatch(config)

    from ..adapters.trainer import resolve_trainer_adapter

    return resolve_trainer_adapter(plan)


def select_trainer_adapter(config: TaskConfig | DispatchPlan):
    """Compatibility alias for trainer adapter selection."""

    return get_trainer_adapter(config)


def resolve_trainer_adapter(config: TaskConfig | DispatchPlan):
    """Compatibility alias for trainer adapter selection."""

    return get_trainer_adapter(config)


def trainer_adapter_for_plan(config: TaskConfig | DispatchPlan):
    """Compatibility alias for trainer adapter selection."""

    return get_trainer_adapter(config)


def build_trainer_adapter(config: TaskConfig | DispatchPlan):
    """Compatibility alias for trainer adapter selection."""

    return get_trainer_adapter(config)


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


def _ensure_backend_task_role(config: TaskConfig, model_entry: ModelRegistryEntry) -> None:
    if config.trainer_backend == TrainerBackend.VJEPA2_NATIVE:
        if config.task_type != TaskType.MASKED_VIDEO_PREDICTION:
            raise DispatchCompatibilityError(
                "Unsupported task/backend combination: "
                f"model_family={model_entry.model_family.value}, "
                f"task_type={config.task_type.value}, "
                f"trainer_backend={TrainerBackend.VJEPA2_NATIVE.value} "
                f"requires task_type={TaskType.MASKED_VIDEO_PREDICTION.value}."
            )
    if (
        config.runtime_backend == RuntimeBackend.VJEPA2_NATIVE
        and config.task_type == TaskType.MASKED_VIDEO_PREDICTION
    ):
        raise DispatchCompatibilityError(
            "Unsupported task/backend combination: "
            f"model_family={model_entry.model_family.value}, "
            f"task_type={TaskType.MASKED_VIDEO_PREDICTION.value} must use "
            f"trainer_backend={TrainerBackend.VJEPA2_NATIVE.value}, not "
            f"runtime_backend={RuntimeBackend.VJEPA2_NATIVE.value}."
        )
    if config.trainer_backend == TrainerBackend.WAN_NATIVE:
        raise DispatchCompatibilityError(
            "Unsupported task/backend combination: "
            f"model_family={model_entry.model_family.value}, "
            f"task_type={config.task_type.value}, "
            f"trainer_backend={TrainerBackend.WAN_NATIVE.value}. "
            f"Wan2.2 support is inference-only; use runtime_backend="
            f"{RuntimeBackend.WAN_NATIVE.value}."
        )


def _requested_backend(config: TaskConfig) -> tuple[str, str]:
    if config.trainer_backend is not None:
        return "trainer_backend", config.trainer_backend.value
    if config.runtime_backend is not None:
        return "runtime_backend", config.runtime_backend.value
    return "backend", "none"


def _ensure_supported_backend(
    *,
    model_entry: ModelRegistryEntry,
    task_type: TaskType,
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
        f"task_type={task_type.value}, "
        f"{backend_field}={backend.value}, "
        f"model_id={model_entry.model_id}. Supported {backend_field} values: {supported}"
    )
