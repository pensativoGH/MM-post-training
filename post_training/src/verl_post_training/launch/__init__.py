"""Repo-level config loading and dispatch helpers."""

from .dispatch import (
    DispatchCompatibilityError,
    DispatchPlan,
    build_runtime_adapter,
    build_trainer_adapter,
    dispatch_config,
    get_runtime_adapter,
    get_trainer_adapter,
    resolve_dispatch,
    runtime_adapter_for_plan,
    select_trainer_adapter,
    select_runtime_adapter,
    trainer_adapter_for_plan,
)
from .load_config import ConfigValidationError, TaskConfig, load_task_config
from ..capabilities import (
    CapabilityUnavailableError,
    capabilities,
    describe_capabilities,
    get_capabilities,
    is_available,
    list_capabilities,
    report_capabilities,
)

__all__ = [
    "build_runtime_adapter",
    "build_trainer_adapter",
    "capabilities",
    "CapabilityUnavailableError",
    "ConfigValidationError",
    "describe_capabilities",
    "DispatchCompatibilityError",
    "DispatchPlan",
    "get_capabilities",
    "TaskConfig",
    "dispatch_config",
    "get_runtime_adapter",
    "get_trainer_adapter",
    "is_available",
    "list_capabilities",
    "load_task_config",
    "report_capabilities",
    "resolve_dispatch",
    "runtime_adapter_for_plan",
    "select_runtime_adapter",
    "select_trainer_adapter",
    "trainer_adapter_for_plan",
]
