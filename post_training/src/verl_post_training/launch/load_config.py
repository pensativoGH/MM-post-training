"""Load and validate repo-level task configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from verl_post_training.registry import RuntimeBackend, TaskType, TrainerBackend


class ConfigValidationError(ValueError):
    """Raised when a repo-level task config fails validation."""


@dataclass(frozen=True)
class TaskConfig:
    """Normalized repo-level config for launch dispatch."""

    task_type: TaskType
    model_id: str
    dataset_adapter: str
    input_manifest: str
    output_dir: str
    launcher: dict[str, Any]
    resources: dict[str, Any]
    trainer_backend: TrainerBackend | None = None
    runtime_backend: RuntimeBackend | None = None
    backend_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> "TaskConfig":
        if not isinstance(raw_config, Mapping):
            raise ConfigValidationError(
                "Task config must deserialize to a mapping at the top level."
            )

        config = dict(raw_config)
        required_fields = (
            "task_type",
            "model_id",
            "dataset_adapter",
            "input_manifest",
            "output_dir",
            "launcher",
            "resources",
        )
        missing = [field for field in required_fields if field not in config]
        if missing:
            raise ConfigValidationError(
                f"Task config is missing required field(s): {', '.join(missing)}"
            )

        trainer_backend = _coerce_optional_enum(
            config.get("trainer_backend"),
            TrainerBackend,
            "trainer_backend",
        )
        runtime_backend = _coerce_optional_enum(
            config.get("runtime_backend"),
            RuntimeBackend,
            "runtime_backend",
        )
        if (trainer_backend is None) == (runtime_backend is None):
            raise ConfigValidationError(
                "Task config must define exactly one of `trainer_backend` or "
                "`runtime_backend`."
            )

        launcher = _require_mapping(config["launcher"], "launcher")
        resources = _require_mapping(config["resources"], "resources")

        backend_config_raw = config.get("backend_config", {})
        backend_config = _require_mapping(backend_config_raw, "backend_config")

        return cls(
            task_type=_coerce_enum(config["task_type"], TaskType, "task_type"),
            model_id=_require_non_empty_string(config["model_id"], "model_id"),
            dataset_adapter=_require_non_empty_string(
                config["dataset_adapter"], "dataset_adapter"
            ),
            input_manifest=_require_non_empty_string(
                config["input_manifest"], "input_manifest"
            ),
            output_dir=_require_non_empty_string(config["output_dir"], "output_dir"),
            launcher=launcher,
            resources=resources,
            trainer_backend=trainer_backend,
            runtime_backend=runtime_backend,
            backend_config=backend_config,
        )


def load_task_config(path: str | Path) -> TaskConfig:
    """Load a YAML task config from disk."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    return TaskConfig.from_mapping(raw_config)


def _coerce_optional_enum(
    value: Any,
    enum_cls: type[TaskType] | type[TrainerBackend] | type[RuntimeBackend],
    field_name: str,
):
    if value is None:
        return None
    return _coerce_enum(value, enum_cls, field_name)


def _coerce_enum(
    value: Any,
    enum_cls: type[TaskType] | type[TrainerBackend] | type[RuntimeBackend],
    field_name: str,
):
    try:
        return enum_cls(value)
    except ValueError as exc:
        valid_values = ", ".join(member.value for member in enum_cls)
        raise ConfigValidationError(
            f"Invalid `{field_name}` value {value!r}. Expected one of: {valid_values}"
        ) from exc


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"`{field_name}` must be a mapping.")
    return dict(value)


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"`{field_name}` must be a non-empty string.")
    return value
