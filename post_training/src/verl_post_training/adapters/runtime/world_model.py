"""Runtime adapter for normalized DreamDojo world-model rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...dreamdojo_runtime import run_dreamdojo_rollout
from ...launch.dispatch import DispatchCompatibilityError, DispatchPlan, resolve_dispatch
from ...launch.load_config import TaskConfig
from ...registry import ModelFamily, RuntimeBackend, TaskType
from ..dataset import get_dataset_adapter


class WorldModelRuntimeAdapter:
    """Prepare DreamDojo rollout inputs and report capability status."""

    adapter_key = RuntimeBackend.DREAMDOJO.value
    task_type = TaskType.WORLD_MODEL_ROLLOUT

    def run(
        self,
        config: TaskConfig | DispatchPlan,
        *,
        pipeline_manifest: Any = None,
        output_dir: Path | None = None,
        split: str = "inference",
        **kwargs: Any,
    ) -> dict[str, Any]:
        plan = _coerce_dispatch_plan(config)
        self.validate(plan)

        dataset_adapter = get_dataset_adapter(plan.config.dataset_adapter)
        resolved_output_dir = Path(output_dir or plan.config.output_dir)
        manifest_path = dataset_adapter.prepare(
            pipeline_manifest=pipeline_manifest or plan.config.input_manifest,
            output_dir=resolved_output_dir / "inputs",
            split=split,
            config=plan.backend_config,
        )
        upstream_result = run_dreamdojo_rollout(
            model_id=plan.config.model_id,
            task_type=plan.task_type.value,
            input_manifest=manifest_path,
            output_dir=resolved_output_dir,
            backend_config=plan.backend_config,
            split=split,
            **kwargs,
        )
        envelope = _normalize_world_model_envelope(
            plan=plan,
            prepared_manifest=manifest_path,
            output_dir=resolved_output_dir,
            upstream_result=upstream_result,
        )
        _write_capability_report(envelope=envelope, output_dir=resolved_output_dir)
        return envelope

    def validate(self, plan: DispatchPlan) -> None:
        if plan.task_type != self.task_type:
            raise DispatchCompatibilityError(
                f"World-model runtime adapter only supports task_type={self.task_type.value}; "
                f"got task_type={plan.task_type.value}."
            )
        if plan.runtime_backend != RuntimeBackend.DREAMDOJO:
            backend = plan.runtime_backend.value if plan.runtime_backend else None
            raise DispatchCompatibilityError(
                f"World-model runtime adapter requires runtime_backend={RuntimeBackend.DREAMDOJO.value}; "
                f"got {backend}."
            )
        if plan.model_entry.model_family != ModelFamily.WORLD_MODEL:
            raise DispatchCompatibilityError(
                "World-model runtime adapter only supports world_model registry entries."
            )

    def invoke(self, config: TaskConfig | DispatchPlan, **kwargs: Any) -> dict[str, Any]:
        return self.run(config, **kwargs)

    def execute(self, config: TaskConfig | DispatchPlan, **kwargs: Any) -> dict[str, Any]:
        return self.run(config, **kwargs)

    def __call__(self, config: TaskConfig | DispatchPlan, **kwargs: Any) -> dict[str, Any]:
        return self.run(config, **kwargs)


def _coerce_dispatch_plan(config: TaskConfig | DispatchPlan) -> DispatchPlan:
    if isinstance(config, DispatchPlan):
        return config
    return resolve_dispatch(config)


def _normalize_world_model_envelope(
    *,
    plan: DispatchPlan,
    prepared_manifest: Path,
    output_dir: Path,
    upstream_result: Any,
) -> dict[str, Any]:
    envelope = dict(upstream_result) if isinstance(upstream_result, dict) else {}
    capability = dict(envelope.get("capability") or {})
    capability.setdefault("available", False)
    capability.setdefault(
        "reason",
        "DreamDojo execution is deferred pending local-first integration approval.",
    )

    envelope.setdefault("status", "unavailable")
    envelope.setdefault("available", False)
    envelope.setdefault("model_id", plan.config.model_id)
    envelope.setdefault("task_type", plan.task_type.value)
    envelope.setdefault("runtime_backend", RuntimeBackend.DREAMDOJO.value)
    envelope.setdefault("output_dir", str(output_dir))
    envelope.setdefault("input_manifest", str(prepared_manifest))
    envelope["capability"] = capability
    envelope["capability_available"] = bool(capability.get("available"))
    return envelope


def _write_capability_report(*, envelope: dict[str, Any], output_dir: Path) -> None:
    path = output_dir / "dreamdojo_capability.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    envelope["capability_report_path"] = str(path)
    envelope["result_path"] = str(path)


DreamDojoRuntimeAdapter = WorldModelRuntimeAdapter
Adapter = WorldModelRuntimeAdapter
RuntimeAdapter = WorldModelRuntimeAdapter
adapter = WorldModelRuntimeAdapter()


__all__ = [
    "Adapter",
    "DreamDojoRuntimeAdapter",
    "RuntimeAdapter",
    "WorldModelRuntimeAdapter",
    "adapter",
]
