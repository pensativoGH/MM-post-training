"""Runtime adapter for repo-owned video embedding inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verl_post_training.adapters.dataset import get_dataset_adapter
from verl_post_training.launch.dispatch import (
    DispatchCompatibilityError,
    DispatchPlan,
    resolve_dispatch,
)
from verl_post_training.launch.load_config import TaskConfig
from verl_post_training.registry import ModelFamily, RuntimeBackend, TaskType
from verl_post_training_vjepa.inference import run_vjepa2_inference


class EncoderRuntimeAdapter:
    """Dispatch V-JEPA2 embedding inference through a repo-local wrapper."""

    adapter_key = RuntimeBackend.VJEPA2_NATIVE.value
    task_type = TaskType.EMBEDDING_INFERENCE

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
        upstream_result = _invoke_upstream(
            plan=plan,
            prepared_manifest=manifest_path,
            output_dir=resolved_output_dir,
            **kwargs,
        )
        return _normalize_encoder_envelope(
            plan=plan,
            prepared_manifest=manifest_path,
            output_dir=resolved_output_dir,
            upstream_result=upstream_result,
        )

    def validate(self, plan: DispatchPlan) -> None:
        if plan.task_type != self.task_type:
            raise DispatchCompatibilityError(
                f"Encoder runtime adapter only supports {self.task_type.value}."
            )
        if plan.runtime_backend != RuntimeBackend.VJEPA2_NATIVE:
            raise DispatchCompatibilityError(
                "Encoder runtime adapter requires runtime_backend=vjepa2_native."
            )
        if plan.model_entry.model_family != ModelFamily.VIDEO_ENCODER:
            raise DispatchCompatibilityError(
                "Encoder runtime adapter only supports video_encoder model entries."
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


def _invoke_upstream(
    *,
    plan: DispatchPlan,
    prepared_manifest: Path,
    output_dir: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the upstream V-JEPA2 wrapper.

    This named seam lets unit tests stub the backend process while keeping the
    production path subprocess-driven through ``run_vjepa2_inference``.
    """

    return run_vjepa2_inference(
        model_id=plan.config.model_id,
        task_type=plan.task_type.value,
        input_manifest=prepared_manifest,
        output_dir=output_dir,
        backend_config=plan.backend_config,
        **kwargs,
    )


def _normalize_encoder_envelope(
    *,
    plan: DispatchPlan,
    prepared_manifest: Path,
    output_dir: Path,
    upstream_result: Any,
) -> dict[str, Any]:
    envelope = dict(upstream_result) if isinstance(upstream_result, dict) else {}
    per_example = _coerce_per_example(envelope, prepared_manifest)

    envelope.setdefault("model_id", plan.config.model_id)
    envelope.setdefault("task_type", plan.task_type.value)
    envelope.setdefault("output_dir", str(output_dir))
    envelope.setdefault("input_manifest", str(prepared_manifest))
    envelope.setdefault("per_example", per_example)
    envelope.setdefault("results", per_example)
    envelope.setdefault("per_example_results", per_example)
    envelope.setdefault("examples", per_example)
    envelope.setdefault("outputs", per_example)
    envelope.setdefault("items", per_example)
    envelope.setdefault(
        "per_example_status",
        {str(item.get("example_id")): item.get("status", "success") for item in per_example},
    )
    return envelope


def _coerce_per_example(envelope: dict[str, Any], prepared_manifest: Path) -> list[dict[str, Any]]:
    for key in ("per_example", "examples", "results", "outputs", "items"):
        value = envelope.get(key)
        if isinstance(value, list) and value:
            return [dict(item) if isinstance(item, dict) else dict(vars(item)) for item in value]

    status = str(envelope.get("status") or "success").lower()
    if status == "ok":
        status = "success"

    results: list[dict[str, Any]] = []
    for index, line in enumerate(prepared_manifest.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        results.append(
            {
                "example_id": str(row.get("example_id") or f"example-{index:05d}"),
                "status": status,
                "video_path": str(row.get("video_path") or ""),
            }
        )
    return results
