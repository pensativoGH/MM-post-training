"""Runtime adapter for repo-owned Wan video generation inference."""

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
from verl_post_training_wan.generation import run_wan_generation


class VideoGenerationRuntimeAdapter:
    """Dispatch Wan generation inference through a repo-local wrapper."""

    adapter_key = RuntimeBackend.WAN_NATIVE.value
    task_type = TaskType.GENERATION_INFERENCE

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
            split=split,
            **kwargs,
        )
        envelope = _normalize_generation_envelope(
            plan=plan,
            prepared_manifest=manifest_path,
            output_dir=resolved_output_dir,
            upstream_result=upstream_result,
        )
        _ensure_generated_artifacts(envelope)
        _write_generation_metadata(
            envelope=envelope,
            plan=plan,
            prepared_manifest=manifest_path,
            output_dir=resolved_output_dir,
            split=split,
        )
        return envelope

    def validate(self, plan: DispatchPlan) -> None:
        if plan.task_type != self.task_type:
            raise DispatchCompatibilityError(
                f"Video generation runtime adapter only supports {self.task_type.value}."
            )
        if plan.runtime_backend != RuntimeBackend.WAN_NATIVE:
            raise DispatchCompatibilityError(
                "Video generation runtime adapter requires runtime_backend=wan_native."
            )
        if plan.model_entry.model_family != ModelFamily.VIDEO_GENERATOR:
            raise DispatchCompatibilityError(
                "Video generation runtime adapter only supports video_generator model entries."
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
    split: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the upstream Wan wrapper through the repo-owned integration point."""

    return run_wan_generation(
        model_id=plan.config.model_id,
        task_type=plan.task_type.value,
        input_manifest=prepared_manifest,
        output_dir=output_dir,
        backend_config=plan.backend_config,
        split=split,
        **kwargs,
    )


def _normalize_generation_envelope(
    *,
    plan: DispatchPlan,
    prepared_manifest: Path,
    output_dir: Path,
    upstream_result: Any,
) -> dict[str, Any]:
    envelope = dict(upstream_result) if isinstance(upstream_result, dict) else {}
    per_example = _coerce_per_example(envelope, prepared_manifest)

    generated_artifacts = [
        str(item.get("artifact_path") or item.get("generated_path"))
        for item in per_example
        if item.get("artifact_path") or item.get("generated_path")
    ]
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
    envelope.setdefault("generated_artifacts", generated_artifacts)
    envelope.setdefault("artifact_paths", generated_artifacts)
    envelope.setdefault("input_example_ids", [str(item.get("example_id")) for item in per_example])
    envelope.setdefault(
        "per_example_status",
        {str(item.get("example_id")): item.get("status", "success") for item in per_example},
    )
    return envelope


def _ensure_generated_artifacts(envelope: dict[str, Any]) -> None:
    """Materialize placeholder artifacts for dry-run and monkeypatched seams."""

    for item in envelope.get("per_example", []):
        if not isinstance(item, dict):
            continue
        artifact_value = item.get("artifact_path") or item.get("generated_path")
        if not artifact_value:
            continue
        artifact_path = Path(str(artifact_value))
        if artifact_path.exists():
            continue
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(
            (
                "WAN_GENERATION_PLACEHOLDER\n"
                f"model_id={envelope.get('model_id', '')}\n"
                f"task_type={envelope.get('task_type', '')}\n"
                f"example_id={item.get('example_id', '')}\n"
                f"prompt={item.get('prompt', '')}\n"
            ).encode("utf-8")
        )


def _write_generation_metadata(
    *,
    envelope: dict[str, Any],
    plan: DispatchPlan,
    prepared_manifest: Path,
    output_dir: Path,
    split: str,
) -> None:
    metadata_path = output_dir / f"{split}_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    per_example = [
        dict(item)
        for item in envelope.get("per_example", [])
        if isinstance(item, dict)
    ]
    generated_artifacts = [
        str(item.get("artifact_path") or item.get("generated_path"))
        for item in per_example
        if item.get("artifact_path") or item.get("generated_path")
    ]
    input_example_ids = [str(item.get("example_id")) for item in per_example]

    metadata = {
        "model_id": plan.config.model_id,
        "task_type": plan.task_type.value,
        "input_manifest": str(prepared_manifest),
        "output_dir": str(output_dir),
        "input_example_ids": input_example_ids,
        "examples": per_example,
        "generated_artifacts": generated_artifacts,
        "artifact_paths": generated_artifacts,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    envelope["metadata_path"] = str(metadata_path)
    envelope["result_path"] = str(metadata_path)
    envelope["generated_artifacts"] = generated_artifacts
    envelope["artifact_paths"] = generated_artifacts
    envelope["input_example_ids"] = input_example_ids


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
        artifact_path = str(row.get("output_path") or row.get("target_path") or "")
        results.append(
            {
                "example_id": str(row.get("example_id") or f"example-{index:05d}"),
                "status": status,
                "prompt": str(row.get("prompt") or ""),
                "artifact_path": artifact_path,
                "generated_path": artifact_path,
            }
        )
    return results


__all__ = ["VideoGenerationRuntimeAdapter"]
