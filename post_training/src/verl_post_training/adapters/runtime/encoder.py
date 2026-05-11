"""Runtime adapter for repo-owned video embedding inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...launch.dispatch import DispatchCompatibilityError, DispatchPlan, resolve_dispatch
from ...launch.load_config import TaskConfig
from ...registry import ModelFamily, RuntimeBackend, TaskType
from ..dataset import get_dataset_adapter
from ...vjepa2_inference import run_vjepa2_inference


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
        return run_vjepa2_inference(
            model_id=plan.config.model_id,
            task_type=plan.task_type.value,
            input_manifest=manifest_path,
            output_dir=resolved_output_dir,
            backend_config=plan.backend_config,
            **kwargs,
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


def _coerce_dispatch_plan(config: TaskConfig | DispatchPlan) -> DispatchPlan:
    if isinstance(config, DispatchPlan):
        return config
    return resolve_dispatch(config)

