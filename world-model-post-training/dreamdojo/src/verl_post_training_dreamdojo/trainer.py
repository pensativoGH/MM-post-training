"""Capability-only trainer adapter for DreamDojo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verl_post_training.launch.dispatch import (
    DispatchCompatibilityError,
    DispatchPlan,
    resolve_dispatch,
)
from verl_post_training.launch.load_config import TaskConfig
from verl_post_training.registry import ModelFamily, TrainerBackend


class DreamDojoTrainerAdapter:
    adapter_key = TrainerBackend.DREAMDOJO.value

    def run(
        self,
        config: TaskConfig | DispatchPlan,
        *,
        output_dir: Path | None = None,
        split: str = "train",
        **_: Any,
    ) -> dict[str, Any]:
        plan = _coerce_dispatch_plan(config)
        self.validate(plan)

        output_root = Path(output_dir or plan.config.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        record = {
            "status": "unavailable",
            "available": False,
            "model_id": plan.config.model_id,
            "model_family": plan.model_entry.model_family.value,
            "task_type": plan.task_type.value,
            "trainer_backend": TrainerBackend.DREAMDOJO.value,
            "capability": {
                "available": False,
                "reason": "DreamDojo training execution is deferred pending local-first integration approval.",
            },
            "output_dir": str(output_root),
            "split": split,
        }
        record_path = output_root / "dreamdojo_training_capability.json"
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        record["capability_report_path"] = str(record_path)
        record["result_path"] = str(record_path)
        return record

    def validate(self, plan: DispatchPlan) -> None:
        if plan.trainer_backend != TrainerBackend.DREAMDOJO:
            backend = plan.trainer_backend.value if plan.trainer_backend else None
            raise DispatchCompatibilityError(
                f"DreamDojo trainer adapter requires model_family={ModelFamily.WORLD_MODEL.value}, "
                f"task_type={plan.task_type.value}, trainer_backend={TrainerBackend.DREAMDOJO.value}; "
                f"got {backend}."
            )
        if plan.model_entry.model_family != ModelFamily.WORLD_MODEL:
            raise DispatchCompatibilityError(
                "DreamDojo trainer adapter only supports "
                f"model_family={ModelFamily.WORLD_MODEL.value}; got "
                f"model_family={plan.model_entry.model_family.value}, "
                f"task_type={plan.task_type.value}, "
                f"trainer_backend={TrainerBackend.DREAMDOJO.value}."
            )

    def launch(self, config: TaskConfig | DispatchPlan, **kwargs: Any) -> dict[str, Any]:
        return self.run(config, **kwargs)

    def dry_run(self, config: TaskConfig | DispatchPlan, **kwargs: Any) -> dict[str, Any]:
        return self.run(config, **kwargs)

    def prepare(self, config: TaskConfig | DispatchPlan, **kwargs: Any) -> dict[str, Any]:
        return self.run(config, **kwargs)

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


Adapter = DreamDojoTrainerAdapter
TrainerAdapter = DreamDojoTrainerAdapter
adapter = DreamDojoTrainerAdapter()


__all__ = ["Adapter", "DreamDojoTrainerAdapter", "TrainerAdapter", "adapter"]
