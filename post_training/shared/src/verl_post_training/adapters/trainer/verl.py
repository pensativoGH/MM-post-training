"""Trainer adapter for Qwen chat RL through VERL."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ...launch.dispatch import DispatchCompatibilityError, DispatchPlan, resolve_dispatch
from ...launch.load_config import TaskConfig
from ...registry import ModelFamily, TaskType, TrainerBackend


class VERLTrainerAdapter:
    adapter_key = TrainerBackend.VERL.value
    task_type = TaskType.CHAT_RL

    def run(
        self,
        config: TaskConfig | DispatchPlan,
        *,
        output_dir: Path | None = None,
        split: str = "train",
        dry_run: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        plan = _coerce_dispatch_plan(config)
        self.validate(plan)

        output_root = Path(output_dir or plan.config.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        backend_config = dict(plan.backend_config)
        backend_config.update(kwargs)
        config_file = Path(str(backend_config.get("config_file") or plan.config.input_manifest))
        argv = build_verl_argv(config_file=config_file, backend_config=backend_config)
        record = {
            "model_id": plan.config.model_id,
            "model_family": plan.model_entry.model_family.value,
            "task_type": plan.task_type.value,
            "trainer_backend": TrainerBackend.VERL.value,
            "dataset_adapter": plan.config.dataset_adapter,
            "input_manifest": plan.config.input_manifest,
            "config_file": str(config_file),
            "output_dir": str(output_root),
            "backend_config": backend_config,
            "argument_list": argv,
            "argv": argv,
            "launcher": dict(plan.config.launcher),
            "resources": dict(plan.config.resources),
            "dry_run": dry_run,
            "split": split,
        }
        record_path = output_root / "verl_training_launch.json"
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        record["launch_record_path"] = str(record_path)
        return record

    def validate(self, plan: DispatchPlan) -> None:
        if plan.task_type != self.task_type:
            raise DispatchCompatibilityError(
                "VERL trainer adapter only supports "
                f"model_family={plan.model_entry.model_family.value}, "
                f"task_type={self.task_type.value}, "
                f"trainer_backend={TrainerBackend.VERL.value}; "
                f"requested task_type={plan.task_type.value}."
            )
        if plan.trainer_backend != TrainerBackend.VERL:
            backend = plan.trainer_backend.value if plan.trainer_backend else None
            raise DispatchCompatibilityError(
                "VERL trainer adapter requires "
                f"model_family={plan.model_entry.model_family.value}, "
                f"task_type={plan.task_type.value}, "
                f"trainer_backend={TrainerBackend.VERL.value}; got {backend}."
            )
        if plan.model_entry.model_family != ModelFamily.VLM_CHAT:
            raise DispatchCompatibilityError(
                "VERL trainer adapter only supports "
                f"model_family={ModelFamily.VLM_CHAT.value}; got "
                f"model_family={plan.model_entry.model_family.value}, "
                f"task_type={plan.task_type.value}, "
                f"trainer_backend={TrainerBackend.VERL.value}."
            )

    def launch(self, config: TaskConfig | DispatchPlan, **kwargs: Any) -> dict[str, Any]:
        return self.run(config, **kwargs)

    def dry_run(self, config: TaskConfig | DispatchPlan, **kwargs: Any) -> dict[str, Any]:
        return self.run(config, dry_run=True, **kwargs)

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


def build_verl_argv(
    *,
    config_file: Path,
    backend_config: dict[str, Any] | None = None,
) -> list[str]:
    backend_config = dict(backend_config or {})
    module = str(backend_config.get("module") or "verl.trainer.main_ppo")
    overrides = _coerce_sequence(backend_config.get("overrides"))
    return [sys.executable, "-m", module, f"config={config_file}", *overrides]


def _coerce_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


Adapter = VERLTrainerAdapter
TrainerAdapter = VERLTrainerAdapter
adapter = VERLTrainerAdapter()


__all__ = ["Adapter", "TrainerAdapter", "VERLTrainerAdapter", "adapter", "build_verl_argv"]
