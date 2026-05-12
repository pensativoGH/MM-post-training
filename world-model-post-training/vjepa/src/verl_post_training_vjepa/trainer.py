"""Trainer adapter for repo-level V-JEPA2 masked-video prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

from verl_post_training.bootstrap.third_party import discover_upstream_root
from verl_post_training.launch.dispatch import (
    DispatchCompatibilityError,
    DispatchPlan,
    resolve_dispatch,
)
from verl_post_training.launch.load_config import TaskConfig
from verl_post_training.registry import ModelFamily, TaskType, TrainerBackend


class VJEPA2TrainerAdapter:
    """Build a dry-run launch record for the pinned V-JEPA2 trainer."""

    adapter_key = TrainerBackend.VJEPA2_NATIVE.value
    task_type = TaskType.MASKED_VIDEO_PREDICTION

    def run(
        self,
        config: TaskConfig | DispatchPlan,
        *,
        output_dir: Path | None = None,
        split: str = "train",
        **kwargs: Any,
    ) -> dict[str, Any]:
        plan = _coerce_dispatch_plan(config)
        self.validate(plan)

        output_root = Path(output_dir or plan.config.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        upstream_root = discover_upstream_root("vjepa2")
        backend_config = dict(plan.backend_config)
        backend_config.update(kwargs)

        argv = build_vjepa2_training_argv(
            input_manifest=Path(plan.config.input_manifest),
            output_dir=output_root,
            backend_config=backend_config,
        )
        record = {
            "model_id": plan.config.model_id,
            "task_type": plan.task_type.value,
            "trainer_backend": plan.trainer_backend.value
            if plan.trainer_backend is not None
            else None,
            "input_manifest": plan.config.input_manifest,
            "dataset_manifest": plan.config.input_manifest,
            "output_dir": str(output_root),
            "upstream_root": str(upstream_root),
            "backend_config": backend_config,
            "argument_list": argv,
            "argv": argv,
            "launcher": dict(plan.config.launcher),
            "resources": dict(plan.config.resources),
            "dry_run": True,
            "split": split,
        }

        record_path = output_root / "vjepa2_training_launch.json"
        record_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        record["launch_record_path"] = str(record_path)
        return record

    def validate(self, plan: DispatchPlan) -> None:
        if plan.task_type != self.task_type:
            raise DispatchCompatibilityError(
                "V-JEPA2 trainer adapter only supports "
                f"task_type={self.task_type.value}; got task_type={plan.task_type.value}."
            )
        if plan.trainer_backend != TrainerBackend.VJEPA2_NATIVE:
            backend = plan.trainer_backend.value if plan.trainer_backend else None
            raise DispatchCompatibilityError(
                "V-JEPA2 trainer adapter requires "
                f"trainer_backend={TrainerBackend.VJEPA2_NATIVE.value}; got {backend}."
            )
        if plan.model_entry.model_family != ModelFamily.VIDEO_ENCODER:
            raise DispatchCompatibilityError(
                "V-JEPA2 trainer adapter only supports video_encoder model entries."
            )
        if TrainerBackend.VJEPA2_NATIVE not in plan.model_entry.trainer_backends:
            raise DispatchCompatibilityError(
                "V-JEPA2 trainer adapter requires a registry entry that advertises "
                f"trainer_backend={TrainerBackend.VJEPA2_NATIVE.value}."
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


def build_vjepa2_training_argv(
    *,
    input_manifest: Path,
    output_dir: Path,
    backend_config: dict[str, Any],
) -> list[str]:
    command = backend_config.get("command")
    if isinstance(command, str) and command.strip():
        return [
            command,
            "--input-manifest",
            str(input_manifest),
            "--output-dir",
            str(output_dir),
        ]
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        base = [str(part) for part in command if str(part).strip()]
        if base:
            return [
                *base,
                "--input-manifest",
                str(input_manifest),
                "--output-dir",
                str(output_dir),
            ]

    entrypoint = str(backend_config.get("entrypoint") or "tools.train").strip()
    return [
        sys.executable,
        "-m",
        entrypoint,
        "--input-manifest",
        str(input_manifest),
        "--output-dir",
        str(output_dir),
    ]


Adapter = VJEPA2TrainerAdapter
TrainerAdapter = VJEPA2TrainerAdapter
adapter = VJEPA2TrainerAdapter()


__all__ = [
    "Adapter",
    "TrainerAdapter",
    "VJEPA2TrainerAdapter",
    "adapter",
    "build_vjepa2_training_argv",
]
