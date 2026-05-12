"""Pipeline-to-DreamDojo world-model dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verl_post_training.bootstrap.third_party import discover_upstream_root
from .chat_sft import coerce_pipeline_rows


class DreamDojoDatasetAdapter:
    adapter_key = "dreamdojo"
    upstream_family = "dreamdojo"

    def discover_upstream_root(self, *, manifest_path: Path | None = None) -> Path:
        """Resolve the pinned repo-owned checkout path for DreamDojo."""

        return discover_upstream_root(self.upstream_family, manifest_path=manifest_path)

    def prepare(
        self,
        pipeline_manifest: Any = None,
        output_dir: Path | None = None,
        split: str = "train",
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Path:
        config = dict(config or {})
        for alias in ("input_manifest", "manifest", "rows"):
            if alias in kwargs and pipeline_manifest is None:
                pipeline_manifest = kwargs.pop(alias)
        if kwargs:
            config.update(kwargs)
        if pipeline_manifest is None:
            raise TypeError("Missing required pipeline_manifest or input_manifest.")
        if output_dir is None:
            raise TypeError("Missing required output_dir.")

        rows = coerce_pipeline_rows(
            pipeline_manifest,
            split=split,
            config=config,
        )

        output_path = Path(output_dir) / f"{split}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                prepared = self.prepare_row(row, index=index)
                handle.write(json.dumps(prepared, ensure_ascii=False) + "\n")
        return output_path

    def prepare_row(self, row: dict[str, Any], *, index: int) -> dict[str, Any]:
        example_id = _coerce_example_id(row, index=index)
        observations = _extract_observations(row)
        actions = _extract_actions(row)
        if actions and len(actions) not in {len(observations), max(0, len(observations) - 1)}:
            raise ValueError(
                "DreamDojo trajectory row has incompatible action/observation "
                f"lengths for example_id={example_id!r}."
            )

        example: dict[str, Any] = {
            "example_id": example_id,
            "trajectory_id": str(row.get("trajectory_id") or row.get("episode_id") or example_id),
            "observations": observations,
            "actions": actions,
            "world_model_input": {
                "observations": observations,
                "actions": actions,
            },
            "source": "pipeline_manifest_reference",
        }
        goal = row.get("goal") or row.get("instruction") or row.get("prompt")
        if goal is not None:
            example["goal"] = str(goal)
            example["world_model_input"]["goal"] = str(goal)
        if "metadata" in row and isinstance(row["metadata"], dict):
            example["metadata"] = dict(row["metadata"])
        return example


def _coerce_example_id(row: dict[str, Any], *, index: int) -> str:
    for key in ("example_id", "id", "sample_id", "trajectory_id", "episode_id"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return f"example-{index:05d}"


def _extract_observations(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("observations", "states", "frames"):
        value = row.get(key)
        if isinstance(value, list) and value:
            return [_normalize_observation(item, index=i) for i, item in enumerate(value)]

    for key in ("observation", "state", "image_path", "video_path", "path", "uri"):
        value = row.get(key)
        if value is not None:
            return [_normalize_observation(value, index=0)]

    raise ValueError("Could not resolve observations for DreamDojo trajectory row.")


def _extract_actions(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("actions")
    if isinstance(value, list):
        return [_normalize_action(item, index=i) for i, item in enumerate(value)]

    value = row.get("action")
    if value is not None:
        return [_normalize_action(value, index=0)]
    return []


def _normalize_observation(value: Any, *, index: int) -> dict[str, Any]:
    if isinstance(value, dict):
        item = dict(value)
    elif isinstance(value, str):
        item = {"path": value}
    else:
        item = {"value": value}
    item.setdefault("timestep", index)
    return item


def _normalize_action(value: Any, *, index: int) -> dict[str, Any]:
    if isinstance(value, dict):
        item = dict(value)
    else:
        item = {"value": value}
    item.setdefault("timestep", index)
    return item


Adapter = DreamDojoDatasetAdapter
DatasetAdapter = DreamDojoDatasetAdapter
adapter = DreamDojoDatasetAdapter()


__all__ = ["Adapter", "DatasetAdapter", "DreamDojoDatasetAdapter", "adapter"]
