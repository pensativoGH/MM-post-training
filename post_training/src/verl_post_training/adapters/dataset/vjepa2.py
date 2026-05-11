"""Pipeline-to-V-JEPA2 dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verl_post_training.bootstrap.third_party import discover_upstream_root
from .chat_sft import coerce_pipeline_rows


class VJEPA2DatasetAdapter:
    adapter_key = "vjepa2"
    upstream_family = "vjepa2"

    def discover_upstream_root(self, *, manifest_path: Path | None = None) -> Path:
        """Resolve the pinned repo-owned checkout path for V-JEPA2."""

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
        if "input_manifest" in kwargs and pipeline_manifest is None:
            pipeline_manifest = kwargs.pop("input_manifest")
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
        video_refs = _extract_video_refs(row)
        example_id = _coerce_example_id(row, index=index)
        example: dict[str, Any] = {
            "example_id": example_id,
            "video_path": video_refs[0],
            "video_paths": video_refs,
            "source": "pipeline_manifest_reference",
        }
        if "metadata" in row and isinstance(row["metadata"], dict):
            example["metadata"] = dict(row["metadata"])
        return example


def _coerce_example_id(row: dict[str, Any], *, index: int) -> str:
    for key in ("example_id", "id", "sample_id"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return f"example-{index:05d}"


def _extract_video_refs(row: dict[str, Any]) -> list[str]:
    direct_fields = (
        row.get("video_path"),
        row.get("video"),
        row.get("media_path"),
        row.get("path"),
        row.get("uri"),
        row.get("url"),
    )
    for value in direct_fields:
        if isinstance(value, str) and value.strip():
            return [value]

    for list_key in ("video_paths", "videos", "media_paths"):
        refs = _extract_string_refs(row.get(list_key))
        if refs:
            return refs

    for list_key in ("media", "assets", "inputs"):
        refs = _extract_mapping_refs(row.get(list_key))
        if refs:
            return refs

    raise ValueError("Could not resolve a video reference for V-JEPA2 input row.")


def _extract_string_refs(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(path).strip() for path in value if str(path).strip()]


def _extract_mapping_refs(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    refs: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            refs.append(item)
            continue
        if not isinstance(item, dict):
            continue
        modality = str(item.get("modality") or item.get("type") or "").lower()
        if modality and "video" not in modality:
            continue
        for key in ("video_path", "path", "uri", "url"):
            ref = item.get(key)
            if isinstance(ref, str) and ref.strip():
                refs.append(ref)
                break
    return refs
