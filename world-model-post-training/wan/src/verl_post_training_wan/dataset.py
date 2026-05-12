"""Pipeline-to-Wan conditioning dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verl_post_training.adapters.dataset.chat_sft import coerce_pipeline_rows
from verl_post_training.bootstrap.third_party import discover_upstream_root


class WanDatasetAdapter:
    adapter_key = "wan"
    upstream_family = "wan22"

    def discover_upstream_root(self, *, manifest_path: Path | None = None) -> Path:
        """Resolve the pinned repo-owned checkout path for Wan2.2."""

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

        output_root = Path(output_dir)
        output_path = output_root / f"{split}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                prepared = self.prepare_row(
                    row,
                    index=index,
                    output_root=output_root,
                    artifact_extension=str(config.get("artifact_extension") or ".mp4"),
                )
                handle.write(json.dumps(prepared, ensure_ascii=False) + "\n")
        return output_path

    def prepare_row(
        self,
        row: dict[str, Any],
        *,
        index: int,
        output_root: Path,
        artifact_extension: str = ".mp4",
    ) -> dict[str, Any]:
        example_id = _coerce_example_id(row, index=index)
        prompt = _extract_prompt(row)
        media_refs = _extract_media_refs(row)
        extension = artifact_extension if artifact_extension.startswith(".") else f".{artifact_extension}"
        output_path = Path("generated") / f"{_sanitize_filename(example_id)}{extension}"

        example: dict[str, Any] = {
            "example_id": example_id,
            "prompt": prompt,
            "media_refs": media_refs,
            "conditioning": {
                "prompt": prompt,
                "media": media_refs,
            },
            "output_path": str(output_root / output_path),
            "target_path": str(output_root / output_path),
            "source": "pipeline_manifest_reference",
        }
        negative_prompt = row.get("negative_prompt")
        if negative_prompt is not None:
            example["negative_prompt"] = str(negative_prompt)
            example["conditioning"]["negative_prompt"] = str(negative_prompt)
        for key in ("height", "width", "num_frames", "fps", "seed", "guidance_scale"):
            if key in row:
                example[key] = row[key]
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


def _extract_prompt(row: dict[str, Any]) -> str:
    for key in ("prompt", "text", "caption", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value

    conditioning = row.get("conditioning")
    if isinstance(conditioning, dict):
        value = conditioning.get("prompt") or conditioning.get("text")
        if isinstance(value, str) and value.strip():
            return value

    messages = row.get("messages") or row.get("conversation") or row.get("conversations")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("from") or "").lower()
            if role not in {"user", "human", ""}:
                continue
            content = message.get("content", message.get("value", ""))
            if isinstance(content, str) and content.strip():
                return content

    raise ValueError("Could not resolve a text prompt for Wan conditioning row.")


def _extract_media_refs(row: dict[str, Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for key, modality in (
        ("image_path", "image"),
        ("image", "image"),
        ("video_path", "video"),
        ("video", "video"),
        ("media_path", "media"),
        ("path", "media"),
        ("uri", "media"),
        ("url", "media"),
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            refs.append({"modality": modality, "path": value})

    for key, modality in (
        ("image_paths", "image"),
        ("images", "image"),
        ("conditioning_images", "image"),
        ("video_paths", "video"),
        ("videos", "video"),
        ("conditioning_videos", "video"),
        ("media_paths", "media"),
        ("conditioning_media_paths", "media"),
    ):
        value = row.get(key)
        if isinstance(value, list):
            refs.extend(
                {
                    "modality": (
                        _infer_modality(key, str(item))
                        if modality == "media"
                        else modality
                    ),
                    "path": str(item),
                }
                for item in value
                if str(item).strip()
            )

    for key in ("media", "assets", "inputs", "conditioning_media"):
        value = row.get(key)
        if isinstance(value, list):
            refs.extend(_extract_media_mapping_refs(value))

    conditioning = row.get("conditioning")
    if isinstance(conditioning, dict):
        for key in ("image", "image_path", "video", "video_path", "media_path"):
            value = conditioning.get(key)
            if isinstance(value, str) and value.strip():
                refs.append({"modality": _infer_modality(key, value), "path": value})
        for key in (
            "images",
            "image_paths",
            "videos",
            "video_paths",
            "media_paths",
            "conditioning_images",
            "conditioning_videos",
            "conditioning_media_paths",
        ):
            value = conditioning.get(key)
            if isinstance(value, list):
                refs.extend(
                    {"modality": _infer_modality(key, str(item)), "path": str(item)}
                    for item in value
                    if str(item).strip()
                )

    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        marker = (ref["modality"], ref["path"])
        if marker not in seen:
            seen.add(marker)
            unique.append(ref)
    return unique


def _extract_media_mapping_refs(items: list[Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            refs.append({"modality": "media", "path": item})
            continue
        if not isinstance(item, dict):
            continue
        path = None
        for key in ("path", "uri", "url", "image_path", "video_path", "media_path"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                path = value
                break
        if path is None:
            continue
        modality = str(item.get("modality") or item.get("type") or _infer_modality("", path))
        refs.append({"modality": modality, "path": path})
    return refs


def _infer_modality(key: str, path: str) -> str:
    lowered = f"{key} {path}".lower()
    if "image" in lowered or Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return "image"
    if "video" in lowered or Path(path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        return "video"
    return "media"


def _sanitize_filename(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return sanitized.strip("._") or "example"
