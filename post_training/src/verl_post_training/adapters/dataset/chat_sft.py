"""Pipeline-to-ShareGPT adapter for chat SFT flows."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

ROLE_MAP = {
    "user": "human",
    "human": "human",
    "assistant": "gpt",
    "gpt": "gpt",
    "system": "system",
    "observation": "observation",
    "tool": "observation",
    "function": "function_call",
    "function_call": "function_call",
}


class ChatSFTDatasetAdapter:
    adapter_key = "chat_sft"

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
        examples = [self.prepare_row(row) for row in rows]

        output_path = Path(output_dir) / f"{split}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(examples, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output_path

    def prepare_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return convert_pipeline_row_to_sharegpt(row)


def load_manifest(path: Path) -> Any:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def coerce_pipeline_rows(
    pipeline_manifest: Any,
    *,
    split: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(pipeline_manifest, (str, Path)):
        manifest = load_manifest(Path(pipeline_manifest))
        return resolve_manifest_rows(manifest, split=split, config=config)
    return resolve_manifest_rows(pipeline_manifest, split=split, config=config)


def resolve_manifest_rows(manifest: Any, *, split: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _resolve_rows_from_manifest(manifest, split=split, config=config)
    normalized = copy.deepcopy(rows)
    if not isinstance(normalized, list):
        raise TypeError(f"Expected manifest rows to resolve to a list, got {type(normalized).__name__}")
    for row in normalized:
        if not isinstance(row, dict):
            raise TypeError(f"Expected row mappings, got {type(row).__name__}")
    return normalized


def _resolve_rows_from_manifest(manifest: Any, *, split: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    split_aliases = tuple(dict.fromkeys((split, config.get("split"), "records", "rows", "examples")))

    if isinstance(manifest, list):
        return manifest

    if not isinstance(manifest, dict):
        raise TypeError(f"Unsupported manifest type: {type(manifest).__name__}")

    for key in split_aliases:
        if key and isinstance(manifest.get(key), list):
            return manifest[key]

    datasets = manifest.get("datasets")
    if isinstance(datasets, dict):
        for key in split_aliases:
            dataset = datasets.get(key) if key else None
            if isinstance(dataset, list):
                return dataset
            if isinstance(dataset, dict):
                for nested_key in ("rows", "records", "examples"):
                    nested = dataset.get(nested_key)
                    if isinstance(nested, list):
                        return nested

    for nested_key in ("rows", "records", "examples"):
        nested = manifest.get(nested_key)
        if isinstance(nested, list):
            return nested

    raise KeyError(f"Could not resolve split {split!r} from manifest.")


def convert_pipeline_row_to_sharegpt(row: dict[str, Any]) -> dict[str, Any]:
    images, videos = infer_media_lists(row)
    messages = list(row.get("messages") or row.get("conversation") or row.get("conversations") or [])
    system, turns = normalize_turns(messages)
    turns = inject_media_tokens(turns, images, videos)
    return {
        "system": system,
        "tools": "[]",
        "images": images,
        "videos": videos,
        "conversations": turns,
    }


def infer_media_lists(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    images: list[str] = []
    videos: list[str] = []
    modality = str(row.get("modality") or "").lower()
    for media_path in row.get("media_paths") or []:
        media_path_str = str(media_path)
        suffix = Path(media_path_str).suffix.lower()
        if modality == "video" or suffix in VIDEO_SUFFIXES:
            videos.append(media_path_str)
        elif modality == "image" or suffix in IMAGE_SUFFIXES:
            images.append(media_path_str)
        else:
            raise ValueError(f"Unsupported media path for SFT conversion: {media_path}")
    return images, videos


def normalize_turns(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    if not messages:
        raise ValueError("Missing messages.")

    turns: list[dict[str, str]] = []
    for idx, message in enumerate(messages):
        raw_role = str(message.get("role") or message.get("from") or "").strip().lower()
        mapped_role = ROLE_MAP.get(raw_role)
        if mapped_role is None:
            raise ValueError(f"Unsupported role {raw_role!r} at turn {idx}.")
        content = message.get("content", message.get("value", ""))
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        turns.append({"from": mapped_role, "value": content.strip()})

    system = ""
    if turns and turns[0]["from"] == "system":
        system = turns[0]["value"]
        turns = turns[1:]

    if not turns:
        raise ValueError("No non-system turns remain after normalization.")
    if turns[-1]["from"] != "gpt":
        raise ValueError("Expected final assistant turn for SFT.")
    return system, turns


def inject_media_tokens(
    turns: list[dict[str, str]],
    images: list[str],
    videos: list[str],
) -> list[dict[str, str]]:
    if not images and not videos:
        return turns

    token_prefix = []
    token_prefix.extend("<image>" for _ in images)
    token_prefix.extend("<video>" for _ in videos)
    prefix = "\n".join(token_prefix)

    if any("<image>" in turn["value"] or "<video>" in turn["value"] for turn in turns):
        return turns

    for turn in turns:
        if turn["from"] == "human":
            turn["value"] = f"{prefix}\n{turn['value']}" if turn["value"] else prefix
            return turns

    raise ValueError("Could not find a human turn to attach media placeholders.")
