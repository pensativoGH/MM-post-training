"""Pipeline-to-RL adapter for chat RL flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .chat_sft import (
    ChatSFTDatasetAdapter,
    load_manifest,
    resolve_manifest_rows,
)


class ChatRLDatasetAdapter:
    adapter_key = "chat_rl"

    def prepare(
        self,
        input_manifest: Path,
        output_dir: Path,
        split: str,
        config: dict[str, Any],
    ) -> Path:
        manifest = load_manifest(input_manifest)
        rows = resolve_manifest_rows(manifest, split=split, config=config)
        converted = [
            convert_record_to_rl(record, config={**config, "index": index})
            for index, record in enumerate(rows)
        ]

        output_path = Path(output_dir) / f"{split}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in converted:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        return output_path


def convert_record_to_rl(record: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    sharegpt_record = _coerce_to_sharegpt_record(record)
    messages = make_messages(sharegpt_record)
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("Expected final assistant answer in ShareGPT record.")

    assistant_answer = messages.pop()["content"]
    gold = parse_gold(assistant_answer)
    images, videos = media_entries(
        sharegpt_record,
        media_root=Path(config.get("media_root") or Path.cwd()),
        video_sampling=config.get("video_sampling"),
    )

    return {
        "prompt": messages,
        "images": images,
        "videos": videos,
        "data_source": str(config.get("data_source") or "chat_rl"),
        "reward_model": {
            "ground_truth": gold,
        },
        "extra_info": {
            "index": config.get("index", 0),
            "gold_response": assistant_answer,
            "media_type": "video" if videos else "image",
        },
    }


def _coerce_to_sharegpt_record(record: dict[str, Any]) -> dict[str, Any]:
    if "conversations" in record:
        return {
            "system": record.get("system", ""),
            "conversations": list(record.get("conversations") or []),
            "images": list(record.get("images") or []),
            "videos": list(record.get("videos") or []),
        }
    return ChatSFTDatasetAdapter().prepare_row(record)


def parse_gold(answer: str) -> dict[str, Any]:
    parsed = json.loads(answer)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected dict answer, got: {type(parsed)}")
    return parsed


def make_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = record.get("system")
    if system:
        messages.append({"role": "system", "content": system})

    for turn in record["conversations"]:
        role = turn["from"]
        if role == "human":
            mapped_role = "user"
        elif role == "gpt":
            mapped_role = "assistant"
        else:
            raise ValueError(f"Unsupported ShareGPT role: {role}")
        messages.append({"role": mapped_role, "content": turn["value"]})
    return messages


def media_entries(
    record: dict[str, Any],
    media_root: Path,
    video_sampling: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    image_entries: list[dict[str, str]] = []
    for image in record.get("images", []):
        image_path = Path(image)
        if not image_path.is_absolute():
            image_path = (media_root / image).resolve()
        image_entries.append({"image": str(image_path)})

    video_entries: list[dict[str, Any]] = []
    for video in record.get("videos", []):
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = (media_root / video).resolve()
        video_entry: dict[str, Any] = {"video": str(video_path)}
        if video_sampling:
            video_entry.update(video_sampling)
        video_entries.append(video_entry)

    return image_entries, video_entries
