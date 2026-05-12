#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
POST_TRAINING_ROOT = REPO_ROOT / "world-model-post-training"
POST_TRAINING_SRCS = [
    POST_TRAINING_ROOT / "shared" / "src",
    POST_TRAINING_ROOT / "vjepa" / "src",
    POST_TRAINING_ROOT / "wan" / "src",
    POST_TRAINING_ROOT / "dreamdojo" / "src",
]
for src in reversed(POST_TRAINING_SRCS):
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

from verl_post_training.adapters.dataset import get_dataset_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SFT ShareGPT multimodal data into VERL RL JSONL.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repo root for verl-post-training.",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def convert_records(
    records: list[dict[str, Any]],
    media_root: Path,
    data_source: str,
    video_sampling: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    adapter = get_dataset_adapter("chat_rl")
    converted: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        converted.append(
            adapter.prepare_row(
                record,
                config={
                    "index": index,
                    "media_root": media_root,
                    "data_source": data_source,
                    "video_sampling": video_sampling,
                },
            )
        )
    return converted


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def convert_split(
    src_path: Path,
    dst_path: Path,
    media_root: Path,
    data_source: str,
    video_sampling: dict[str, Any] | None = None,
) -> int:
    rows = convert_records(
        load_json(src_path),
        media_root=media_root,
        data_source=data_source,
        video_sampling=video_sampling,
    )
    write_jsonl(dst_path, rows)
    return len(rows)


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()

    specs = [
        {
            "name": "video_native",
            "src_train": root_dir / "SFT/data/procedural_grpo_smoke/video_native/train.json",
            "src_val": root_dir / "SFT/data/procedural_grpo_smoke/video_native/val.json",
            "media_root": root_dir / "SFT/data/procedural_grpo_smoke/video_native",
            "dst_train": root_dir / "RL/data/procedural_grpo_smoke/video_native/train.jsonl",
            "dst_val": root_dir / "RL/data/procedural_grpo_smoke/video_native/val.jsonl",
            "data_source": "procedural_grpo/video_next_step",
            "video_sampling": {
                "fps": 1.0,
                "min_frames": 4,
                "max_frames": 8,
            },
        },
        {
            "name": "image_native",
            "src_train": root_dir / "SFT/data/procedural_grpo_smoke/image_native/train.json",
            "src_val": root_dir / "SFT/data/procedural_grpo_smoke/image_native/val.json",
            "media_root": root_dir / "SFT/data/procedural_grpo_smoke/image_native",
            "dst_train": root_dir / "RL/data/procedural_grpo_smoke/image_native/train.jsonl",
            "dst_val": root_dir / "RL/data/procedural_grpo_smoke/image_native/val.jsonl",
            "data_source": "procedural_grpo/image_next_step",
        },
    ]

    summary: dict[str, Any] = {}
    for spec in specs:
        train_count = convert_split(
            src_path=spec["src_train"],
            dst_path=spec["dst_train"],
            media_root=spec["media_root"],
            data_source=spec["data_source"],
            video_sampling=spec.get("video_sampling"),
        )
        val_count = convert_split(
            src_path=spec["src_val"],
            dst_path=spec["dst_val"],
            media_root=spec["media_root"],
            data_source=spec["data_source"],
            video_sampling=spec.get("video_sampling"),
        )
        summary[spec["name"]] = {
            "train": train_count,
            "val": val_count,
            "train_path": str(spec["dst_train"]),
            "val_path": str(spec["dst_val"]),
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
