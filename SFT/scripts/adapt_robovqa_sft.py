#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from json import JSONDecoder
from pathlib import Path
from typing import Any, Iterator


DEFAULT_SOURCE_JSON = os.environ.get("ROBOVQA_SOURCE_JSON")
DEFAULT_SOURCE_MEDIA_ROOT = os.environ.get("ROBOVQA_SOURCE_MEDIA_ROOT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt the RoboVQA Cosmos SFT dataset into LLaMA-Factory ShareGPT format."
    )
    parser.add_argument("--source-json", type=Path, default=Path(DEFAULT_SOURCE_JSON) if DEFAULT_SOURCE_JSON else None)
    parser.add_argument(
        "--source-media-root",
        type=Path,
        default=Path(DEFAULT_SOURCE_MEDIA_ROOT) if DEFAULT_SOURCE_MEDIA_ROOT else None,
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sample-denominator", type=int, default=4)
    parser.add_argument("--sample-remainder", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()
    if args.source_json is None or args.source_media_root is None:
        parser.error(
            "Provide --source-json and --source-media-root, or set ROBOVQA_SOURCE_JSON and ROBOVQA_SOURCE_MEDIA_ROOT."
        )

    return args


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Any]:
    decoder = JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        cursor = 0

        while True:
            if cursor >= len(buffer):
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(f"{path} is empty.")
                buffer = chunk
                cursor = 0
                break

        while cursor < len(buffer) and buffer[cursor].isspace():
            cursor += 1
        if cursor >= len(buffer) or buffer[cursor] != "[":
            raise ValueError(f"{path} must contain a top-level JSON array.")
        cursor += 1

        while True:
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor < len(buffer):
                    break
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(f"Unexpected EOF while reading {path}.")
                buffer = buffer[cursor:] + chunk
                cursor = 0

            if buffer[cursor] == "]":
                return
            if buffer[cursor] == ",":
                cursor += 1
                continue

            while True:
                try:
                    item, next_cursor = decoder.raw_decode(buffer, cursor)
                    yield item
                    cursor = next_cursor
                    break
                except json.JSONDecodeError:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        raise
                    buffer = buffer[cursor:] + chunk
                    cursor = 0


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = row["conversations"]
    if len(messages) < 3:
        raise ValueError("Expected at least system, user, and assistant messages.")

    system_text = str(messages[0]["content"]).strip()
    user_text = str(messages[1]["content"]).strip()
    assistant_text = str(messages[2]["content"]).strip()

    video_path = row["video"]
    return {
        "system": system_text,
        "tools": "[]",
        "images": [],
        "videos": [video_path],
        "conversations": [
            {
                "from": "human",
                "value": f"<video>\n{user_text}",
            },
            {
                "from": "gpt",
                "value": assistant_text,
            },
        ],
    }


def ensure_media_symlink(output_root: Path, source_media_root: Path) -> None:
    source_clips = source_media_root / "clips"
    target = output_root / "clips"
    if target.is_symlink() or target.exists():
        if target.is_symlink() and os.path.realpath(target) == str(source_clips.resolve()):
            return
        raise FileExistsError(f"{target} already exists and does not point to {source_clips}.")
    target.symlink_to(source_clips)


def write_dataset(
    source_json: Path,
    source_media_root: Path,
    output_root: Path,
    sample_denominator: int,
    sample_remainder: int,
    max_examples: int | None,
    max_videos: int | None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_media_symlink(output_root, source_media_root)

    output_path = output_root / "train.json"
    selected = 0
    seen = 0
    unique_videos: set[str] = set()
    task_names: dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        first = True
        for idx, row in enumerate(iter_json_array(source_json)):
            if idx % sample_denominator != sample_remainder:
                continue
            if max_examples is not None and selected >= max_examples:
                break

            converted = convert_row(row)
            if not first:
                handle.write(",\n")
            json.dump(converted, handle, ensure_ascii=False)
            first = False

            selected += 1
            seen += 1
            unique_videos.add(row["video"])
            for task_meta in row.get("metadata", {}).get("task_metadata", []):
                task = task_meta.get("task")
                if task:
                    task_names[task] = task_names.get(task, 0) + 1

            if max_videos is not None and len(unique_videos) >= max_videos:
                break

        handle.write("\n]\n")

    summary = {
        "source_json": str(source_json),
        "source_media_root": str(source_media_root),
        "output_path": str(output_path),
        "sample_denominator": sample_denominator,
        "sample_remainder": sample_remainder,
        "examples": selected,
        "unique_videos": len(unique_videos),
        "task_counts": dict(sorted(task_names.items())),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = args.output_root or (repo_root / "SFT" / "data" / "robovqa_cosmos_sft_q1")
    summary = write_dataset(
        source_json=args.source_json,
        source_media_root=args.source_media_root,
        output_root=output_root,
        sample_denominator=args.sample_denominator,
        sample_remainder=args.sample_remainder,
        max_examples=args.max_examples,
        max_videos=args.max_videos,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
