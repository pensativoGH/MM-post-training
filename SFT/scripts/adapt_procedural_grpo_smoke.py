#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = os.environ.get("PROCEDURAL_GRPO_SOURCE_ROOT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt procedural-grpo smoke SFT data into LLaMA-Factory ShareGPT datasets.")
    parser.add_argument("--source-root", type=Path, default=Path(DEFAULT_SOURCE_ROOT) if DEFAULT_SOURCE_ROOT else None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=2)
    parser.add_argument("--val-limit", type=int, default=1)
    args = parser.parse_args()
    if args.source_root is None:
        parser.error("Provide --source-root, or set PROCEDURAL_GRPO_SOURCE_ROOT.")

    return args


def load_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def find_ffmpeg() -> str:
    import shutil

    path = shutil.which("ffmpeg")
    if path:
        return path
    return "ffmpeg"


def extract_frames(video_path: Path, output_dir: Path, fps: int, max_frames: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                ffmpeg,
                "-i",
                str(video_path),
                "-vf",
                f"fps={fps}",
                "-q:v",
                "2",
                f"{tmpdir}/frame_%04d.jpg",
                "-loglevel",
                "quiet",
            ],
            check=True,
        )
        frame_paths = sorted(Path(tmpdir).glob("frame_*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"No frames extracted from {video_path}")
        if len(frame_paths) > max_frames:
            if max_frames == 1:
                frame_paths = [frame_paths[0]]
            else:
                indices = [int(i * (len(frame_paths) - 1) / (max_frames - 1)) for i in range(max_frames)]
                frame_paths = [frame_paths[i] for i in indices]

        materialized: list[Path] = []
        for idx, src in enumerate(frame_paths):
            dst = output_dir / f"frame_{idx:04d}.jpg"
            dst.write_bytes(src.read_bytes())
            materialized.append(dst)
        return materialized


def extract_system_text(messages: list[dict[str, Any]]) -> str:
    system_content = messages[0]["content"]
    if isinstance(system_content, list):
        return "".join(item.get("text", "") for item in system_content if item.get("type") == "text")
    return str(system_content)


def extract_user_text(messages: list[dict[str, Any]]) -> str:
    user_content = messages[1]["content"]
    parts: list[str] = []
    for item in user_content:
        if item.get("type") == "text":
            parts.append(item["text"])
    return "\n".join(parts).strip()


def extract_assistant_text(messages: list[dict[str, Any]]) -> str:
    assistant_content = messages[2]["content"]
    for item in assistant_content:
        if item.get("type") == "text":
            return item["text"]
    raise ValueError("Assistant message has no text item.")


def to_video_native(row: dict[str, Any], output_root: Path) -> dict[str, Any]:
    video_path = Path(row["video_path"]).resolve()
    return {
        "system": extract_system_text(row["messages"]),
        "tools": "[]",
        "images": [],
        "videos": [str(video_path)],
        "conversations": [
            {
                "from": "human",
                "value": f"<video>\n{extract_user_text(row['messages'])}",
            },
            {
                "from": "gpt",
                "value": extract_assistant_text(row["messages"]),
            },
        ],
    }


def to_image_native(row: dict[str, Any], output_root: Path, fps: int, max_frames: int) -> dict[str, Any]:
    video_path = Path(row["video_path"]).resolve()
    frame_dir = output_root / "frames" / row["id"]
    frames = extract_frames(video_path, frame_dir, fps=fps, max_frames=max_frames)
    rel_frames = [str(frame.relative_to(output_root)) for frame in frames]
    placeholder_prefix = "".join("<image>\n" for _ in rel_frames)
    return {
        "system": extract_system_text(row["messages"]),
        "tools": "[]",
        "images": rel_frames,
        "videos": [],
        "conversations": [
            {
                "from": "human",
                "value": f"{placeholder_prefix}{extract_user_text(row['messages'])}",
            },
            {
                "from": "gpt",
                "value": extract_assistant_text(row["messages"]),
            },
        ],
    }


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = args.output_root or (repo_root / "SFT" / "data" / "procedural_grpo_smoke")

    train_rows = load_jsonl(args.source_root / "train.jsonl", args.train_limit)
    val_rows = load_jsonl(args.source_root / "val.jsonl", args.val_limit)

    video_train = [to_video_native(row, output_root / "video_native") for row in train_rows]
    video_val = [to_video_native(row, output_root / "video_native") for row in val_rows]
    image_train = [to_image_native(row, output_root / "image_native", args.fps, args.max_frames) for row in train_rows]
    image_val = [to_image_native(row, output_root / "image_native", args.fps, args.max_frames) for row in val_rows]

    write_json(output_root / "video_native" / "train.json", video_train)
    write_json(output_root / "video_native" / "val.json", video_val)
    write_json(output_root / "image_native" / "train.json", image_train)
    write_json(output_root / "image_native" / "val.json", image_val)

    summary = {
        "video_native_train": len(video_train),
        "video_native_val": len(video_val),
        "image_native_train": len(image_train),
        "image_native_val": len(image_val),
        "fps": args.fps,
        "max_frames": args.max_frames,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
