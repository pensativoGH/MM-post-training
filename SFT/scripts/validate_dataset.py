#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"
ALLOWED_ROLES = {"human", "gpt", "observation", "system", "function_call"}


def load_registry(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_examples(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a top-level JSON list.")
    return payload


def count_token(messages: list[dict], token: str) -> int:
    total = 0
    for message in messages:
        content = message.get("value", "")
        if isinstance(content, str):
            total += content.count(token)
    return total


def assert_media_exists(dataset_root: Path, media_entries: list[str] | list[list[str]], key: str) -> None:
    if key == "images":
        for media_path in media_entries:
            resolved = dataset_root / media_path
            if not resolved.is_file():
                raise ValueError(f"Missing image file: {resolved}")
    elif key == "videos":
        for video in media_entries:
            if isinstance(video, str):
                resolved = dataset_root / video
                if not resolved.exists():
                    raise ValueError(f"Missing video file: {resolved}")
            else:
                for frame_path in video:
                    resolved = dataset_root / frame_path
                    if not resolved.is_file():
                        raise ValueError(f"Missing video frame file: {resolved}")


def validate_example(example: dict, dataset_root: Path, idx: int) -> dict[str, int]:
    conversations = example.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"Example {idx} has no conversations list.")

    for turn in conversations:
        role = turn.get("from")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Example {idx} has invalid role {role!r}.")
        if not isinstance(turn.get("value"), str):
            raise ValueError(f"Example {idx} has non-string message content.")

    images = example.get("images", [])
    videos = example.get("videos", [])
    if not isinstance(images, list):
        raise ValueError(f"Example {idx} images must be a list.")
    if not isinstance(videos, list):
        raise ValueError(f"Example {idx} videos must be a list.")

    image_tokens = count_token(conversations, IMAGE_TOKEN)
    video_tokens = count_token(conversations, VIDEO_TOKEN)

    if image_tokens != len(images):
        raise ValueError(
            f"Example {idx} has {len(images)} images but {image_tokens} {IMAGE_TOKEN} placeholders."
        )
    if video_tokens != len(videos):
        raise ValueError(
            f"Example {idx} has {len(videos)} videos but {video_tokens} {VIDEO_TOKEN} placeholders."
        )

    assert_media_exists(dataset_root, images, "images")
    assert_media_exists(dataset_root, videos, "videos")

    return {
      "image_count": len(images),
      "video_count": len(videos),
      "turn_count": len(conversations),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local multimodal SFT dataset definitions.")
    parser.add_argument(
        "--dataset-info",
        default=None,
        help="Path to dataset_info.json. Defaults to SFT/data/dataset_info.json.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Single dataset name to validate. Defaults to all registry entries.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_info_path = Path(args.dataset_info) if args.dataset_info else repo_root / "SFT" / "data" / "dataset_info.json"
    registry = load_registry(dataset_info_path)

    dataset_names = [args.dataset] if args.dataset else list(registry.keys())
    total_examples = 0
    total_images = 0
    total_videos = 0

    for dataset_name in dataset_names:
        if dataset_name not in registry:
            raise ValueError(f"Dataset {dataset_name!r} is not defined in {dataset_info_path}.")

        entry = registry[dataset_name]
        file_name = entry["file_name"]
        dataset_file = dataset_info_path.parent / file_name
        examples = load_examples(dataset_file)
        dataset_root = dataset_file.parent

        dataset_images = 0
        dataset_videos = 0
        for idx, example in enumerate(examples):
            stats = validate_example(example, dataset_root, idx)
            dataset_images += stats["image_count"]
            dataset_videos += stats["video_count"]

        total_examples += len(examples)
        total_images += dataset_images
        total_videos += dataset_videos
        print(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "path": str(dataset_file),
                    "examples": len(examples),
                    "images": dataset_images,
                    "videos": dataset_videos,
                }
            )
        )

    print(
        json.dumps(
            {
                "ok": True,
                "datasets": len(dataset_names),
                "examples": total_examples,
                "images": total_images,
                "videos": total_videos,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
