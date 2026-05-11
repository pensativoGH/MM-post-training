#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
POST_TRAINING_SRC = REPO_ROOT / "post_training" / "src"
if str(POST_TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(POST_TRAINING_SRC))

from verl_post_training.adapters.dataset import get_dataset_adapter


DEFAULT_PIPELINE_REPO_NAME = "multimodal-data-pipeline-clean"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
SHAREGPT_COLUMNS = {
    "messages": "conversations",
    "images": "images",
    "videos": "videos",
    "system": "system",
    "tools": "tools",
}
SHAREGPT_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
    "system_tag": "system",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pipeline RoboVQA dataset versions into local LLaMA-Factory ShareGPT train/val datasets."
    )
    parser.add_argument("--pipeline-config", type=Path, required=True, help="Path to pipeline YAML config.")
    parser.add_argument("--train-dataset-version-id", required=True)
    parser.add_argument("--val-dataset-version-id", required=True)
    parser.add_argument("--parent-dataset-name", default="robovqa_cosmos_sft_q1")
    parser.add_argument("--train-dataset-name", default="robovqa_cosmos_sft_q1_train")
    parser.add_argument("--val-dataset-name", default="robovqa_cosmos_sft_q1_val")
    parser.add_argument("--dataset-info", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def resolve_pipeline_repo(pipeline_config_path: Path) -> Path:
    env_repo = os.environ.get("MULTIMODAL_DATA_PIPELINE_REPO") or os.environ.get("PIPELINE_REPO")
    candidates: list[Path] = []
    if env_repo:
        candidates.append(Path(env_repo).expanduser())

    resolved_config = pipeline_config_path.resolve()
    for candidate in [resolved_config, *resolved_config.parents]:
        if (candidate / "src" / "multimodal_data_pipeline").is_dir():
            candidates.append(candidate)

    candidates.append(Path.home() / "code" / DEFAULT_PIPELINE_REPO_NAME)
    candidates.append(Path("/home/pensativo/code") / DEFAULT_PIPELINE_REPO_NAME)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "src" / "multimodal_data_pipeline").is_dir():
            return resolved

    searched = "\n".join(str(path) for path in seen)
    raise FileNotFoundError(
        "Could not locate multimodal-data-pipeline-clean with src/multimodal_data_pipeline.\n"
        f"Searched:\n{searched}"
    )


def ensure_pipeline_importable(pipeline_config_path: Path) -> Path:
    pipeline_repo = resolve_pipeline_repo(pipeline_config_path)
    pipeline_src = pipeline_repo / "src"
    if str(pipeline_src) not in sys.path:
        sys.path.insert(0, str(pipeline_src))
    return pipeline_repo


def load_pipeline_backend(pipeline_config_path: Path):
    pipeline_repo = ensure_pipeline_importable(pipeline_config_path)
    from multimodal_data_pipeline.backends.metadata import build_metadata_backend
    from multimodal_data_pipeline.config import load_config

    pipeline_config = load_config(pipeline_config_path)
    metadata_backend = build_metadata_backend(
        pipeline_config.metadata_backend,
        output_root=pipeline_config.output_root,
        warehouse_root=pipeline_config.warehouse_root,
        catalog_config=pipeline_config.catalog_config,
    )
    metadata_backend.initialize()
    return pipeline_repo, pipeline_config, metadata_backend


def load_dataset_version_rows(metadata_backend, dataset_version_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from multimodal_data_pipeline.trainer_reader import DatasetVersionReader

    reader = DatasetVersionReader.from_metadata_backend(metadata_backend)
    versions = {
        row["dataset_version_id"]: row
        for row in metadata_backend.read_table("dataset_versions")
        if row.get("dataset_version_id")
    }
    version = versions.get(dataset_version_id)
    if version is None:
        raise KeyError(f"Unknown dataset_version_id: {dataset_version_id}")
    return version, reader.read(dataset_version_id)


def infer_media_lists(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    images: list[str] = []
    videos: list[str] = []
    modality = str(row.get("modality") or "").lower()
    for media_path in row.get("media_paths") or []:
        resolved = str(Path(media_path).resolve())
        suffix = Path(resolved).suffix.lower()
        if modality == "video" or suffix in VIDEO_SUFFIXES:
            videos.append(resolved)
        elif modality == "image" or suffix in IMAGE_SUFFIXES:
            images.append(resolved)
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


def inject_media_tokens(turns: list[dict[str, str]], images: list[str], videos: list[str]) -> list[dict[str, str]]:
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


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    adapter = get_dataset_adapter("chat_sft")
    return adapter.prepare_row(row)


def write_examples(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_registry(
    dataset_info_path: Path,
    *,
    train_dataset_name: str,
    val_dataset_name: str,
    output_root: Path,
    pipeline_config_path: Path,
    train_version: dict[str, Any],
    val_version: dict[str, Any],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    parent_dataset_name: str,
) -> None:
    registry = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    relative_root = output_root.relative_to(dataset_info_path.parent)
    common = {
        "formatting": "sharegpt",
        "columns": dict(SHAREGPT_COLUMNS),
        "tags": dict(SHAREGPT_TAGS),
        "source": {
            "type": "multimodal-data-pipeline",
            "pipeline_config": str(pipeline_config_path.resolve()),
            "parent_dataset_name": parent_dataset_name,
        },
    }
    registry[train_dataset_name] = {
        **common,
        "file_name": str(relative_root / "train.json"),
        "source": {
            **common["source"],
            "split": "train",
            "dataset_version_id": train_version["dataset_version_id"],
            "source_dataset_name": train_version.get("dataset_name"),
            "row_count": len(train_rows),
        },
    }
    registry[val_dataset_name] = {
        **common,
        "file_name": str(relative_root / "val.json"),
        "source": {
            **common["source"],
            "split": "val",
            "dataset_version_id": val_version["dataset_version_id"],
            "source_dataset_name": val_version.get("dataset_name"),
            "row_count": len(val_rows),
        },
    }
    dataset_info_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def write_manifest(
    path: Path,
    *,
    pipeline_repo: Path,
    pipeline_config_path: Path,
    dataset_info_path: Path,
    output_root: Path,
    parent_dataset_name: str,
    train_dataset_name: str,
    val_dataset_name: str,
    train_version: dict[str, Any],
    val_version: dict[str, Any],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_repo": str(pipeline_repo.resolve()),
        "pipeline_config": str(pipeline_config_path.resolve()),
        "dataset_info_path": str(dataset_info_path.resolve()),
        "output_root": str(output_root.resolve()),
        "parent_dataset_name": parent_dataset_name,
        "datasets": {
            "train": {
                "dataset_name": train_dataset_name,
                "dataset_version_id": train_version["dataset_version_id"],
                "source_dataset_name": train_version.get("dataset_name"),
                "row_count": len(train_rows),
                "file_name": "train.json",
            },
            "val": {
                "dataset_name": val_dataset_name,
                "dataset_version_id": val_version["dataset_version_id"],
                "source_dataset_name": val_version.get("dataset_name"),
                "row_count": len(val_rows),
                "file_name": "val.json",
            },
        },
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    dataset_info_path = args.dataset_info or (repo_root / "SFT" / "data" / "dataset_info.json")
    output_root = args.output_root or (repo_root / "SFT" / "data" / "robovqa_cosmos_sft_q1")
    manifest_path = output_root / "pipeline_sft_dataset_manifest.json"

    pipeline_repo, _, metadata_backend = load_pipeline_backend(args.pipeline_config.resolve())
    train_version, train_source_rows = load_dataset_version_rows(metadata_backend, args.train_dataset_version_id)
    val_version, val_source_rows = load_dataset_version_rows(metadata_backend, args.val_dataset_version_id)

    train_examples = [convert_row(row) for row in train_source_rows]
    val_examples = [convert_row(row) for row in val_source_rows]

    write_examples(output_root / "train.json", train_examples)
    write_examples(output_root / "val.json", val_examples)
    update_registry(
        dataset_info_path,
        train_dataset_name=args.train_dataset_name,
        val_dataset_name=args.val_dataset_name,
        output_root=output_root,
        pipeline_config_path=args.pipeline_config,
        train_version=train_version,
        val_version=val_version,
        train_rows=train_examples,
        val_rows=val_examples,
        parent_dataset_name=args.parent_dataset_name,
    )
    manifest = write_manifest(
        manifest_path,
        pipeline_repo=pipeline_repo,
        pipeline_config_path=args.pipeline_config,
        dataset_info_path=dataset_info_path,
        output_root=output_root,
        parent_dataset_name=args.parent_dataset_name,
        train_dataset_name=args.train_dataset_name,
        val_dataset_name=args.val_dataset_name,
        train_version=train_version,
        val_version=val_version,
        train_rows=train_examples,
        val_rows=val_examples,
    )

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
