#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


IGNORE_INDEX = -100


def find_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start or __file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "SFT").exists() and (candidate / "README.md").exists():
            return candidate
    raise FileNotFoundError("Could not locate verl-post-training repo root.")


def _ensure_llamafactory_importable(root_dir: Path) -> None:
    opensearch_sft_dir = Path(os.environ.get("OPENSEARCH_SFT_DIR", str(Path.home() / "code" / "OpenSearch-VL" / "SFT")))
    sys.path.insert(0, str(opensearch_sft_dir / "src"))


def _first_dataset_name(dataset_field: str | list[str]) -> str:
    if isinstance(dataset_field, str):
        return dataset_field.split(",")[0].strip()
    return str(dataset_field[0]).strip()


def _resolve_media_entries(entries: list[Any], dataset_root: Path) -> list[Any]:
    resolved: list[Any] = []
    for entry in entries:
        if isinstance(entry, str):
            path = Path(entry)
            resolved.append(str(path if path.is_absolute() else (dataset_root / path)))
        elif isinstance(entry, list):
            resolved.append(_resolve_media_entries(entry, dataset_root))
        else:
            resolved.append(entry)
    return resolved


def _flatten_encoded_pairs(encoded_pairs: list[tuple[list[int], list[int]]]) -> tuple[list[int], list[int]]:
    prompt_ids: list[int] = []
    response_ids: list[int] = []
    for source_ids, target_ids in encoded_pairs:
        prompt_ids.extend(source_ids)
        response_ids.extend(target_ids)
    return prompt_ids, response_ids


def _map_messages(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    from llamafactory.data.data_utils import Role

    role_map = {
        "human": Role.USER,
        "gpt": Role.ASSISTANT,
        "observation": Role.OBSERVATION,
        "function_call": Role.FUNCTION,
    }
    return [{"role": role_map[item["from"]], "content": item["value"]} for item in conversations]


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {key: _serialize_value(val) for key, val in value.__dict__.items()}
    return value


def _extract_video_debug(mm_inputs: dict[str, Any], processor: Any, template: Any) -> dict[str, Any]:
    video_grid_thw = mm_inputs.get("video_grid_thw", [])
    video_metadata = mm_inputs.get("video_metadata", [])
    if len(video_grid_thw) == 0:
        return {
            "sampled_frames": 0,
            "video_grid_thw": [],
            "video_soft_tokens_per_frame": 0,
            "video_soft_tokens_total": 0,
            "timestamps_seconds": [],
        }

    video_processor = getattr(processor, "video_processor")
    merge_length = getattr(video_processor, "merge_size") ** 2
    patch_size = int(getattr(video_processor, "patch_size"))
    temporal_patch_size = int(getattr(video_processor, "temporal_patch_size", 1))
    merge_size = int(getattr(video_processor, "merge_size"))
    grid = video_grid_thw[0]
    sampled_frames = int(grid[0])
    soft_tokens_per_frame = int((int(grid[1]) * int(grid[2])) // merge_length)
    soft_tokens_total = soft_tokens_per_frame * sampled_frames

    timestamps: list[float] = []
    if len(video_metadata) > 0:
        metadata = video_metadata[0]
        frames_indices = getattr(metadata, "frames_indices", None)
        fps = getattr(metadata, "fps", None)
        duration = getattr(metadata, "duration", None)
        total_num_frames = getattr(metadata, "total_num_frames", None)
        width = getattr(metadata, "width", None)
        height = getattr(metadata, "height", None)
        if frames_indices is None and isinstance(metadata, dict):
            frames_indices = metadata.get("frames_indices")
            fps = metadata.get("fps")
            duration = metadata.get("duration")
            total_num_frames = metadata.get("total_num_frames")
            width = metadata.get("width")
            height = metadata.get("height")
        if frames_indices is not None and fps is not None:
            timestamps = [
                float(round(float(ts), 1))
                for ts in processor._calculate_timestamps(frames_indices, fps, video_processor.merge_size)
            ]
        sampled_raw_frames = len(frames_indices) if frames_indices is not None else None
    else:
        duration = None
        total_num_frames = None
        width = None
        height = None
        sampled_raw_frames = None

    resized_height = int(grid[1]) * patch_size
    resized_width = int(grid[2]) * patch_size

    return {
        "sampled_frames": sampled_frames,
        "sampled_raw_frames": sampled_raw_frames,
        "video_grid_thw": _serialize_value(grid),
        "video_patch_size": patch_size,
        "video_merge_size": merge_size,
        "video_temporal_patch_size": temporal_patch_size,
        "resized_frame_height": resized_height,
        "resized_frame_width": resized_width,
        "video_soft_tokens_per_frame": soft_tokens_per_frame,
        "video_soft_tokens_total": soft_tokens_total,
        "timestamps_seconds": timestamps,
        "video_duration_seconds": duration,
        "video_total_num_frames_after_sampling": total_num_frames,
        "video_metadata_width": width,
        "video_metadata_height": height,
    }


def _count_placeholder_strings(text: str) -> dict[str, int]:
    keys = [
        "<video>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|video_pad|>",
        "<|image_pad|>",
        "<think>",
        "</think>",
        "<answer>",
        "</answer>",
    ]
    counts = Counter()
    for key in keys:
        counts[key] = text.count(key)
    return dict(counts)


def load_pipeline(config_path: str | Path) -> dict[str, Any]:
    root_dir = find_repo_root(Path(config_path).resolve())
    _ensure_llamafactory_importable(root_dir)

    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams.parser import _parse_train_args
    from llamafactory.model.loader import load_tokenizer

    config_path = Path(config_path).resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(raw_config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_processor = SupervisedDatasetProcessor(
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
    )

    registry_path = root_dir / "SFT" / "data" / "dataset_info.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    dataset_name = _first_dataset_name(data_args.dataset)
    dataset_file = registry_path.parent / registry[dataset_name]["file_name"]
    dataset_root = dataset_file.parent
    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))

    resolved_config = {
        "config_path": str(config_path),
        "dataset": dataset_name,
        "model_name_or_path": model_args.model_name_or_path,
        "template": data_args.template,
        "cutoff_len": data_args.cutoff_len,
        "image_max_pixels": model_args.image_max_pixels,
        "image_min_pixels": model_args.image_min_pixels,
        "video_max_pixels": model_args.video_max_pixels,
        "video_min_pixels": model_args.video_min_pixels,
        "video_fps": model_args.video_fps,
        "video_maxlen": model_args.video_maxlen,
        "bf16": training_args.bf16,
        "torch_compile": getattr(training_args, "torch_compile", None),
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
    }

    return {
        "root_dir": root_dir,
        "config_path": config_path,
        "raw_config": raw_config,
        "resolved_config": resolved_config,
        "model_args": model_args,
        "data_args": data_args,
        "training_args": training_args,
        "finetuning_args": finetuning_args,
        "generating_args": generating_args,
        "tokenizer": tokenizer,
        "processor": processor,
        "template": template,
        "dataset_processor": dataset_processor,
        "dataset_name": dataset_name,
        "dataset_file": dataset_file,
        "dataset_root": dataset_root,
        "dataset": dataset,
    }


def analyze_example(config_path: str | Path, index: int) -> dict[str, Any]:
    pipeline = load_pipeline(config_path)
    example = pipeline["dataset"][index]
    tokenizer = pipeline["tokenizer"]
    processor = pipeline["processor"]
    template = pipeline["template"]
    dataset_processor = pipeline["dataset_processor"]
    dataset_root = pipeline["dataset_root"]
    data_args = pipeline["data_args"]

    mapped_messages = _map_messages(example["conversations"])
    prompt_messages = mapped_messages[:-1]
    response_messages = mapped_messages[-1:]

    images = _resolve_media_entries(example.get("images", []) or [], dataset_root)
    videos = _resolve_media_entries(example.get("videos", []) or [], dataset_root)
    audios: list[str] = []

    raw_pairs = template.encode_multiturn(tokenizer, prompt_messages + response_messages, example.get("system"), example.get("tools"))
    raw_prompt_ids, raw_response_ids = _flatten_encoded_pairs(raw_pairs)

    processed_messages = template.mm_plugin.process_messages(
        prompt_messages + response_messages,
        images,
        videos,
        audios,
        processor,
    )
    expanded_pairs = template.encode_multiturn(
        tokenizer,
        processed_messages,
        example.get("system"),
        example.get("tools"),
    )
    expanded_prompt_ids, expanded_response_ids = _flatten_encoded_pairs(expanded_pairs)
    full_input_ids, labels = dataset_processor._encode_data_example(
        prompt=prompt_messages,
        response=response_messages,
        system=example.get("system"),
        tools=example.get("tools"),
        images=images,
        videos=videos,
        audios=audios,
    )
    mm_inputs = template.mm_plugin._get_mm_inputs(images, videos, audios, processor)  # type: ignore[attr-defined]
    video_debug = _extract_video_debug(mm_inputs, processor, template)
    valid_label_ids = [label for label in labels if label != IGNORE_INDEX]

    expanded_prompt_text = "\n\n".join(
        f"{message['role']}:\n{message['content']}" for message in processed_messages[:-1]
    )
    raw_prompt_text = "\n\n".join(
        f"{message['role']}:\n{message['content']}" for message in prompt_messages
    )

    summary = {
        "example_index": index,
        "video_paths": videos,
        "image_paths": images,
        "raw_prompt_tokens": len(raw_prompt_ids),
        "video_added_prompt_tokens": len(expanded_prompt_ids) - len(raw_prompt_ids),
        "expanded_prompt_tokens": len(expanded_prompt_ids),
        "assistant_response_tokens": len(expanded_response_ids),
        "expanded_total_tokens_before_cutoff": len(expanded_prompt_ids) + len(expanded_response_ids),
        "final_input_ids_after_cutoff": len(full_input_ids),
        "supervised_target_tokens": len(valid_label_ids),
        "cutoff_len": data_args.cutoff_len,
    }
    summary.update(video_debug)
    summary["video_non_soft_overhead_tokens"] = summary["video_added_prompt_tokens"] - summary["video_soft_tokens_total"]
    summary["placeholder_counts"] = _count_placeholder_strings(analysis_text := f"{expanded_prompt_text}\n\n{response_messages[-1]['content']}")

    return {
        "resolved_config": pipeline["resolved_config"],
        "dataset_file": str(pipeline["dataset_file"]),
        "dataset_root": str(dataset_root),
        "example_index": index,
        "example": example,
        "raw_prompt_text": raw_prompt_text,
        "assistant_text": response_messages[-1]["content"],
        "expanded_prompt_text": expanded_prompt_text,
        "analysis_text": analysis_text,
        "processed_messages": processed_messages,
        "summary": summary,
        "input_ids": full_input_ids,
        "labels": labels,
        "valid_label_ids": valid_label_ids,
        "decoded_input": tokenizer.decode(full_input_ids, skip_special_tokens=False),
        "decoded_target": tokenizer.decode(valid_label_ids, skip_special_tokens=False),
        "tokenizer": tokenizer,
    }


def build_modality_breakdown(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    summary = analysis["summary"]
    rows = [
        {
            "component": "Raw user text",
            "tokens": summary["raw_prompt_tokens"],
            "why": "Question text plus chat-template text before multimodal expansion.",
        },
        {
            "component": "Video soft tokens",
            "tokens": summary["video_soft_tokens_total"],
            "why": "Learned per-frame visual patch placeholders derived from video_grid_thw and merge size.",
        },
        {
            "component": "Video overhead tokens",
            "tokens": summary["video_non_soft_overhead_tokens"],
            "why": "Timestamp text plus <|vision_start|>/<|vision_end|> wrappers around each sampled frame block.",
        },
        {
            "component": "Assistant target",
            "tokens": summary["assistant_response_tokens"],
            "why": "Supervised reasoning trace and answer tokens that contribute to loss.",
        },
    ]
    rows.append(
        {
            "component": "Expanded total",
            "tokens": summary["expanded_total_tokens_before_cutoff"],
            "why": "Prompt plus assistant after multimodal expansion, before cutoff_len truncation.",
        }
    )
    return rows


def build_token_rows(analysis: dict[str, Any], start: int = 0, end: int = 256) -> list[dict[str, Any]]:
    tokenizer = analysis["tokenizer"]
    input_ids = analysis["input_ids"]
    labels = analysis["labels"]
    rows: list[dict[str, Any]] = []
    upper = min(end, len(input_ids))
    for idx in range(start, upper):
        label_id = labels[idx]
        rows.append(
            {
                "position": idx,
                "input_id": input_ids[idx],
                "token": tokenizer.convert_ids_to_tokens(input_ids[idx]),
                "label_id": None if label_id == IGNORE_INDEX else label_id,
                "label_token": None if label_id == IGNORE_INDEX else tokenizer.convert_ids_to_tokens(label_id),
                "is_supervised": label_id != IGNORE_INDEX,
            }
        )

    return rows


def build_mask_segments(analysis: dict[str, Any], max_segments: int = 24) -> list[dict[str, Any]]:
    tokenizer = analysis["tokenizer"]
    input_ids = analysis["input_ids"]
    labels = analysis["labels"]
    if not input_ids:
        return []

    segments: list[dict[str, Any]] = []
    start = 0
    current_supervised = labels[0] != IGNORE_INDEX
    for idx in range(1, len(labels)):
        is_supervised = labels[idx] != IGNORE_INDEX
        if is_supervised != current_supervised:
            segment_ids = input_ids[start:idx]
            segments.append(
                {
                    "start": start,
                    "end": idx,
                    "length": idx - start,
                    "is_supervised": current_supervised,
                    "text_preview": tokenizer.decode(segment_ids, skip_special_tokens=False)[:240],
                }
            )
            start = idx
            current_supervised = is_supervised

    segment_ids = input_ids[start:]
    segments.append(
        {
            "start": start,
            "end": len(labels),
            "length": len(labels) - start,
            "is_supervised": current_supervised,
            "text_preview": tokenizer.decode(segment_ids, skip_special_tokens=False)[:240],
        }
    )
    return segments[:max_segments]
