#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure SFT sequence lengths using the same template/mm expansion path as training."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="SFT/examples/local/robovqa_cosmos_sft_qwen3_vl_4b_full_q1.yaml",
        help="Repo-local SFT YAML config.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=512,
        help="Number of dataset examples to sample. Use 0 or a negative number for the full dataset.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    return parser.parse_args()


def summarize(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    n = len(ordered)
    return {
        "count": n,
        "mean": round(statistics.mean(ordered), 1),
        "median": round(statistics.median(ordered), 1),
        "p90": ordered[min(n - 1, int(0.90 * n))],
        "p95": ordered[min(n - 1, int(0.95 * n))],
        "p99": ordered[min(n - 1, int(0.99 * n))],
        "max": ordered[-1],
    }


def first_dataset_name(dataset_field: str | list[str]) -> str:
    if isinstance(dataset_field, str):
        return dataset_field.split(",")[0].strip()
    return str(dataset_field[0]).strip()


def resolve_media_entries(entries: list, dataset_root: Path) -> list:
    resolved = []
    for entry in entries:
        if isinstance(entry, str):
            path = Path(entry)
            resolved.append(str(path if path.is_absolute() else (dataset_root / path)))
        elif isinstance(entry, list):
            resolved.append(resolve_media_entries(entry, dataset_root))
        else:
            resolved.append(entry)
    return resolved


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[2]
    config_path = (root_dir / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)

    opensearch_sft_dir = Path(
        Path.home() / "code" / "OpenSearch-VL" / "SFT"
        if "OPENSEARCH_SFT_DIR" not in os.environ
        else os.environ["OPENSEARCH_SFT_DIR"]
    )
    sys.path.insert(0, str(opensearch_sft_dir / "src"))

    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.data_utils import Role
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments, TrainingArguments
    from llamafactory.hparams.parser import _parse_train_args
    from llamafactory.model.loader import load_tokenizer

    config_dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_args, data_args, _training_args, _finetuning_args, _generating_args = _parse_train_args(config_dict)
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
    dataset_name = first_dataset_name(data_args.dataset)
    dataset_file = registry_path.parent / registry[dataset_name]["file_name"]
    dataset_root = dataset_file.parent
    dataset_examples = json.loads(dataset_file.read_text(encoding="utf-8"))

    if args.sample_size and args.sample_size > 0 and args.sample_size < len(dataset_examples):
        rng = random.Random(args.seed)
        dataset_examples = rng.sample(dataset_examples, args.sample_size)

    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    expanded_total_lengths: list[int] = []
    truncated_lengths: list[int] = []
    video_counts: list[int] = []
    frame_counts: list[int] = []
    cutoff_hits = 0

    for example in dataset_examples:
        messages = []
        for turn in example["conversations"]:
            role = turn["from"]
            if role == "human":
                mapped_role = Role.USER
            elif role == "gpt":
                mapped_role = Role.ASSISTANT
            elif role == "observation":
                mapped_role = Role.OBSERVATION
            elif role == "function_call":
                mapped_role = Role.FUNCTION
            else:
                raise ValueError(f"Unsupported role: {role}")
            messages.append({"role": mapped_role, "content": turn["value"]})

        prompt = messages[:-1]
        response = messages[-1:]
        images = resolve_media_entries(example.get("images", []) or [], dataset_root)
        videos = resolve_media_entries(example.get("videos", []) or [], dataset_root)
        audios: list[str] = []

        processed_messages = template.mm_plugin.process_messages(
            prompt + response, images, videos, audios, processor
        )
        encoded_pairs = template.encode_multiturn(
            tokenizer,
            processed_messages,
            example.get("system"),
            example.get("tools"),
        )
        prompt_ids = []
        response_ids = []
        for source_ids, target_ids in encoded_pairs:
            prompt_ids.extend(source_ids)
            response_ids.extend(target_ids)

        full_ids, _ = dataset_processor._encode_data_example(
            prompt=prompt,
            response=response,
            system=example.get("system"),
            tools=example.get("tools"),
            images=images,
            videos=videos,
            audios=audios,
        )

        prompt_lengths.append(len(prompt_ids))
        response_lengths.append(len(response_ids))
        expanded_total_lengths.append(len(prompt_ids) + len(response_ids))
        truncated_lengths.append(len(full_ids))
        video_counts.append(len(videos))
        if len(videos) > 0:
            mm_inputs = template.mm_plugin._get_mm_inputs(images, videos, audios, processor)  # type: ignore[attr-defined]
            video_grid_thw = mm_inputs.get("video_grid_thw")
            if video_grid_thw is not None and len(video_grid_thw) > 0:
                frame_counts.append(int(video_grid_thw[0][0]))

        if len(full_ids) >= data_args.cutoff_len:
            cutoff_hits += 1

    report = {
        "config": str(config_path),
        "dataset": dataset_name,
        "examples_measured": len(dataset_examples),
        "cutoff_len": data_args.cutoff_len,
        "cutoff_hit_rate": round(cutoff_hits / max(len(dataset_examples), 1), 4),
        "prompt_tokens": summarize(prompt_lengths),
        "response_tokens": summarize(response_lengths),
        "expanded_total_tokens_before_cutoff": summarize(expanded_total_lengths),
        "final_input_ids_after_cutoff": summarize(truncated_lengths),
        "videos_per_example": summarize(video_counts),
        "sampled_frames_per_video": summarize(frame_counts) if frame_counts else None,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
