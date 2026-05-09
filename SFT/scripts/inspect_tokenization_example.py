#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tokenization_inspector import analyze_example, build_mask_segments, build_modality_breakdown, build_token_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one SFT example after multimodal tokenization.")
    parser.add_argument(
        "--config",
        default="SFT/examples/local/robovqa_cosmos_sft_qwen3_vl_4b_full_q1.yaml",
        help="Path to the SFT YAML config.",
    )
    parser.add_argument("--index", type=int, default=0, help="Dataset example index.")
    parser.add_argument("--token-start", type=int, default=0, help="First token row to print.")
    parser.add_argument("--token-end", type=int, default=128, help="Exclusive token row end.")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    analysis = analyze_example(args.config, args.index)
    payload = {
        "resolved_config": analysis["resolved_config"],
        "dataset_file": analysis["dataset_file"],
        "example_index": analysis["example_index"],
        "summary": analysis["summary"],
        "raw_prompt_text": analysis["raw_prompt_text"],
        "assistant_text": analysis["assistant_text"],
        "expanded_prompt_text": analysis["expanded_prompt_text"],
        "decoded_target": analysis["decoded_target"],
        "mask_segments": build_mask_segments(analysis),
        "modality_breakdown": build_modality_breakdown(analysis),
        "token_rows": build_token_rows(analysis, args.token_start, args.token_end),
    }
    print(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
