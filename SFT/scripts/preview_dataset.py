#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview one local multimodal SFT dataset example.")
    parser.add_argument("--dataset", default="video_agent_sft_demo")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    registry_path = repo_root / "SFT" / "data" / "dataset_info.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry[args.dataset]
    dataset_path = registry_path.parent / entry["file_name"]
    examples = json.loads(dataset_path.read_text(encoding="utf-8"))
    sample = examples[0]
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "path": str(dataset_path),
                "num_examples": len(examples),
                "sample": sample,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
