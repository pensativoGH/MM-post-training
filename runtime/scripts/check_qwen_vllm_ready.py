#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "runtime" / "src"))

from verl_post_training_runtime.local_runtime import probe_openai_compatible_runtime, resolve_local_runtime_spec


def main() -> int:
    parser = argparse.ArgumentParser(description="Readiness probe for local multimodal vLLM runtime.")
    parser.add_argument("--selector", default="thinking", choices=["thinking", "instruct", "thinking32b"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    args = parser.parse_args()

    spec = resolve_local_runtime_spec(args.model or args.selector)
    result = probe_openai_compatible_runtime(
        base_url=args.base_url or spec["base_url"],
        expected_model=args.model or spec["model_id"],
        api_key=args.api_key,
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
