#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/RL/configs/procedural_grpo_smoke_video_vllm.yaml}"

python3 "$ROOT_DIR/RL/scripts/prepare_rl_dataset.py" --root-dir "$ROOT_DIR"
python3 "$ROOT_DIR/RL/scripts/run_local_grpo.py" "$CONFIG_PATH"
