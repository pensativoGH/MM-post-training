#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runtime/runs}"
mkdir -p "$RUN_DIR"

HF_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_ROOT="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"

DEFAULT_INSTRUCT_ID="Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_THINKING_ID="Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_32B_THINKING_ID="Qwen/Qwen3-VL-32B-Thinking"

default_port_for() {
  local selector="${1:-thinking}"
  case "$selector" in
    thinking) printf '%s\n' "8010" ;;
    instruct) printf '%s\n' "8011" ;;
    thinking32b) printf '%s\n' "8012" ;;
    *) printf '%s\n' "8010" ;;
  esac
}

resolve_model_id() {
  local selector="${1:-thinking}"
  case "$selector" in
    thinking) printf '%s\n' "$DEFAULT_THINKING_ID" ;;
    instruct) printf '%s\n' "$DEFAULT_INSTRUCT_ID" ;;
    thinking32b) printf '%s\n' "$DEFAULT_32B_THINKING_ID" ;;
    Qwen/*) printf '%s\n' "$selector" ;;
    *) printf '%s\n' "$selector" ;;
  esac
}

resolve_model_path() {
  local selector="${1:-thinking}"
  case "$selector" in
    thinking)
      printf '%s\n' "${THINKING_MODEL_PATH:-$DEFAULT_THINKING_ID}"
      ;;
    instruct)
      printf '%s\n' "${INSTRUCT_MODEL_PATH:-$DEFAULT_INSTRUCT_ID}"
      ;;
    thinking32b)
      printf '%s\n' "${THINKING32B_MODEL_PATH:-$DEFAULT_32B_THINKING_ID}"
      ;;
    /*)
      printf '%s\n' "$selector"
      ;;
    Qwen/*)
      printf '%s\n' "$selector"
      ;;
    *)
      printf '%s\n' "$selector"
      ;;
  esac
}

require_local_model_if_path() {
  local model_path="$1"
  if [[ "$model_path" == /* ]] && [[ ! -d "$model_path" ]]; then
    echo "Missing local model directory: $model_path" >&2
    exit 1
  fi
}

pid_file_for() {
  local selector="$1"
  printf '%s/vllm_%s.pid\n' "$RUN_DIR" "$selector"
}

log_file_for() {
  local selector="$1"
  printf '%s/vllm_%s.log\n' "$RUN_DIR" "$selector"
}
