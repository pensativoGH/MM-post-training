#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runtime_common.sh"

MODEL_SELECTOR="${1:-thinking}"
MODEL_PATH="$(resolve_model_path "$MODEL_SELECTOR")"
MODEL_ID="$(resolve_model_id "$MODEL_SELECTOR")"
PORT="${PORT:-$(default_port_for "$MODEL_SELECTOR")}"
HOST="${HOST:-127.0.0.1}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
ENV_DIR="${ENV_DIR:-$HOME/.venvs/vllm}"
PID_FILE="${PID_FILE:-$(pid_file_for "$MODEL_SELECTOR")}"
LOG_FILE="${LOG_FILE:-$(log_file_for "$MODEL_SELECTOR")}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":8,\"video\":1}}"

require_local_model_if_path "$MODEL_PATH"

if [[ ! -x "$ENV_DIR/bin/vllm" ]]; then
  echo "vLLM executable not found in $ENV_DIR" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "vLLM already running with PID $OLD_PID"
    echo "base_url: http://$HOST:$PORT/v1"
    echo "log: $LOG_FILE"
    exit 0
  fi
fi

source "$ENV_DIR/bin/activate"

VLLM_ARGS=(
  serve "$MODEL_PATH"
  --served-model-name "$MODEL_ID"
  --host "$HOST"
  --port "$PORT"
  --dtype auto
  --trust-remote-code
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT"
  --chat-template-content-format openai
  --reasoning-parser "$REASONING_PARSER"
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
fi

nohup vllm "${VLLM_ARGS[@]}" >"$LOG_FILE" 2>&1 &
echo "$!" > "$PID_FILE"
sleep 2

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Started vLLM for $MODEL_ID on $HOST:$PORT"
  echo "pid: $PID_FILE"
  echo "log: $LOG_FILE"
  echo "base_url: http://$HOST:$PORT/v1"
else
  echo "vLLM failed to start. Check $LOG_FILE" >&2
  exit 1
fi
