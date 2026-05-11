#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/SFT/examples/local/robovqa_cosmos_sft_qwen3_vl_4b_full_q1_pipeline.yaml}"
PIPELINE_CONFIG_PATH="${PIPELINE_CONFIG_PATH:-${2:-}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINER_LAUNCHER="${TRAINER_LAUNCHER:-$ROOT_DIR/SFT/scripts/run_dgx_spark_sft.sh}"
TRAIN_DATASET_VERSION_ID="${ROBOVQA_TRAIN_DATASET_VERSION_ID:-}"
VAL_DATASET_VERSION_ID="${ROBOVQA_VAL_DATASET_VERSION_ID:-}"
PARENT_DATASET_NAME="${ROBOVQA_PARENT_DATASET_NAME:-robovqa_cosmos_sft_q1}"
TRAIN_DATASET_NAME="${ROBOVQA_SFT_TRAIN_DATASET_NAME:-robovqa_cosmos_sft_q1_train}"
VAL_DATASET_NAME="${ROBOVQA_SFT_VAL_DATASET_NAME:-robovqa_cosmos_sft_q1_val}"
DATA_OUTPUT_ROOT="${ROBOVQA_SFT_OUTPUT_ROOT:-$ROOT_DIR/SFT/data/robovqa_cosmos_sft_q1}"
MANIFEST_PATH="$DATA_OUTPUT_ROOT/pipeline_sft_dataset_manifest.json"
TRAIN_JSON_PATH="$DATA_OUTPUT_ROOT/train.json"
VAL_JSON_PATH="$DATA_OUTPUT_ROOT/val.json"

resolve_pipeline_python() {
  if [[ -n "${PIPELINE_PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PIPELINE_PYTHON_BIN"
    return
  fi

  if [[ -n "$PIPELINE_CONFIG_PATH" ]]; then
    local config_dir
    config_dir="$(cd "$(dirname "$PIPELINE_CONFIG_PATH")" && pwd)"
    local candidate="$config_dir/.venv/bin/python"
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
    candidate="$(cd "$config_dir/.." && pwd)/.venv/bin/python"
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  fi

  printf '%s\n' "$PYTHON_BIN"
}

if [[ -z "$PIPELINE_CONFIG_PATH" ]]; then
  echo "Set PIPELINE_CONFIG_PATH or pass the pipeline config as the second argument." >&2
  exit 1
fi

if [[ -z "$TRAIN_DATASET_VERSION_ID" || -z "$VAL_DATASET_VERSION_ID" ]]; then
  echo "Set ROBOVQA_TRAIN_DATASET_VERSION_ID and ROBOVQA_VAL_DATASET_VERSION_ID." >&2
  exit 1
fi

PIPELINE_PYTHON="$(resolve_pipeline_python)"
if [[ ! -x "$PIPELINE_PYTHON" ]]; then
  echo "Could not find a usable pipeline Python: $PIPELINE_PYTHON" >&2
  exit 1
fi

extract_output_dir() {
  "$PYTHON_BIN" - "$CONFIG_PATH" "$ROOT_DIR" <<'PY'
from pathlib import Path
import re
import sys

config_path = Path(sys.argv[1])
root_dir = Path(sys.argv[2])
text = config_path.read_text(encoding="utf-8")
match = re.search(r"^output_dir:\s*(.+?)\s*$", text, flags=re.MULTILINE)
if not match:
    raise SystemExit("Could not find output_dir in config.")
value = match.group(1).strip()
path = Path(value)
print(str(path if path.is_absolute() else (root_dir / path)))
PY
}

publish_manifest() {
  local output_dir="$1"
  if [[ ! -f "$MANIFEST_PATH" ]]; then
    return
  fi

  mkdir -p "$output_dir"
  cp "$MANIFEST_PATH" "$output_dir/dataset_versions.json"

  while IFS= read -r checkpoint_dir; do
    cp "$MANIFEST_PATH" "$checkpoint_dir/dataset_versions.json"
  done < <(find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort)
}

build_media_mount_args() {
  "$PYTHON_BIN" - "$TRAIN_JSON_PATH" "$VAL_JSON_PATH" <<'PY'
from pathlib import Path
import json
import os
import sys

mount_roots = []
seen = set()

for dataset_path in [Path(arg) for arg in sys.argv[1:]]:
    if not dataset_path.is_file():
        continue
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    media_paths = []
    for row in rows:
        media_paths.extend(row.get("images") or [])
        media_paths.extend(row.get("videos") or [])
    if not media_paths:
        continue

    try:
        common = Path(os.path.commonpath(media_paths))
    except ValueError:
        common = None

    if common is None:
        for media_path in media_paths:
            parent = str(Path(media_path).resolve().parent)
            if parent not in seen:
                mount_roots.append(parent)
                seen.add(parent)
        continue

    common = common.resolve()
    mount_root = common if common.is_dir() else common.parent
    mount_root_str = str(mount_root)
    if mount_root_str not in seen:
        mount_roots.append(mount_root_str)
        seen.add(mount_root_str)

for root in mount_roots:
    print(f"-v {root}:{root}")
PY
}

"$PIPELINE_PYTHON" "$ROOT_DIR/SFT/scripts/prepare_robovqa_pipeline_sft.py" \
  --pipeline-config "$PIPELINE_CONFIG_PATH" \
  --train-dataset-version-id "$TRAIN_DATASET_VERSION_ID" \
  --val-dataset-version-id "$VAL_DATASET_VERSION_ID" \
  --parent-dataset-name "$PARENT_DATASET_NAME" \
  --train-dataset-name "$TRAIN_DATASET_NAME" \
  --val-dataset-name "$VAL_DATASET_NAME" \
  --output-root "$DATA_OUTPUT_ROOT"

OUTPUT_DIR="$(extract_output_dir)"
publish_manifest "$OUTPUT_DIR"

AUTO_MEDIA_DOCKER_ARGS="$(build_media_mount_args)"
if [[ -n "$AUTO_MEDIA_DOCKER_ARGS" ]]; then
  if [[ -n "${EXTRA_DOCKER_ARGS:-}" ]]; then
    export EXTRA_DOCKER_ARGS="$EXTRA_DOCKER_ARGS $AUTO_MEDIA_DOCKER_ARGS"
  else
    export EXTRA_DOCKER_ARGS="$AUTO_MEDIA_DOCKER_ARGS"
  fi
fi

status=0
if ! "$TRAINER_LAUNCHER" "$CONFIG_PATH"; then
  status=$?
fi

publish_manifest "$OUTPUT_DIR"
exit "$status"
