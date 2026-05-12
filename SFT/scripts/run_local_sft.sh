#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/SFT/examples/local/qwen3_vl_video_sft_8b.yaml}"
OPENSEARCH_SFT_DIR="${OPENSEARCH_SFT_DIR:-$ROOT_DIR/../OpenSearch-VL/SFT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT_DIR/post_training/src${PYTHONPATH:+:$PYTHONPATH}"

resolve_trainer_dispatch() {
  "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
import yaml

from verl_post_training.adapters.trainer import resolve_trainer_adapter
from verl_post_training.launch.dispatch import resolve_dispatch
from verl_post_training.launch.load_config import TaskConfig

config_path = Path(sys.argv[1])
raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
model_id = str(raw.get("model_name_or_path") or "").strip()
if not model_id:
    raise SystemExit("SFT config must define model_name_or_path for trainer dispatch.")

task_config = TaskConfig.from_mapping(
    {
        "task_type": "chat_sft",
        "model_id": model_id,
        "trainer_backend": "llamafactory",
        "dataset_adapter": "chat_sft",
        "input_manifest": str(config_path),
        "output_dir": str(raw.get("output_dir") or "outputs/sft/local"),
        "launcher": {"kind": "llamafactory-cli"},
        "resources": {"precision": "bf16" if raw.get("bf16", False) else "fp32"},
        "backend_config": {"config_file": str(config_path)},
    }
)
plan = resolve_dispatch(task_config)
adapter = resolve_trainer_adapter(plan)
print(f"Resolved trainer dispatch: {adapter.adapter_key} handles {plan.task_type.value} for {plan.model_entry.model_family.value}")
PY
}

extract_dataset_names() {
  "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import re
import sys

config_path = Path(sys.argv[1])
text = config_path.read_text(encoding="utf-8")
names = []
for key in ("dataset", "eval_dataset"):
    match = re.search(rf"^{key}:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not match:
        continue
    value = match.group(1).strip()
    if not value or value.startswith("#"):
        continue
    for name in value.split(","):
        cleaned = name.strip()
        if cleaned:
            names.append(cleaned)

seen = set()
for name in names:
    if name not in seen:
        print(name)
        seen.add(name)
PY
}

run_validation() {
  local dataset_name
  mapfile -t dataset_names < <(extract_dataset_names)
  if [[ "${#dataset_names[@]}" -eq 0 ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/SFT/scripts/validate_dataset.py"
    return
  fi

  for dataset_name in "${dataset_names[@]}"; do
    "$PYTHON_BIN" "$ROOT_DIR/SFT/scripts/validate_dataset.py" --dataset "$dataset_name"
  done
}

run_with_local_install() {
  llamafactory-cli train "$CONFIG_PATH"
}

run_with_vendored_copy() {
  if [[ ! -d "$OPENSEARCH_SFT_DIR/src/llamafactory" ]]; then
    echo "Could not find vendored LLaMA-Factory under $OPENSEARCH_SFT_DIR" >&2
    return 1
  fi

  PYTHONPATH="$OPENSEARCH_SFT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m llamafactory.cli train "$CONFIG_PATH"
}

resolve_trainer_dispatch
run_validation

if command -v llamafactory-cli >/dev/null 2>&1; then
  echo "Using llamafactory-cli from PATH"
  run_with_local_install
  exit 0
fi

echo "llamafactory-cli not found; falling back to vendored source at $OPENSEARCH_SFT_DIR"
run_with_vendored_copy
