#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_ARG="${1:-$ROOT_DIR/SFT/examples/local/robovqa_cosmos_sft_qwen3_vl_4b_full_q1.yaml}"

if [[ -f "$CONFIG_ARG" ]]; then
  CONFIG_PATH="$(cd "$(dirname "$CONFIG_ARG")" && pwd)/$(basename "$CONFIG_ARG")"
else
  CONFIG_PATH="$ROOT_DIR/$CONFIG_ARG"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Could not find config: $CONFIG_ARG" >&2
  exit 1
fi

case "$CONFIG_PATH" in
  "$ROOT_DIR"/*) CONFIG_IN_CONTAINER="/workspace/verl-post-training/${CONFIG_PATH#$ROOT_DIR/}" ;;
  *)
    echo "Config must live under $ROOT_DIR so it can be mounted into the container." >&2
    exit 1
    ;;
esac

OPENSEARCH_ROOT="${OPENSEARCH_ROOT:-$ROOT_DIR/../OpenSearch-VL}"
OPENSEARCH_SFT_DIR="$OPENSEARCH_ROOT/SFT"
if [[ ! -f "$OPENSEARCH_SFT_DIR/pyproject.toml" ]]; then
  echo "Could not find vendored LLaMA-Factory tree at $OPENSEARCH_SFT_DIR" >&2
  exit 1
fi

DOCKER_BIN="${DOCKER_BIN:-docker}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3}"
CONTAINER_REPO_ROOT="/workspace/verl-post-training"
CONTAINER_OPENSEARCH_ROOT="/workspace/OpenSearch-VL"
CONTAINER_OPENSEARCH_SFT_DIR="$CONTAINER_OPENSEARCH_ROOT/SFT"
ENV_DIR="${ENV_DIR:-$ROOT_DIR/.venv_sft_ngc}"
ENV_DIR_IN_CONTAINER="$CONTAINER_REPO_ROOT/${ENV_DIR#$ROOT_DIR/}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
FORCE_BOOTSTRAP="${FORCE_BOOTSTRAP:-0}"
DROP_CACHES_BEFORE_RUN="${DROP_CACHES_BEFORE_RUN:-0}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$UV_CACHE_DIR"

if [[ "$DROP_CACHES_BEFORE_RUN" == "1" ]]; then
  echo "Dropping Linux page cache before run (requires sudo/root)."
  sync
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
fi

if ! "$DOCKER_BIN" info >/dev/null 2>&1; then
  echo "Docker is installed but this shell cannot access the Docker daemon." >&2
  echo "Current user: $(id -un)" >&2
  echo "Docker socket: /var/run/docker.sock" >&2
  echo "Fix by running with a user in the docker group, or run the launcher via sudo." >&2
  exit 1
fi

readarray -t AUTO_MOUNT_ARGS < <("$PYTHON_BIN" - "$ROOT_DIR" "$CONFIG_PATH" "$CONTAINER_REPO_ROOT" <<'PY'
from pathlib import Path
import json
import re
import sys

root_dir = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
container_root = Path(sys.argv[3])
text = config_path.read_text(encoding="utf-8")
m = re.search(r"^dataset:\s*(.+?)\s*$", text, flags=re.MULTILINE)
if not m:
    raise SystemExit(0)

datasets = [name.strip() for name in m.group(1).split(",") if name.strip()]
registry = json.loads((root_dir / "SFT" / "data" / "dataset_info.json").read_text(encoding="utf-8"))

for dataset_name in datasets:
    entry = registry.get(dataset_name)
    if not entry:
        continue
    dataset_file = root_dir / "SFT" / "data" / entry["file_name"]
    dataset_root = dataset_file.parent
    dataset_root_in_container = container_root / dataset_root.relative_to(root_dir)
    summary_path = dataset_root / "summary.json"
    if not summary_path.is_file():
        continue

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_media_root = summary.get("source_media_root")
    if not source_media_root:
        continue

    host_media_root = Path(source_media_root)
    host_clips_dir = host_media_root / "clips"
    if host_clips_dir.is_dir():
        print(f"-v\n{host_clips_dir}:{dataset_root_in_container / 'clips'}")
PY
)

EXTRA_ARGS=()
if [[ -n "$EXTRA_DOCKER_ARGS" ]]; then
  read -r -a EXTRA_ARGS <<< "$EXTRA_DOCKER_ARGS"
fi

read -r -d '' INNER_CMD <<EOF || true
set -euo pipefail
cd "$CONTAINER_REPO_ROOT"
export HF_HOME=/cache/huggingface
export PIP_CACHE_DIR=/cache/pip
export UV_CACHE_DIR=/cache/uv
export TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENABLE_PARALLEL_LOADING="\${HF_ENABLE_PARALLEL_LOADING:-true}"
export TOKENIZERS_PARALLELISM="\${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-8}"
export OPENSEARCH_SFT_DIR="$CONTAINER_OPENSEARCH_SFT_DIR"
export ENV_DIR="$ENV_DIR_IN_CONTAINER"
export PYTHON_BIN=python3
export INSTALL_FLASH_ATTN="$INSTALL_FLASH_ATTN"

if [[ "$FORCE_BOOTSTRAP" == "1" || ! -x "$ENV_DIR_IN_CONTAINER/bin/python" ]]; then
  ./SFT/scripts/bootstrap_llamafactory_env.sh
fi

source "$ENV_DIR_IN_CONTAINER/bin/activate"
./SFT/scripts/run_local_sft.sh "$CONFIG_IN_CONTAINER"
EOF

exec "$DOCKER_BIN" run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_HOME=/cache/huggingface \
  -e PIP_CACHE_DIR=/cache/pip \
  -e UV_CACHE_DIR=/cache/uv \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -e HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}" \
  -e TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
  -v "$ROOT_DIR:$CONTAINER_REPO_ROOT" \
  -v "$ROOT_DIR:$ROOT_DIR" \
  -v "$OPENSEARCH_ROOT:$CONTAINER_OPENSEARCH_ROOT" \
  -v "$OPENSEARCH_ROOT:$OPENSEARCH_ROOT" \
  -v "$HF_HOME:/cache/huggingface" \
  -v "$HF_HOME:$HF_HOME" \
  -v "$PIP_CACHE_DIR:/cache/pip" \
  -v "$UV_CACHE_DIR:/cache/uv" \
  -v "$TORCHINDUCTOR_CACHE_DIR:/cache/torchinductor" \
  "${AUTO_MOUNT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  -w "$CONTAINER_REPO_ROOT" \
  "$IMAGE" \
  bash -lc "$INNER_CMD"
