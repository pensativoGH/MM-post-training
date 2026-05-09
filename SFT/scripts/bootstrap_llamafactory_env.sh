#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${ENV_DIR:-$ROOT_DIR/.venv_sft}"
OPENSEARCH_SFT_DIR="${OPENSEARCH_SFT_DIR:-$ROOT_DIR/../OpenSearch-VL/SFT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

if [[ ! -f "$OPENSEARCH_SFT_DIR/pyproject.toml" ]]; then
  echo "Could not find LLaMA-Factory-compatible source tree at $OPENSEARCH_SFT_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "$OPENSEARCH_SFT_DIR"
python -m pip install qwen-vl-utils pillow av
python -m pip install decord || echo "Skipping decord install; no compatible wheel found for this platform."

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  python -m pip install flash-attn
fi

echo "Bootstrapped SFT environment at $ENV_DIR"
echo "activate: source $ENV_DIR/bin/activate"
echo "train:    $ROOT_DIR/SFT/scripts/run_local_sft.sh"
