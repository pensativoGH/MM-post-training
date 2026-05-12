#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv_infer}"
SHIM_DIR="${ROOT_DIR}/outputs/inference_sanity/py_shims"

python3 -m venv --system-site-packages "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install \
  "numpy>=1.23.5,<1.25" \
  "transformers==4.51.3" \
  accelerate \
  diffusers \
  opencv-python \
  imageio \
  imageio-ffmpeg \
  ftfy \
  safetensors \
  timm \
  iopath \
  submitit \
  braceexpand \
  webdataset \
  peft \
  beartype \
  python-box \
  fire \
  tyro \
  pydantic \
  huggingface_hub \
  librosa \
  soundfile \
  dashscope

bash "${ROOT_DIR}/world-model-post-training/scripts/bootstrap_third_party.sh"

mkdir -p "${SHIM_DIR}/decord"
cat > "${SHIM_DIR}/decord/__init__.py" <<'PY'
class VideoReader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("decord shim: VideoReader is unavailable in this environment")
PY

python - <<'PY'
import numpy
import torch
import transformers

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"numpy={numpy.__version__}")
print(f"transformers={transformers.__version__}")
PY

cat <<EOF

Inference environment ready.

Activate:
  source ${VENV_DIR}/bin/activate

Wan decord shim:
  export PYTHONPATH=${SHIM_DIR}:\${PYTHONPATH:-}
EOF
