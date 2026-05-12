# Lambda Inference Sanity

This runbook documents the Lambda sanity setup used for V-JEPA2 and Wan2.2
inference. It intentionally skips DreamDojo until the server has a clean
DreamDojo/Cosmos environment, Hugging Face access for gated guardrail models,
and a matching GR1/DreamDojo eval dataset.

## Server Layout

Clone the repo into a fresh folder to avoid conflicts with other copies:

```bash
mkdir -p ~/inference
cd ~/inference
git clone https://github.com/pensativoGH/MM-post-training.git
cd MM-post-training
git checkout -b inference/sanity-world-models
```

Expected model/video locations from the Lambda host used for the check:

```text
/home/ubuntu/hf_models/vjepa2/direct_checkpoints/vitl.pt
/home/ubuntu/hf_models/Wan2.2/Wan2.2-TI2V-5B
/home/ubuntu/datasets/robovqa_cosmos_sft/robovqa/clips/*.mp4
```

## Environment

Create the venv, install the known-good V-JEPA/Wan dependencies, bootstrap the
pinned third-party repos, and create the aarch64 `decord` import shim:

```bash
bash world-model-post-training/scripts/setup_lambda_inference_env.sh
source .venv_infer/bin/activate
export PYTHONPATH="$PWD/outputs/inference_sanity/py_shims:${PYTHONPATH:-}"
```

The setup pins NumPy to `<1.25` because the server PyTorch stack used during
the sanity check was built against NumPy 1.x. Wan CLI imports `decord`, but no
usable aarch64 wheel was available; the shim is enough for non-S2V Wan tasks.

## V-JEPA2 Sanity

The verified check used the upstream V-JEPA2 checkout from `third_party/vjepa2`,
the local `vitl.pt` checkpoint, and a RoboVQA clip. The expected output is a
JSON summary containing an embedding tensor shape similar to `[1, 8192, 1024]`.

From the repo clone:

```bash
source .venv_infer/bin/activate
python world-model-post-training/scripts/run_lambda_vjepa2_sanity.py
```

Verified output path from the Lambda run:

```text
outputs/inference_sanity/vjepa2_embedding_summary.json
```

## Wan2.2 Sanity

From the repo clone:

```bash
source .venv_infer/bin/activate
export PYTHONPATH="$PWD/outputs/inference_sanity/py_shims:${PYTHONPATH:-}"
cd third_party/wan22

python generate.py \
  --task ti2v-5B \
  --size '704*1280' \
  --frame_num 5 \
  --ckpt_dir /home/ubuntu/hf_models/Wan2.2/Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --sample_steps 1 \
  --sample_shift 5 \
  --sample_guide_scale 1.0 \
  --base_seed 7 \
  --save_file /home/ubuntu/inference/MM-post-training/outputs/inference_sanity/wan_ti2v_5f_steps1.mp4 \
  --prompt "A short sanity-check video of a robot arm moving on a table."
```

Verified output path from the Lambda run:

```text
outputs/inference_sanity/wan_ti2v_5f_steps1.mp4
```

## DreamDojo Status

DreamDojo was skipped for now. The server has checkpoints, but the attempted
sanity setup exposed these blockers:

- The DreamDojo/Cosmos stack expects NumPy 2-era packages, while the system
  PyTorch stack emitted NumPy 1.x binary ABI warnings and failures.
- The action-conditioned path requires `pytorch3d`.
- Base inference imports gated `nvidia/Cosmos-Guardrail1` at import time and
  failed without Hugging Face auth.
- No GR1/DreamDojo eval dataset was found under `/home/ubuntu/datasets`.
