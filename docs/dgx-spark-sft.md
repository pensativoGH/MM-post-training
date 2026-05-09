# DGX Spark SFT

This repo keeps the default SFT path local and simple, but DGX Spark benefits
from running the same workflow inside NVIDIA's PyTorch container rather than a
host Python environment.

The launcher is:

```bash
./SFT/scripts/run_dgx_spark_sft.sh
```

By default it uses:

- `nvcr.io/nvidia/pytorch:25.11-py3`
- `--gpus all`
- `--ipc=host`
- persistent Hugging Face, pip, uv, and TorchInductor caches
- the vendored LLaMA-Factory tree at `../OpenSearch-VL/SFT`

## Why this path exists

DGX Spark is a Grace Blackwell unified-memory system, so a few things matter
more than on a standard workstation:

- use NVIDIA's tuned PyTorch container first
- keep `bf16: true`
- keep small `per_device_train_batch_size` and raise
  `gradient_accumulation_steps` instead
- use `torch_compile: true` on longer runs, but not tiny one-step smoke runs
- keep compile caches persistent across runs

The new launcher preserves the same YAML-driven SFT workflow and simply runs it
inside the container.

For containerized runs, prefer model IDs such as `Qwen/Qwen3-VL-4B-Thinking`
over hardcoded local snapshot paths. Hugging Face snapshot directories are
symlink-heavy, and model IDs are more robust inside Docker as long as the HF
cache is mounted.

## Examples

Run the default RoboVQA full-SFT config:

```bash
./SFT/scripts/run_dgx_spark_sft.sh
```

Run a specific config:

```bash
./SFT/scripts/run_dgx_spark_sft.sh SFT/examples/local/qwen3_vl_video_sft_8b.yaml
```

Force a fresh in-container bootstrap:

```bash
FORCE_BOOTSTRAP=1 ./SFT/scripts/run_dgx_spark_sft.sh
```

Use a different NGC image:

```bash
IMAGE=nvcr.io/nvidia/pytorch:25.10-py3 ./SFT/scripts/run_dgx_spark_sft.sh
```

## Spark-specific knobs

Useful environment overrides:

- `HF_HOME`: mount a different Hugging Face cache
- `TORCHINDUCTOR_CACHE_DIR`: persist `torch.compile` artifacts elsewhere
- `PYTORCH_CUDA_ALLOC_CONF`: defaults to `expandable_segments:True`
- `OMP_NUM_THREADS`: defaults to `8`
- `INSTALL_FLASH_ATTN=1`: attempt flash-attn install during bootstrap
- `EXTRA_DOCKER_ARGS`: add extra binds for external absolute media roots

Example for datasets that reference host-absolute media paths outside this repo:

```bash
EXTRA_DOCKER_ARGS="-v /path/to/datasets:/path/to/datasets -v /path/to/procedural-grpo:/path/to/procedural-grpo" \
  ./SFT/scripts/run_dgx_spark_sft.sh
```

## Cache pressure

DGX Spark uses unified memory, so failed large-model loads can leave the system
in a bad cache state. If you see memory pressure after an OOM or repeated model
loads, you can retry with:

```bash
DROP_CACHES_BEFORE_RUN=1 ./SFT/scripts/run_dgx_spark_sft.sh
```

That calls `sync` and `echo 3 > /proc/sys/vm/drop_caches` before starting the
container. It is intentionally opt-in because it requires elevated privileges.

## Current recommendation for this repo

For the active RoboVQA full-SFT config:

- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 4`
- `bf16: true`
- `torch_compile: true`
- `model_name_or_path: Qwen/Qwen3-VL-4B-Thinking`

That keeps the effective batch size steady while lowering peak memory demand on
Spark.
