# SFT

Local-first multimodal supervised fine-tuning using LLaMA-Factory.

This subtree stays thin, matching the role that `SFT/` plays in
`OpenSearch-VL`:

- dataset registry and schemas
- training YAMLs
- small helper scripts
- minimal project-specific glue

## Local Default

Use a standard local LLaMA-Factory training path rather than Ray:

```bash
./SFT/scripts/run_local_sft.sh
```

## Data Direction

The first target format is ShareGPT-style multimodal chat data with native
LLaMA-Factory `images` and `videos` columns:

- `messages`
- `images`
- `videos`
- `system`
- `tools`

The live registry is [SFT/data/dataset_info.json](data/dataset_info.json).

## Local Demo

The repo now includes a tiny runnable demo dataset:

- one image sample
- one video sample represented as a list of frame files

Useful commands:

```bash
./SFT/scripts/bootstrap_llamafactory_env.sh
python3 SFT/scripts/adapt_procedural_grpo_smoke.py
python3 SFT/scripts/validate_dataset.py
python3 SFT/scripts/preview_dataset.py
./SFT/scripts/run_local_sft.sh
./SFT/scripts/run_dgx_spark_sft.sh
```

For the pipeline-backed RoboVQA train/val SFT path that runs through the DGX
Spark Docker launcher, use:

```bash
PIPELINE_CONFIG_PATH=/home/pensativo/code/multimodal-data-pipeline-clean/configs/videos_all.yaml \
  ROBOVQA_TRAIN_DATASET_VERSION_ID=976cb46bc108301f0b787b41 \
  ROBOVQA_VAL_DATASET_VERSION_ID=a2f9ef2238a327ed3ee5fc2a \
  ./SFT/scripts/run_robovqa_pipeline_sft.sh
```

`run_local_sft.sh` uses `llamafactory-cli` from `PATH` if installed. If not, it
falls back to the vendored source tree at `../OpenSearch-VL/SFT`
by setting `PYTHONPATH` and invoking `python -m llamafactory.cli`.

The local SFT YAMLs also support the standard Hugging Face compile flags exposed
through `Seq2SeqTrainingArguments`, so you can set `torch_compile: true` beside
`bf16: true` in a config when you want compiled training. The smoke configs keep
it disabled because compile startup dominates one-step runs, while the longer
full-SFT examples enable it.

The vendored LLaMA-Factory tree also exposes Muon as an optimizer option
through the `use_muon: true` finetuning flag in the YAML. In that mode,
matrix-shaped trainable parameters are optimized with Muon while embeddings,
heads, and lower-rank parameters stay on the internal AdamW path. This is best
treated as an alternative to the default AdamW family, not a drop-in speedup
for tiny smoke runs.

`bootstrap_llamafactory_env.sh` creates a local venv, installs the vendored
LLaMA-Factory tree from `OpenSearch-VL/SFT`, and adds the core multimodal
packages required for local image/video SFT.

For DGX Spark, use [docs/dgx-spark-sft.md](../docs/dgx-spark-sft.md)
and the containerized launcher at
[SFT/scripts/run_dgx_spark_sft.sh](scripts/run_dgx_spark_sft.sh).
