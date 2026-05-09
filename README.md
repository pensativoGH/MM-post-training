# verl-post-training

Local-first multimodal post-training scaffold for image/video SFT and RL.

This repository mirrors the high-level structure of `OpenSearch-VL` while
changing the default execution model:

- `SFT/`: LLaMA-Factory-driven supervised fine-tuning
- `RL/`: VERL-oriented reinforcement-learning workflows
- `runtime/`: shared OpenAI-compatible multimodal serving/client/probe layer

The default local path is:

1. Serve a multimodal base or policy model with `vLLM`
2. Fine-tune with LLaMA-Factory on image/video conversational data
3. Train RL with VERL against the same runtime contract

For DGX Spark specifically, the SFT path can also run inside NVIDIA's PyTorch
container via [SFT/scripts/run_dgx_spark_sft.sh](SFT/scripts/run_dgx_spark_sft.sh).

`SGLang` is intentionally not the default local backend. The runtime surface is
kept backend-agnostic so it can be added later behind the same interface.

## Layout

```text
verl-post-training/
├── PLAN.md
├── SFT/
│   ├── README.md
│   ├── data/
│   │   ├── dataset_info.json
│   │   └── video_agent_sft_demo/
│   ├── examples/
│   │   └── local/
│   │       └── qwen3_vl_video_sft_8b.yaml
│   └── scripts/
│       ├── bootstrap_llamafactory_env.sh
│       ├── preview_dataset.py
│       ├── run_dgx_spark_sft.sh
│       ├── run_local_sft.sh
│       └── validate_dataset.py
├── docs/
│   └── dgx-spark-sft.md
├── RL/
│   ├── README.md
│   ├── configs/
│   │   └── local_grpo_vllm.yaml
│   ├── run/
│   │   └── local-grpo-vllm.sh
│   └── src/
│       └── verl_post_training_rl/
└── runtime/
    ├── README.md
    ├── scripts/
    │   ├── runtime_common.sh
    │   ├── start_qwen_vllm_server.sh
    │   ├── check_qwen_vllm_ready.py
    │   └── smoke_qwen_openai_mm.py
    └── src/
        └── verl_post_training_runtime/
            ├── __init__.py
            └── local_runtime.py
```

## Design Rules

- Prefer single-machine execution first.
- Keep model-serving access OpenAI-compatible.
- Keep dataset schemas explicit for both images and videos.
- Keep cluster-only concerns optional, not baked into the default path.
- Reuse the same runtime contract for local eval, SFT validation, and RL rollout.

## Immediate Next Work

- Add a first VERL local GRPO training entrypoint under `RL/`.
- Add shared prompt/tool schemas for image/video tool-use data.
- Add evaluation scripts that hit the local runtime directly.
