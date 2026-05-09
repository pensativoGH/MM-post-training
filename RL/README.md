# RL

Local-first reinforcement learning scaffold for multimodal post-training.

This subtree keeps the same *role* as `OpenSearch-VL/RL/`, but the first
execution path is intentionally simpler:

- `VERL` as the trainer/orchestrator
- OpenAI-compatible local multimodal runtime for rollout
- `vLLM` as the default backend
- no default dependency on Ray or Megatron

## First local target

- single-machine GRPO or RLOO
- image/video prompts
- rollout policy served at a local OpenAI-compatible endpoint
- reward and tool-use logic kept separate from serving

## Boundary

Framework-heavy internals should stay upstream when possible. This repo should
mainly contain:

- local configs
- project-specific workflow logic
- launch scripts
- shared multimodal task and reward contracts
