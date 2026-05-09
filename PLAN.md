# Implementation Plan

## Goal

Build a local-first multimodal post-training repo with a structure similar to
`OpenSearch-VL`, but using:

- `LLaMA-Factory` for SFT
- `VERL` as the RL orchestrator
- `vLLM` as the default local rollout/policy runtime
- optional `SGLang` later behind the same runtime interface

The primary target is image/video SFT and RL on a single machine. Multi-node
and Megatron/Ray are future extensions, not required for the first usable path.

## Repository Shape

- `SFT/`
  - local-first LLaMA-Factory configs
  - dataset registry for image/video conversations
  - training and validation scripts
- `RL/`
  - VERL-oriented configs and launchers
  - shared rollout/runtime settings
  - project-specific workflow code for multimodal tool-use RL
- `runtime/`
  - local OpenAI-compatible serving conventions
  - shared readiness probes and smoke tests
  - backend abstraction for `vLLM` now, `SGLang` later

## Execution Defaults

- SFT default:
  - `llamafactory-cli train ...`
  - single node, local GPUs
  - `torchrun`/HF path first
- RL default:
  - local VERL training
  - rollout model served over OpenAI-compatible multimodal API
  - no Ray or Megatron in the first path
- Runtime default:
  - `vLLM serve`
  - one model selector, one base URL, one readiness contract

## Data Contracts

### SFT

- ShareGPT-style multimodal conversations
- support `images` and `videos`
- preserve `system`, `tools`, `observation` roles
- dataset registry kept explicit in `dataset_info.*.json`

### RL

- prompt/answer/image/video episodes
- room for tool schemas and rollout metadata
- keep reward inputs and judge/runtime dependencies isolated from trainer logic

## Phases

### Phase 1

- scaffold repo shape
- add runtime scripts and local client utilities
- add initial SFT and RL config shells

### Phase 2

- wire in real LLaMA-Factory training configs
- define video-capable data preprocessing path
- validate local multimodal runtime against target models

### Phase 3

- implement local VERL GRPO/RLOO flow
- add shared rollout prompt/tool interfaces
- add reward plumbing and local eval

### Phase 4

- optional SGLang backend
- optional Ray/Megatron path
- larger-scale cluster launchers

## Constraints

- local-first over cluster-first
- stable runtime contract over framework-specific shortcuts
- minimal backend assumptions in project-specific code
