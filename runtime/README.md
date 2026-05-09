# runtime

Shared local runtime contract for SFT validation, RL rollout, and evaluation.

## Default backend

`vLLM` is the default local backend.

## Contract

- OpenAI-compatible `/v1/chat/completions`
- multimodal prompt support
- shared readiness probe
- shared smoke test

This keeps SFT, RL, and eval code decoupled from the serving backend.

## Near-term backend policy

- `vLLM`: default and supported
- `SGLang`: optional future backend behind the same interface
