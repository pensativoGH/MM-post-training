# Post-Training Pipeline Implementation Plan

Status: Approved plan draft for implementation

## 1. Purpose

This document turns the approved design in
`docs/post-training-pipeline-design-doc-approved.md` into an implementation
sequence that can be executed as a series of small, mergeable branches.

The plan preserves the current Qwen-centric chat workflow while introducing the
repo-owned control plane needed to support additional model families:

- V-JEPA2
- Wan2.2
- DreamDojo

## 2. Planning Constraints

All milestones in this plan must satisfy the following:

- each milestone is small enough to land on its own branch without depending on
  a large multi-week stack
- each milestone preserves or improves the current Qwen SFT and RL workflow
- repo-owned orchestration code lives outside backend-specific upstream trees
- new model-family support routes through the registry, adapter, and dispatch
  contracts from the approved design
- acceptance criteria are specific enough to support deterministic unit or
  smoke tests

## 3. Merge Strategy

Recommended landing order:

1. Milestones M1 through M4 establish the shared control plane.
2. Milestones M5 through M7 add one family integration at a time.
3. Milestone M8 extends training coverage only where earlier runtime and
   adapter work already proved the interface.
4. Milestone M9 is explicitly human-blocked and must not be auto-landed without
   human approval.

A milestone is considered mergeable only if:

- its acceptance criteria are satisfied
- its tests can run without unpublished local state
- it does not require a sibling milestone to merge in the same branch

## 4. Milestones

### M1. Add Repo-Owned Registry and Core Schemas

Objective:
Create the smallest control-plane surface that can describe current chat models
and future non-chat families without changing training behavior yet.

Scope:

- add a repo-owned Python package for shared orchestration code under
  `post_training/src/verl_post_training/`
- add registry and schema modules, including the approved
  `ModelRegistryEntry`-style contract
- define the approved enums for `model_family`, `task_type`,
  `trainer_backend`, and `runtime_backend`
- seed the registry with at least one working `vlm_chat` entry representing the
  current Qwen path
- add placeholder registry entries or fixtures covering `video_encoder`,
  `video_generator`, and `world_model`

Expected file touch points:

- `post_training/src/verl_post_training/registry/model_registry.py`
- `post_training/src/verl_post_training/registry/schemas.py`
- `post_training/README.md`
- new tests under `tests/`

Dependencies:

- none

Human-blocked:

- no

Acceptance criteria:

- importing the registry from a clean checkout returns at least four model
  families: `vlm_chat`, `video_encoder`, `video_generator`, `world_model`
- the registry contains at least one resolvable Qwen chat model entry with
  non-empty values for `supported_task_types`, `trainer_backends`,
  `runtime_backends`, `dataset_adapter_keys`, and `launcher_type`
- requesting an unknown `model_id` raises a typed error or returns a clearly
  named failure value; it must not silently fall back to Qwen defaults
- the schema layer exposes normalized enum values that other milestones can
  import without reaching into backend-specific code

Tests Claude should be able to write:

- `tests/test_model_registry.py`
- `tests/test_registry_schema_enums.py`
- `tests/test_registry_lookup_errors.py`

### M2. Add Repo-Level Config Loader and Dispatch Entry Point

Objective:
Introduce a normalized task config and dispatcher that selects launch behavior
from the registry instead of script-local assumptions.

Scope:

- implement the approved top-level config contract
- add a config loader and validator
- add a dispatcher entry point that resolves a config into registry-backed
  backend selection
- keep backend-specific arguments isolated under `backend_config`
- do not replace existing user-facing scripts yet; this milestone only adds the
  shared entry point

Expected file touch points:

- `post_training/src/verl_post_training/launch/load_config.py`
- `post_training/src/verl_post_training/launch/dispatch.py`
- `post_training/configs/`
- `post_training/scripts/run_task.py`
- new tests under `tests/`

Dependencies:

- M1

Human-blocked:

- no

Acceptance criteria:

- a valid YAML config containing `task_type`, `model_id`, `dataset_adapter`,
  `input_manifest`, `output_dir`, `launcher`, `resources`, and either
  `trainer_backend` or `runtime_backend` loads into a validated in-memory
  structure
- top-level config validation rejects unknown enum values before any backend
  code is invoked
- `backend_config` keys are preserved and passed through without being promoted
  into top-level dispatch logic
- dispatching an unsupported combination such as `model_family=world_model`
  with `runtime_backend=openai_chat_vllm` fails with an explicit compatibility
  error

Tests Claude should be able to write:

- `tests/test_config_schema_contract.py`
- `tests/test_dispatch_compatibility_errors.py`
- `tests/test_backend_config_passthrough.py`

### M3. Route Existing Chat Runtime Through the Control Plane

Objective:
Prove backward compatibility by making the existing local runtime resolve
through the new registry and dispatch layer.

Scope:

- update the local runtime selector to use registry-backed runtime resolution
- preserve current Qwen `vLLM` startup and readiness behavior
- keep the public runtime scripts stable unless a small argument addition is
  necessary
- add a compatibility shim if needed so current scripts continue to work while
  the new control plane lands

Expected file touch points:

- `runtime/src/verl_post_training_runtime/local_runtime.py`
- `runtime/src/verl_post_training_runtime/__init__.py`
- `runtime/scripts/start_qwen_vllm_server.sh`
- `runtime/scripts/check_qwen_vllm_ready.py`
- shared repo-owned modules added in M1 and M2

Dependencies:

- M1
- M2

Human-blocked:

- no

Acceptance criteria:

- when given the current supported Qwen selector or explicit Qwen model id, the
  runtime resolves `openai_chat_vllm` through the registry instead of
  hard-coded Qwen-only branches
- the readiness check still succeeds against a healthy Qwen `vLLM` server using
  the existing smoke path
- requesting a non-chat family through the chat runtime entry point fails
  before server startup begins
- existing chat runtime behavior remains unchanged for supported selectors:
  base URL format, health check contract, and multimodal smoke invocation all
  remain compatible

Tests Claude should be able to write:

- `tests/test_local_runtime_backcompat.py`
- `tests/test_runtime_registry_resolution.py`
- `tests/test_non_chat_runtime_rejection.py`

### M4. Extract Reusable Dataset Adapters for Existing SFT and RL Flows

Objective:
Move current pipeline-to-dataset conversion logic behind reusable adapter
interfaces before adding new model families.

Scope:

- add the approved `DatasetAdapter` protocol or equivalent base contract
- factor current chat SFT preparation logic into a reusable adapter
- factor current chat RL preparation logic into a reusable adapter
- preserve current data outputs for RoboVQA and procedural smoke paths
- add adapter registration keyed by the names used in repo-level configs
- add stub adapter definitions for V-JEPA2, Wan2.2, and DreamDojo so later
  milestones can register against stable keys

Expected file touch points:

- `post_training/src/verl_post_training/adapters/dataset/base.py`
- `post_training/src/verl_post_training/adapters/dataset/chat_sft.py`
- `post_training/src/verl_post_training/adapters/dataset/chat_rl.py`
- `SFT/scripts/prepare_robovqa_pipeline_sft.py`
- `RL/scripts/prepare_rl_dataset.py`
- `RL/scripts/prepare_robovqa_pipeline_dataset.py`
- `SFT/data/dataset_info.json`

Dependencies:

- M1
- M2

Human-blocked:

- no

Acceptance criteria:

- invoking the chat SFT adapter on an existing pipeline manifest writes a
  backend-consumable dataset manifest without mutating the source manifest
- invoking the chat RL adapter on an existing pipeline manifest writes a
  backend-consumable RL manifest without mutating the source manifest
- existing RoboVQA or procedural smoke manifests still prepare successfully
  through the refactored entry points
- the adapter registry exposes stable keys for `chat_sft`, `chat_rl`,
  `vjepa2`, `wan`, and `dreamdojo`

Tests Claude should be able to write:

- `tests/test_sft_pipeline_adapter.py`
- `tests/test_rl_pipeline_adapter.py`
- `tests/test_adapter_registry_keys.py`

### M5. Add Pinned `third_party/` Bootstrap and Discovery Metadata

Objective:
Create the external dependency boundary before integrating new upstream model
 families.

Scope:

- add repo-owned bootstrap logic for V-JEPA2, Wan2.2, and DreamDojo
- pin each upstream dependency to an explicit commit or release tag
- record the pinned revision in repo-owned metadata
- add discovery helpers so wrappers can locate upstream checkouts without
  hard-coded ad hoc paths
- keep all repo-specific wrapper logic outside `third_party/`

Expected file touch points:

- `post_training/src/verl_post_training/bootstrap/third_party.py`
- `post_training/scripts/bootstrap_third_party.sh`
- `third_party/` metadata files
- new tests under `tests/`

Dependencies:

- M2

Human-blocked:

- no

Acceptance criteria:

- repo-owned metadata lists pinned revisions for V-JEPA2, Wan2.2, and
  DreamDojo
- the bootstrap path can report whether each upstream checkout is absent,
  present at the pinned revision, or present at a mismatched revision
- wrapper code can discover each upstream root from metadata rather than from
  hard-coded absolute or relative paths in user-facing scripts
- the repo contains no business-logic Python modules inside `third_party/`

Tests Claude should be able to write:

- `tests/test_third_party_manifest.py`
- `tests/test_third_party_revision_status.py`
- `tests/test_wrapper_discovery.py`

### M6. Add V-JEPA2 Inference-First Integration

Objective:
Land the first non-chat family through the repo-owned adapter and runtime
contracts.

Scope:

- implement a V-JEPA2 dataset adapter
- implement an encoder runtime adapter for `embed_video`
- add repo-local wrapper code that calls the pinned V-JEPA2 checkout
- add a small smoke path covering one or two pipeline video examples
- write outputs to repo-standard artifact locations

Expected file touch points:

- `post_training/src/verl_post_training/adapters/dataset/vjepa2.py`
- `post_training/src/verl_post_training/adapters/runtime/encoder.py`
- repo-local wrapper modules under `post_training/src/verl_post_training/`
- smoke tests under `post_training/src/verl_post_training/smoke/` or `tests/`

Dependencies:

- M4
- M5

Human-blocked:

- no

Acceptance criteria:

- given a pipeline manifest with video inputs, the V-JEPA2 adapter writes a
  backend-ready manifest or asset reference set without duplicating media when
  references are sufficient
- a repo-level config with `task_type=embedding_inference` and a V-JEPA2 model
  id dispatches to the encoder runtime adapter
- the wrapper returns normalized output containing enough metadata for a smoke
  test to assert success, including model id, output location, and per-example
  result status
- a smoke run against one or two examples completes without requiring the user
  to manually change into the upstream V-JEPA2 repo

Tests Claude should be able to write:

- `tests/test_vjepa2_adapter.py`
- `tests/test_vjepa2_dispatch.py`
- `tests/test_vjepa2_smoke_contract.py`

### M7. Add Wan2.2 Inference-First Integration

Objective:
Add the first generation-oriented family while keeping the control-plane
contract non-chat-specific.

Scope:

- implement a Wan conditioning dataset adapter
- implement a `generate_video` runtime adapter
- add repo-local wrapper code for checkpoint resolution and generation
- add artifact-writing conventions for generated videos and metadata
- add a smoke path covering one or two pipeline examples

Expected file touch points:

- `post_training/src/verl_post_training/adapters/dataset/wan.py`
- `post_training/src/verl_post_training/adapters/runtime/video_generation.py`
- repo-local wrapper modules under `post_training/src/verl_post_training/`
- smoke tests under `post_training/src/verl_post_training/smoke/` or `tests/`

Dependencies:

- M4
- M5

Human-blocked:

- no

Acceptance criteria:

- given a pipeline manifest with supported conditioning inputs, the Wan adapter
  writes a backend-ready manifest describing prompt, media references, and
  output target locations
- a repo-level config with `task_type=generation_inference` and a Wan model id
  dispatches to the video generation runtime adapter
- the wrapper writes generated artifacts and a machine-readable metadata file
  under the configured `output_dir`
- a smoke run against one or two examples completes without requiring the user
  to operate inside the upstream Wan2.2 repo

Tests Claude should be able to write:

- `tests/test_wan_adapter.py`
- `tests/test_wan_dispatch.py`
- `tests/test_wan_artifact_contract.py`

### M8. Add DreamDojo Integration and Capability-Gated Training Dispatch

Objective:
Integrate DreamDojo through the same repo-owned boundaries, then extend
training dispatch only for workflows the repo can describe and validate.

Scope:

- implement a DreamDojo dataset adapter for trajectory or world-model inputs
- implement a DreamDojo runtime adapter for `rollout_world_model` or equivalent
  normalized inference
- add repo-local wrapper code for DreamDojo invocation
- add trainer adapter base implementations for existing chat backends and any
  DreamDojo training path that is concretely supportable
- extend existing SFT and RL entrypoints so they can dispatch through the
  control plane
- add capability reporting so unsupported training combinations fail early

Expected file touch points:

- `post_training/src/verl_post_training/adapters/dataset/dreamdojo.py`
- `post_training/src/verl_post_training/adapters/runtime/world_model.py`
- `post_training/src/verl_post_training/adapters/trainer/base.py`
- `post_training/src/verl_post_training/adapters/trainer/llamafactory.py`
- `post_training/src/verl_post_training/adapters/trainer/verl.py`
- `post_training/src/verl_post_training/adapters/trainer/dreamdojo.py`
- `SFT/scripts/run_local_sft.sh`
- `RL/scripts/run_local_grpo.py`

Dependencies:

- M3
- M4
- M5

Human-blocked:

- partially

Blocked conditions:

- if DreamDojo cannot run within the repo's local-first environment envelope,
  only the adapter, config, and capability-reporting portions of this milestone
  should land; actual runnable DreamDojo smoke coverage must move to M9

Acceptance criteria:

- given a pipeline manifest with supported temporal records, the DreamDojo
  adapter writes a backend-ready manifest or trajectory bundle without mutating
  source records
- a repo-level config with a DreamDojo model id dispatches to a normalized
  world-model runtime adapter rather than a chat runtime path
- `SFT/scripts/run_local_sft.sh` and `RL/scripts/run_local_grpo.py` can resolve
  their supported Qwen workflows through the trainer-dispatch layer without
  changing required user inputs
- unsupported training combinations fail before launch with a capability error
  that names the requested model family, task type, and backend

Tests Claude should be able to write:

- `tests/test_dreamdojo_adapter.py`
- `tests/test_dreamdojo_runtime_dispatch.py`
- `tests/test_training_dispatch.py`
- `tests/test_capability_error_messages.py`

### M9. Human Approval Gate for Environment-Heavy Scope

Objective:
Hold the environment and support decisions that should not be auto-approved by
implementation alone.

Scope:

- review whether DreamDojo runtime or training support fits the local-first
  contract
- review whether Wan2.2 training support should be added beyond inference
- review whether the bootstrap approach should remain script-based or move
  toward submodules or vendored snapshots
- review whether top-level repo reorganization should follow once the control
  plane is stable

Dependencies:

- M8

Human-blocked:

- yes

Acceptance criteria:

- a human reviewer records a decision for DreamDojo runnable support in the
  repo
- a human reviewer records a decision for Wan2.2 training scope
- a human reviewer records a decision for the long-term `third_party/`
  maintenance strategy

Tests Claude should be able to write:

- none; this is a decision gate, not an automated code milestone

## 5. Dependency Summary

| Milestone | Depends On | Can Land Independently? | Human-Blocked |
| --- | --- | --- | --- |
| M1 | none | yes | no |
| M2 | M1 | yes | no |
| M3 | M1, M2 | yes | no |
| M4 | M1, M2 | yes | no |
| M5 | M2 | yes | no |
| M6 | M4, M5 | yes | no |
| M7 | M4, M5 | yes | no |
| M8 | M3, M4, M5 | yes, with partial scope if DreamDojo runtime is blocked | partially |
| M9 | M8 | no | yes |

## 6. Milestones That Must Explicitly Preserve Backward Compatibility

The following milestones modify existing user entrypoints and must include
backward-compatibility verification:

- M3 for `runtime/` local runtime behavior
- M4 for SFT and RL dataset preparation scripts
- M8 for `SFT/scripts/run_local_sft.sh` and `RL/scripts/run_local_grpo.py`

Minimum backward-compatibility checks:

- existing Qwen smoke configs still resolve
- current script arguments still work or fail with a clearly documented error
- output paths used by current local workflows do not silently change

## 7. Known Risks

- The repo currently contains active in-flight changes outside this plan. Each
  milestone branch should limit itself to the files listed in scope to avoid
  accidental coupling.
- DreamDojo may exceed the local-first hardware or environment assumptions
  approved in the design doc. That is why M8 is only partially auto-executable
  and M9 is explicitly human-blocked.
- If M3 or M8 bypass the registry for convenience, the implementation will
  regress back to model-specific script sprawl. Reviewer approval should reject
  any milestone that reintroduces that pattern.

## 8. Definition of Done for the Plan

This implementation plan is ready for execution if a reviewer can hand any
single milestone above to Claude and expect Claude to:

- identify the intended files to change
- implement the milestone without bundling neighboring milestones
- write or update tests directly from the acceptance criteria
- know when to stop and request human input
