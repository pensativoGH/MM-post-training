# Post-Training Pipeline Design Doc

Status: Approved for implementation

## 1. Purpose

This document defines the approved architecture for extending
`verl-post-training` from a Qwen-centric multimodal post-training repo into a
repo-owned orchestration layer that can support multiple model families without
forcing users into upstream repos as the primary workflow.

The approved design keeps the existing strengths of the repo:

- local-first execution
- pipeline-owned data preparation
- LLaMA-Factory for chat SFT
- VERL for chat RL
- OpenAI-compatible runtime for chat-style serving

It also introduces explicit boundaries so new model families can be integrated
without leaking model-specific assumptions across the codebase.

## 2. Scope

### In Scope

- Add repo-level abstractions for model families, task types, trainer
  backends, runtime backends, and dataset adapters.
- Support integration of three new upstream families:
  - V-JEPA2
  - Wan2.2
  - DreamDojo
- Preserve the existing Qwen chat SFT and RL workflow.
- Define concrete file and interface boundaries for implementation.
- Define phased milestones and deferred work.

### Out of Scope

- Rewriting upstream training code into a unified internal framework.
- Forcing all inference paths into OpenAI chat semantics.
- Adding generic serving APIs for all model families in phase 1.
- Solving cross-family checkpoint conversion in the first milestone.
- Renaming top-level directories immediately if that would cause churn.

## 3. Current State

The current repo has three strong but narrow subsystems:

- `SFT/`: LLaMA-Factory-driven multimodal chat SFT
- `RL/`: VERL-driven RL assuming `vLLM` rollout
- `runtime/`: OpenAI-compatible runtime and client probes for chat models

That architecture works for Qwen-VL style chat models, but it overfits the
following assumptions:

- training is either chat SFT or chat RL
- runtime is chat-completions oriented
- rollout uses `vLLM`
- dataset outputs are primarily chat examples or RL prompt-response records

Those assumptions do not hold for:

- V-JEPA2 video encoders
- Wan2.2 video generation models
- DreamDojo world models

## 4. Approved Architecture

### 4.1 Guiding Decision

The repo will gain a repo-owned control plane before deep upstream integration.
New model families will plug into that control plane through explicit adapter
interfaces.

This is the main constraint:

- upstream repos remain execution backends
- this repo owns orchestration, config normalization, dataset adaptation,
  smoke tests, artifact conventions, and integration boundaries

### 4.2 Top-Level Concept Model

The control plane introduces these first-class concepts:

- `model_family`: what kind of model is being integrated
- `task_type`: what operation is being run
- `trainer_backend`: what training implementation is used
- `runtime_backend`: what inference or rollout implementation is used
- `dataset_adapter`: how pipeline data is reshaped for the backend
- `model_registry_entry`: the declared capabilities and launch metadata for a
  specific model/checkpoint family

Approved initial enums:

`model_family`

- `vlm_chat`
- `video_encoder`
- `video_generator`
- `world_model`

`task_type`

- `chat_sft`
- `chat_rl`
- `masked_video_prediction`
- `video_generation`
- `world_model_posttrain`
- `embedding_inference`
- `generation_inference`
- `rollout_inference`

`trainer_backend`

- `llamafactory`
- `verl`
- `vjepa2_native`
- `wan_native`
- `dreamdojo_native`

`runtime_backend`

- `openai_chat_vllm`
- `encoder_native`
- `video_generation_native`
- `world_model_native`

## 5. Ownership Boundaries

The architecture is approved only if ownership is explicit.

### 5.1 Repo-Owned Control Plane

The repo will own the following concerns:

- model capability declarations
- config schema and validation
- dataset adaptation from pipeline-native inputs
- backend dispatch
- output/artifact directory conventions
- smoke tests
- environment bootstrap entrypoints
- wrapper launch scripts

These are the stable interfaces the repo promises to maintain.

### 5.2 Backend-Owned Execution

Backend implementations own the following concerns:

- actual training loops
- actual inference kernels
- distributed launch details beyond what wrappers need to pass through
- backend-specific checkpoint loading logic
- backend-specific environment requirements

For phase 1, upstream repos remain the implementation source for new families.

### 5.3 Existing Directory Ownership

- `SFT/` remains the home for LLaMA-Factory chat SFT until a later repo
  reorganization milestone.
- `RL/` remains the home for VERL chat RL.
- `runtime/` remains the home for OpenAI-compatible chat runtime code.
- `docs/` owns approved architecture and implementation documents.

New cross-family orchestration code should not be embedded directly into
`SFT/`, `RL/`, or `runtime/` unless it is backend-specific to those areas.

## 6. Approved File Layout

The following layout is approved as the implementation target:

```text
docs/
  post-training-pipeline-design-doc-approved.md

post_training/
  README.md
  src/verl_post_training/
    registry/
      model_registry.py
      schemas.py
    adapters/
      dataset/
        base.py
        chat_sft.py
        chat_rl.py
        vjepa2.py
        wan.py
        dreamdojo.py
      trainer/
        base.py
        llamafactory.py
        verl.py
        vjepa2.py
        wan.py
        dreamdojo.py
      runtime/
        base.py
        openai_chat.py
        encoder.py
        video_generation.py
        world_model.py
    launch/
      load_config.py
      dispatch.py
    smoke/
      test_chat_runtime.py
      test_vjepa2_inference.py
      test_wan_generation.py
      test_dreamdojo_rollout.py
    bootstrap/
      third_party.py
  configs/
    models/
    tasks/
    pipelines/
  scripts/
    bootstrap_third_party.sh
    run_task.py

third_party/
  vjepa2/
  Wan2.2/
  DreamDojo/
```

Notes:

- `post_training/` is a new repo-owned orchestration package. The exact package
  name may be adjusted during implementation, but the control-plane code must
  live outside `SFT/`, `RL/`, and `runtime/`.
- `third_party/` is a controlled integration boundary, not a place for local
  business logic.
- Thin wrappers may be added under `SFT/`, `RL/`, or `runtime/` only when they
  are clearly backend-specific.

## 7. Interface Contracts

Implementation must follow explicit interfaces rather than backend-specific
ad hoc scripts.

### 7.1 Model Registry Contract

Each model entry must declare:

- `model_id`
- `model_family`
- `supported_task_types`
- `trainer_backends`
- `runtime_backends`
- `checkpoint_source`
- `checkpoint_format`
- `required_modalities`
- `dataset_adapter_keys`
- `launcher_type`
- `default_precision`
- `distributed_requirements`
- `environment_tags`

Approved Python shape:

```python
@dataclass(frozen=True)
class ModelRegistryEntry:
    model_id: str
    model_family: str
    supported_task_types: tuple[str, ...]
    trainer_backends: tuple[str, ...]
    runtime_backends: tuple[str, ...]
    checkpoint_source: str
    checkpoint_format: str
    required_modalities: tuple[str, ...]
    dataset_adapter_keys: tuple[str, ...]
    launcher_type: str
    default_precision: str
    distributed_requirements: dict[str, Any]
    environment_tags: tuple[str, ...]
```

The model registry is the single source of truth for dispatch decisions. Launch
scripts must not hardcode family-specific assumptions outside the registry.

### 7.2 Dataset Adapter Contract

Dataset adapters convert pipeline-native records into backend-ready assets or
manifests.

Approved interface:

```python
class DatasetAdapter(Protocol):
    adapter_key: str

    def prepare(
        self,
        input_manifest: Path,
        output_dir: Path,
        split: str,
        config: dict[str, Any],
    ) -> Path:
        """Return the backend-consumable manifest path."""
```

Constraints:

- Input comes from pipeline-owned manifests or tables, not backend-specific raw
  directories.
- Output may be JSON, JSONL, CSV, parquet, or a backend-native manifest.
- Media duplication should be avoided when symlinks or references are enough.
- Adapters may materialize backend-specific metadata, but not silently mutate
  source pipeline data.

Approved initial adapters:

- pipeline -> chat SFT dataset
- pipeline -> chat RL dataset
- pipeline -> V-JEPA2 dataset
- pipeline -> Wan conditioning dataset
- pipeline -> DreamDojo trajectory/world-model dataset

### 7.3 Trainer Adapter Contract

Trainer adapters normalize how the repo launches backend training jobs.

Approved interface:

```python
class TrainerAdapter(Protocol):
    backend_name: str

    def train(
        self,
        model_entry: ModelRegistryEntry,
        config_path: Path,
        dataset_manifest: Path,
        output_dir: Path,
    ) -> int:
        """Launch training and return the process exit code."""
```

Responsibilities:

- validate compatibility between task, model, and backend
- construct backend-specific CLI arguments
- set required environment variables
- write resolved launch metadata into `output_dir`

Non-responsibilities:

- rewriting backend training loops
- embedding complex backend internals in the control plane

### 7.4 Runtime Adapter Contract

Runtime adapters expose repo-owned inference surfaces without pretending all
families are chat models.

Approved interface:

```python
class RuntimeAdapter(Protocol):
    backend_name: str

    def run(self, request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        """Execute inference or rollout and return a normalized response."""
```

Approved normalized repo-local surfaces:

- `predict_chat`
- `embed_video`
- `generate_video`
- `rollout_world_model`

The public repo-local API may be implemented as typed wrappers over `run()`,
but those four task shapes are the stable contract for smoke tests and eval
code.

### 7.5 Config Contract

Every repo-level task config must declare:

- `task_type`
- `model_id`
- `trainer_backend` or `runtime_backend`
- `dataset_adapter`
- `input_manifest`
- `output_dir`
- `launcher`
- `resources`
- `backend_config`

Approved YAML shape:

```yaml
task_type: masked_video_prediction
model_id: facebook/vjepa2-base
trainer_backend: vjepa2_native
dataset_adapter: vjepa2_video_dataset
input_manifest: data/pipeline/train_manifest.jsonl
output_dir: outputs/post_training/vjepa2_run_001
launcher:
  kind: torchrun
  num_nodes: 1
  nproc_per_node: 8
resources:
  precision: bf16
  devices: 8
backend_config:
  config_file: third_party/vjepa2/configs/train.yaml
  extra_args: []
```

Backend-specific configuration is permitted only under `backend_config`.
Cross-family launch logic must use the normalized top-level fields.

## 8. External Dependency Strategy

The repo will integrate upstream dependencies behind pinned and inspectable
boundaries.

### 8.1 Approved Policy

- upstream repos live under `third_party/`
- each upstream repo must be pinned to an exact commit or release tag
- bootstrap scripts must record the pinned revision used
- local wrapper code must live outside `third_party/`
- direct edits inside upstream trees are disallowed unless an explicit patch
  management mechanism is introduced later

### 8.2 Approved First-Pass Bootstrap Model

Use repo-owned bootstrap scripts rather than git submodules in phase 1.

Rationale:

- lower friction while integration interfaces are still settling
- easier local iteration
- simpler path for quick proof-of-life smoke tests

Deferred item:

- submodule or vendoring policy can be revisited after the first stable
  integration milestone

### 8.3 External Dependencies by Family

`vlm_chat`

- LLaMA-Factory
- VERL
- `vLLM`

`video_encoder`

- V-JEPA2 upstream training and inference code

`video_generator`

- Wan2.2 upstream generation and possible fine-tuning code

`world_model`

- DreamDojo upstream training and rollout code

## 9. Execution Flows

### 9.1 Chat SFT

1. Load repo-level config.
2. Resolve `model_id` through the registry.
3. Run chat SFT dataset adapter.
4. Dispatch to the LLaMA-Factory trainer adapter.
5. Write outputs to repo-standard locations under `outputs/`.

### 9.2 Chat RL

1. Load repo-level config.
2. Resolve `model_id` through the registry.
3. Prepare RL dataset manifest from pipeline inputs.
4. Start or resolve the OpenAI-compatible runtime backend.
5. Dispatch to the VERL trainer adapter.

### 9.3 V-JEPA2 Training or Inference

1. Load repo-level config.
2. Resolve `model_id` through the registry.
3. Adapt pipeline videos into the V-JEPA2 manifest shape.
4. Dispatch to either the V-JEPA2 trainer adapter or encoder runtime adapter.
5. Store artifacts under repo-standard output paths.

### 9.4 Wan2.2 Inference

1. Load repo-level config.
2. Resolve `model_id` through the registry.
3. Adapt pipeline conditioning inputs into the Wan manifest shape.
4. Dispatch to the Wan runtime adapter.
5. Store generated video artifacts and metadata under repo-standard outputs.

### 9.5 DreamDojo Post-Training or Rollout

1. Load repo-level config.
2. Resolve `model_id` through the registry.
3. Adapt pipeline temporal records into DreamDojo-compatible trajectory data.
4. Dispatch to the DreamDojo trainer or runtime adapter.
5. Store rollout artifacts, traces, and metadata under repo-standard outputs.

## 10. Phase Plan

### Phase 1A: Control Plane

Deliverables:

- model registry
- config loader and validator
- dataset adapter base classes
- trainer adapter base classes
- runtime adapter base classes
- bootstrap scripts for `third_party/`

Exit criteria:

- existing Qwen chat SFT and chat RL can be described through the new control
  plane without regression in current workflows

### Phase 1B: V-JEPA2 Integration

Deliverables:

- pinned V-JEPA2 bootstrap
- V-JEPA2 dataset adapter
- V-JEPA2 trainer adapter
- V-JEPA2 encoder runtime adapter
- smoke test covering one or two pipeline videos

Exit criteria:

- launch training from a repo-level config
- run encoder inference on pipeline videos
- avoid manual execution inside the upstream repo as the normal path

### Phase 1C: Wan2.2 Inference First

Deliverables:

- pinned Wan2.2 bootstrap
- Wan dataset adapter
- Wan runtime adapter
- generation smoke test

Exit criteria:

- download or resolve one supported checkpoint
- generate a short sample from one or two pipeline examples
- write outputs into repo-standard artifact locations

Deferred from this phase:

- Wan fine-tuning support

### Phase 1D: DreamDojo Integration

Deliverables:

- pinned DreamDojo bootstrap
- DreamDojo dataset adapter
- DreamDojo trainer adapter
- DreamDojo runtime adapter
- world-model smoke test

Exit criteria:

- run inference or rollout from repo-level config
- demonstrate pipeline-to-world-model dataset adaptation

Deferred from this phase:

- full DreamDojo pretraining support if environment requirements are heavier
  than the repo should absorb early

## 11. Risks and Assumptions

### 11.1 Active Risks

- The current repo naming still centers `SFT/` and `RL/`, which can confuse
  ownership if cross-family code is added there.
- DreamDojo may have materially heavier environment or hardware assumptions
  than the current local-first workflow.
- Wan2.2 training support may require more invasive backend handling than
  inference support.
- If wrappers bypass the model registry, the repo will drift back into
  model-specific scripts.

### 11.2 Assumptions

- The existing pipeline can expose manifests with enough temporal and modality
  metadata for all target families.
- Phase 1 success does not require unified checkpoint conversion.
- For the first milestone, wrapper launchers are acceptable even if they pass
  through backend-specific flags under `backend_config`.
- Existing Qwen workflows must remain functional while the control plane is
  introduced.

## 12. Deferred Decisions

These items are intentionally deferred and require later milestones:

- whether `SFT/`, `RL/`, and `runtime/` should be renamed under a larger
  `post_training/` top-level reorganization
- whether `third_party/` moves from bootstrap scripts to submodules or vendored
  snapshots
- whether a generic service layer should wrap non-chat runtime backends
- whether Wan2.2 and DreamDojo training become first-class in the same milestone
  as inference
- whether backend environments are managed in a single lockfile or per-family
  bootstrap environment

## 13. Implementation Constraints

Implementation must respect these constraints:

- do not add new model-family logic by copy-pasting per-model scripts as the
  long-term interface
- do not make `vLLM` or OpenAI chat semantics mandatory for non-chat families
- do not store repo-specific wrapper code inside upstream trees
- do not bypass dataset adapters by requiring users to handcraft backend-native
  datasets manually
- do not regress the current Qwen chat SFT and RL workflow while phase 1A is
  landing

## 14. Acceptance Criteria for This Design

This design is considered implementation-ready when the first implementation
milestone can answer all of the following without further architectural debate:

- Where does model capability metadata live?
- How does a task config declare backend choice?
- How is pipeline data adapted for each family?
- What code remains repo-owned versus upstream-owned?
- How do smoke tests call inference without forcing chat semantics?
- What is phase 1 versus deferred work?

This document answers those questions and is therefore approved as the
architecture baseline for implementation.
