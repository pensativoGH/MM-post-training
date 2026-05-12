"""Seeded repo-owned model registry."""

from __future__ import annotations

from .schemas import (
    ModelFamily,
    ModelRegistryEntry,
    RuntimeBackend,
    TaskType,
    TrainerBackend,
)


class ModelNotFoundError(LookupError):
    """Raised when a model_id is not present in the repo-owned registry."""


def _build_registry() -> dict[str, ModelRegistryEntry]:
    entries = (
        ModelRegistryEntry(
            model_id="Qwen/Qwen3-VL-8B-Thinking",
            model_family=ModelFamily.VLM_CHAT,
            supported_task_types=(TaskType.CHAT_SFT, TaskType.CHAT_RL),
            trainer_backends=(
                TrainerBackend.LLAMAFACTORY,
                TrainerBackend.VERL,
            ),
            runtime_backends=(RuntimeBackend.OPENAI_CHAT_VLLM,),
            checkpoint_source="Qwen/Qwen3-VL-8B-Thinking",
            checkpoint_format="huggingface",
            required_modalities=("image", "text", "video"),
            dataset_adapter_keys=("chat_sft", "chat_rl"),
            launcher_type="python_module",
            default_precision="bf16",
            distributed_requirements={"min_gpus": 1},
            environment_tags=("local", "gpu", "qwen"),
        ),
        ModelRegistryEntry(
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            model_family=ModelFamily.VLM_CHAT,
            supported_task_types=(TaskType.CHAT_SFT, TaskType.CHAT_RL),
            trainer_backends=(
                TrainerBackend.LLAMAFACTORY,
                TrainerBackend.VERL,
            ),
            runtime_backends=(RuntimeBackend.OPENAI_CHAT_VLLM,),
            checkpoint_source="Qwen/Qwen3-VL-8B-Instruct",
            checkpoint_format="huggingface",
            required_modalities=("image", "text", "video"),
            dataset_adapter_keys=("chat_sft", "chat_rl"),
            launcher_type="python_module",
            default_precision="bf16",
            distributed_requirements={"min_gpus": 1},
            environment_tags=("local", "gpu", "qwen"),
        ),
        ModelRegistryEntry(
            model_id="Qwen/Qwen3-VL-32B-Thinking",
            model_family=ModelFamily.VLM_CHAT,
            supported_task_types=(TaskType.CHAT_SFT, TaskType.CHAT_RL),
            trainer_backends=(
                TrainerBackend.LLAMAFACTORY,
                TrainerBackend.VERL,
            ),
            runtime_backends=(RuntimeBackend.OPENAI_CHAT_VLLM,),
            checkpoint_source="Qwen/Qwen3-VL-32B-Thinking",
            checkpoint_format="huggingface",
            required_modalities=("image", "text", "video"),
            dataset_adapter_keys=("chat_sft", "chat_rl"),
            launcher_type="python_module",
            default_precision="bf16",
            distributed_requirements={"min_gpus": 1},
            environment_tags=("local", "gpu", "qwen"),
        ),
        ModelRegistryEntry(
            model_id="qwen3-vl-4b-instruct",
            model_family=ModelFamily.VLM_CHAT,
            supported_task_types=(TaskType.CHAT_SFT, TaskType.CHAT_RL),
            trainer_backends=(
                TrainerBackend.LLAMAFACTORY,
                TrainerBackend.VERL,
            ),
            runtime_backends=(RuntimeBackend.OPENAI_CHAT_VLLM,),
            checkpoint_source="Qwen/Qwen3-VL-4B-Instruct",
            checkpoint_format="huggingface",
            required_modalities=("image", "text", "video"),
            dataset_adapter_keys=("chat_sft", "chat_rl"),
            launcher_type="python_module",
            default_precision="bf16",
            distributed_requirements={"min_gpus": 1},
            environment_tags=("local", "gpu", "qwen"),
        ),
        ModelRegistryEntry(
            model_id="vjepa2-video-encoder-placeholder",
            model_family=ModelFamily.VIDEO_ENCODER,
            supported_task_types=(
                TaskType.EMBEDDING_INFERENCE,
                TaskType.MASKED_VIDEO_PREDICTION,
            ),
            trainer_backends=(TrainerBackend.VJEPA2_NATIVE,),
            runtime_backends=(RuntimeBackend.VJEPA2_NATIVE,),
            checkpoint_source="facebook/v-jepa-2",
            checkpoint_format="native",
            required_modalities=("video",),
            dataset_adapter_keys=("vjepa2",),
            launcher_type="python_module",
            default_precision="bf16",
            distributed_requirements={},
            environment_tags=("placeholder", "gpu"),
        ),
        ModelRegistryEntry(
            model_id="facebook/vjepa2-base",
            model_family=ModelFamily.VIDEO_ENCODER,
            supported_task_types=(
                TaskType.EMBEDDING_INFERENCE,
                TaskType.MASKED_VIDEO_PREDICTION,
            ),
            trainer_backends=(TrainerBackend.VJEPA2_NATIVE,),
            runtime_backends=(RuntimeBackend.VJEPA2_NATIVE,),
            checkpoint_source="facebook/vjepa2-base",
            checkpoint_format="native",
            required_modalities=("video",),
            dataset_adapter_keys=("vjepa2",),
            launcher_type="python_module",
            default_precision="bf16",
            distributed_requirements={},
            environment_tags=("gpu", "vjepa2"),
        ),
        ModelRegistryEntry(
            model_id="wan-video-generator-placeholder",
            model_family=ModelFamily.VIDEO_GENERATOR,
            supported_task_types=(TaskType.GENERATION_INFERENCE,),
            trainer_backends=(TrainerBackend.WAN_NATIVE,),
            runtime_backends=(RuntimeBackend.WAN_NATIVE,),
            checkpoint_source="wan/wan-placeholder",
            checkpoint_format="native",
            required_modalities=("text",),
            dataset_adapter_keys=("wan",),
            launcher_type="python_module",
            default_precision="bf16",
            distributed_requirements={},
            environment_tags=("placeholder", "gpu"),
        ),
        ModelRegistryEntry(
            model_id="Wan-AI/Wan2.2-T2V-A14B",
            model_family=ModelFamily.VIDEO_GENERATOR,
            supported_task_types=(TaskType.GENERATION_INFERENCE,),
            trainer_backends=(TrainerBackend.WAN_NATIVE,),
            runtime_backends=(RuntimeBackend.WAN_NATIVE,),
            checkpoint_source="Wan-AI/Wan2.2-T2V-A14B",
            checkpoint_format="native",
            required_modalities=("text",),
            dataset_adapter_keys=("wan",),
            launcher_type="python_module",
            default_precision="bf16",
            distributed_requirements={"min_gpus": 1},
            environment_tags=("gpu", "wan22"),
        ),
        ModelRegistryEntry(
            model_id="Wan-AI/Wan2.2-I2V-A14B",
            model_family=ModelFamily.VIDEO_GENERATOR,
            supported_task_types=(TaskType.GENERATION_INFERENCE,),
            trainer_backends=(TrainerBackend.WAN_NATIVE,),
            runtime_backends=(RuntimeBackend.WAN_NATIVE,),
            checkpoint_source="Wan-AI/Wan2.2-I2V-A14B",
            checkpoint_format="native",
            required_modalities=("text", "image"),
            dataset_adapter_keys=("wan",),
            launcher_type="python_module",
            default_precision="bf16",
            distributed_requirements={"min_gpus": 1},
            environment_tags=("gpu", "wan22"),
        ),
        ModelRegistryEntry(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            model_family=ModelFamily.VIDEO_GENERATOR,
            supported_task_types=(TaskType.GENERATION_INFERENCE,),
            trainer_backends=(TrainerBackend.WAN_NATIVE,),
            runtime_backends=(RuntimeBackend.WAN_NATIVE,),
            checkpoint_source="Wan-AI/Wan2.2-TI2V-5B",
            checkpoint_format="native",
            required_modalities=("text", "image", "video"),
            dataset_adapter_keys=("wan",),
            launcher_type="python_module",
            default_precision="bf16",
            distributed_requirements={"min_gpus": 1},
            environment_tags=("gpu", "wan22"),
        ),
        ModelRegistryEntry(
            model_id="dreamdojo-world-model-placeholder",
            model_family=ModelFamily.WORLD_MODEL,
            supported_task_types=(TaskType.WORLD_MODEL_ROLLOUT,),
            trainer_backends=(TrainerBackend.DREAMDOJO,),
            runtime_backends=(RuntimeBackend.DREAMDOJO,),
            checkpoint_source="dreamdojo/world-model-placeholder",
            checkpoint_format="native",
            required_modalities=("action", "observation"),
            dataset_adapter_keys=("dreamdojo",),
            launcher_type="python_module",
            default_precision="bf16",
            distributed_requirements={},
            environment_tags=("placeholder", "gpu"),
        ),
    )
    return {entry.model_id: entry for entry in entries}


MODEL_REGISTRY: dict[str, ModelRegistryEntry] = _build_registry()
REGISTRY = MODEL_REGISTRY


def get_model_entry(model_id: str) -> ModelRegistryEntry:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise ModelNotFoundError(f"Unknown model_id: {model_id}") from exc


def iter_entries() -> tuple[ModelRegistryEntry, ...]:
    return tuple(MODEL_REGISTRY.values())
