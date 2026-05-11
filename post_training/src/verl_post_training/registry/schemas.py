"""Repo-owned schema types for the model registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ModelFamily(StrEnum):
    VLM_CHAT = "vlm_chat"
    VIDEO_ENCODER = "video_encoder"
    VIDEO_GENERATOR = "video_generator"
    WORLD_MODEL = "world_model"


class TaskType(StrEnum):
    CHAT_SFT = "chat_sft"
    CHAT_RL = "chat_rl"
    EMBEDDING_INFERENCE = "embedding_inference"
    MASKED_VIDEO_PREDICTION = "masked_video_prediction"
    GENERATION_INFERENCE = "generation_inference"
    WORLD_MODEL_ROLLOUT = "world_model_rollout"


class TrainerBackend(StrEnum):
    LLAMAFACTORY = "llamafactory"
    VERL = "verl"
    VJEPA2_NATIVE = "vjepa2_native"
    WAN_NATIVE = "wan_native"
    DREAMDOJO = "dreamdojo"


class RuntimeBackend(StrEnum):
    OPENAI_CHAT_VLLM = "openai_chat_vllm"
    VJEPA2_NATIVE = "vjepa2_native"
    WAN_NATIVE = "wan_native"
    DREAMDOJO = "dreamdojo"


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_id: str
    model_family: ModelFamily
    supported_task_types: tuple[TaskType, ...]
    trainer_backends: tuple[TrainerBackend, ...]
    runtime_backends: tuple[RuntimeBackend, ...]
    checkpoint_source: str
    checkpoint_format: str
    required_modalities: tuple[str, ...]
    dataset_adapter_keys: tuple[str, ...]
    launcher_type: str
    default_precision: str
    distributed_requirements: dict[str, object] = field(default_factory=dict)
    environment_tags: tuple[str, ...] = ()

