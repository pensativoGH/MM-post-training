from __future__ import annotations

import textwrap

import pytest

from verl_post_training.launch import (
    ConfigValidationError,
    DispatchCompatibilityError,
    load_task_config,
    resolve_dispatch,
)


def _write_config(tmp_path, content: str):
    path = tmp_path / "task.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_valid_repo_level_config_loads(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
        task_type: chat_rl
        model_id: qwen3-vl-4b-instruct
        trainer_backend: verl
        dataset_adapter: chat_rl
        input_manifest: data/manifest.jsonl
        output_dir: outputs/run
        launcher:
          kind: torchrun
        resources:
          precision: bf16
        backend_config:
          config_file: RL/configs/local_grpo_vllm.yaml
        """,
    )

    config = load_task_config(config_path)

    assert config.task_type.value == "chat_rl"
    assert config.trainer_backend.value == "verl"
    assert config.runtime_backend is None
    assert config.backend_config == {"config_file": "RL/configs/local_grpo_vllm.yaml"}


def test_unknown_enum_rejected_by_loader(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
        task_type: not_real
        model_id: qwen3-vl-4b-instruct
        trainer_backend: verl
        dataset_adapter: chat_rl
        input_manifest: data/manifest.jsonl
        output_dir: outputs/run
        launcher:
          kind: torchrun
        resources:
          precision: bf16
        """,
    )

    with pytest.raises(ConfigValidationError, match="task_type"):
        load_task_config(config_path)


def test_backend_config_stays_passthrough(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
        task_type: chat_sft
        model_id: qwen3-vl-4b-instruct
        trainer_backend: llamafactory
        dataset_adapter: chat_sft
        input_manifest: data/manifest.jsonl
        output_dir: outputs/run
        launcher:
          kind: python
        resources:
          precision: bf16
        backend_config:
          extra_args:
            - --flag
          config_file: path/to/backend.yaml
        """,
    )

    plan = resolve_dispatch(load_task_config(config_path))

    assert plan.backend_config == {
        "extra_args": ["--flag"],
        "config_file": "path/to/backend.yaml",
    }
    assert not hasattr(plan.config, "config_file")
    assert not hasattr(plan.config, "extra_args")


def test_incompatible_backend_combination_fails_explicitly(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
        task_type: world_model_rollout
        model_id: dreamdojo-world-model-placeholder
        runtime_backend: openai_chat_vllm
        dataset_adapter: dreamdojo
        input_manifest: data/manifest.jsonl
        output_dir: outputs/run
        launcher:
          kind: python
        resources:
          precision: bf16
        """,
    )

    with pytest.raises(
        DispatchCompatibilityError,
        match="model_family=world_model.*runtime_backend=openai_chat_vllm",
    ):
        resolve_dispatch(load_task_config(config_path))
