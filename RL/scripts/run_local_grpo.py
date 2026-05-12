#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

_POST_TRAINING_ROOT = Path(__file__).resolve().parents[2] / "post_training"
_POST_TRAINING_SRCS = [
    _POST_TRAINING_ROOT / "shared" / "src",
    _POST_TRAINING_ROOT / "vjepa" / "src",
    _POST_TRAINING_ROOT / "wan" / "src",
    _POST_TRAINING_ROOT / "dreamdojo" / "src",
]
for _src in reversed(_POST_TRAINING_SRCS):
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local VERL GRPO smoke config.")
    parser.add_argument("config", type=Path, help="Path to repo-local RL YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing.")
    return parser.parse_args()


def q(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def build_overrides(config: dict, root_dir: Path) -> list[str]:
    algorithm = config.get("algorithm", {})
    trainer = config["trainer"]
    model = config["model"]
    rollout = config["rollout"]
    data = config["data"]
    reward = config["reward"]
    ray = config.get("ray", {})
    lora_rank = int(model.get("lora_rank", 0) or 0)

    default_output_dir = trainer.get("default_local_dir", "outputs/rl/local_grpo_vllm")
    reward_function = reward.get("custom_reward_function", {})

    overrides = [
        "algorithm.adv_estimator=grpo",
        "trainer.logger=['console']",
        f"trainer.project_name={trainer['project_name']}",
        f"trainer.experiment_name={trainer['experiment_name']}",
        f"trainer.nnodes={trainer.get('nnodes', 1)}",
        f"trainer.n_gpus_per_node={trainer.get('n_gpus_per_node', 1)}",
        f"trainer.total_epochs={trainer['total_epochs']}",
        f"trainer.save_freq={trainer.get('save_freq', -1)}",
        f"trainer.test_freq={trainer.get('test_freq', 1)}",
        f"trainer.val_before_train={q(trainer.get('val_before_train', False))}",
        f"trainer.default_local_dir={root_dir / default_output_dir}",
        f"trainer.device={trainer.get('device', 'cuda')}",
        "actor_rollout_ref.model.use_remove_padding=false",
        f"actor_rollout_ref.model.path={model['path']}",
        f"actor_rollout_ref.model.trust_remote_code={q(model.get('trust_remote_code', True))}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={q(model.get('enable_gradient_checkpointing', True))}",
        f"+actor_rollout_ref.model.override_config.attn_implementation={model.get('attn_implementation', 'sdpa')}",
        f"actor_rollout_ref.actor.use_kl_loss={q(algorithm.get('actor_kl_loss', {}).get('enabled', False))}",
        f"actor_rollout_ref.actor.freeze_vision_tower={q(model.get('freeze_vision_tower', False))}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={q(trainer.get('use_dynamic_bsz', False))}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={trainer.get('ppo_mini_batch_size', 1)}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={trainer.get('ppo_micro_batch_size_per_gpu', 1)}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={trainer.get('ppo_max_token_len_per_gpu', data.get('max_prompt_length', rollout.get('max_prompt_length', 4096)))}",
        f"actor_rollout_ref.actor.optim.lr={trainer.get('lr', 1.0e-6)}",
        f"actor_rollout_ref.actor.grad_clip={trainer.get('grad_clip', 1.0)}",
        f"actor_rollout_ref.actor.ppo_epochs={trainer.get('ppo_epochs', 1)}",
        f"actor_rollout_ref.actor.use_torch_compile={q(trainer.get('use_torch_compile', False))}",
        "critic.enable=false",
        f"data.train_files={root_dir / (data.get('train_file') or data.get('train_path'))}",
        f"data.val_files={root_dir / (data.get('val_file') or data.get('val_path'))}",
        "data.prompt_key=prompt",
        "data.reward_fn_key=data_source",
        f"data.max_prompt_length={data.get('max_prompt_length', rollout.get('max_prompt_length', 4096))}",
        f"data.max_response_length={data.get('max_response_length', rollout.get('max_response_length', 8192))}",
        f"data.train_batch_size={data.get('train_batch_size', 1)}",
        f"data.val_batch_size={data.get('val_batch_size', 1)}",
        f"data.return_raw_chat={q(data.get('return_raw_chat', True))}",
        f"data.return_multi_modal_inputs={q(data.get('return_multi_modal_inputs', True))}",
        f"data.image_key={data.get('image_key', 'images')}",
        f"data.video_key={data.get('video_key', 'videos')}",
        f"data.filter_overlong_prompts={q(data.get('filter_overlong_prompts', False))}",
        f"data.dataloader_num_workers={data.get('dataloader_num_workers', 0)}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout.get('tensor_model_parallel_size', 1)}",
        f"actor_rollout_ref.rollout.data_parallel_size={rollout.get('data_parallel_size', 1)}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout.get('gpu_memory_utilization', 0.2)}",
        f"actor_rollout_ref.rollout.temperature={rollout['temperature']}",
        f"actor_rollout_ref.rollout.top_p={rollout['top_p']}",
        f"actor_rollout_ref.rollout.top_k={rollout['top_k']}",
        f"actor_rollout_ref.rollout.n={rollout.get('n', rollout.get('n_resp_per_prompt', 1))}",
        f"actor_rollout_ref.rollout.prompt_length={data.get('max_prompt_length', rollout.get('max_prompt_length', 4096))}",
        f"actor_rollout_ref.rollout.response_length={data.get('max_response_length', rollout.get('max_response_length', 8192))}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={rollout.get('max_num_batched_tokens', rollout.get('max_model_len', 8192))}",
        f"actor_rollout_ref.rollout.max_num_seqs={rollout.get('max_num_seqs', rollout.get('n_resp_per_prompt', 1))}",
        f"actor_rollout_ref.rollout.max_model_len={rollout.get('max_model_len', data.get('max_prompt_length', rollout.get('max_prompt_length', 4096)) + data.get('max_response_length', rollout.get('max_response_length', 8192)))}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={rollout.get('log_prob_micro_batch_size_per_gpu', 1)}",
        f"actor_rollout_ref.rollout.calculate_log_probs={q(rollout.get('calculate_log_probs', False))}",
        f"actor_rollout_ref.rollout.enforce_eager={q(rollout.get('enforce_eager', True))}",
        f"actor_rollout_ref.rollout.free_cache_engine={q(rollout.get('free_cache_engine', True))}",
        f"actor_rollout_ref.rollout.load_format={rollout.get('load_format', 'hf')}",
        f"actor_rollout_ref.rollout.disable_log_stats={q(rollout.get('disable_log_stats', True))}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={q(rollout.get('enable_chunked_prefill', False))}",
        f"actor_rollout_ref.rollout.enable_prefix_caching={q(rollout.get('enable_prefix_caching', False))}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={rollout.get('val_temperature', 0.0)}",
        f"actor_rollout_ref.rollout.val_kwargs.do_sample={q(False)}",
        f"actor_rollout_ref.rollout.agent.num_workers={rollout.get('agent_num_workers', 1)}",
        f"reward.custom_reward_function.path={root_dir / reward_function.get('path', 'RL/src/verl_post_training_rl/reward_functions.py')}",
        f"reward.custom_reward_function.name={reward_function.get('name', 'compute_score')}",
        f"reward.num_workers={reward.get('num_workers', 1)}",
    ]

    warmup_ratio = trainer.get("warmup_ratio")
    if warmup_ratio is not None:
        overrides.append(f"actor_rollout_ref.actor.optim.lr_warmup_steps_ratio={warmup_ratio}")

    min_lr_ratio = trainer.get("min_lr_ratio")
    if min_lr_ratio is not None:
        overrides.append(f"actor_rollout_ref.actor.optim.min_lr_ratio={min_lr_ratio}")

    num_cycles = trainer.get("num_cycles")
    if num_cycles is not None:
        overrides.append(f"actor_rollout_ref.actor.optim.num_cycles={num_cycles}")

    warmup_style = trainer.get("warmup_style")
    if warmup_style is not None:
        overrides.append(f"actor_rollout_ref.actor.optim.warmup_style={warmup_style}")

    actor_kl_loss = algorithm.get("actor_kl_loss", {})
    if actor_kl_loss.get("coef") is not None:
        overrides.append(f"actor_rollout_ref.actor.kl_loss_coef={actor_kl_loss['coef']}")
    if actor_kl_loss.get("type"):
        overrides.append(f"actor_rollout_ref.actor.kl_loss_type={actor_kl_loss['type']}")

    if algorithm.get("use_kl_in_reward") is not None:
        overrides.append(f"algorithm.use_kl_in_reward={q(algorithm['use_kl_in_reward'])}")
    if algorithm.get("kl_penalty"):
        overrides.append(f"algorithm.kl_penalty={algorithm['kl_penalty']}")

    kl_ctrl = algorithm.get("kl_ctrl", {})
    if kl_ctrl.get("type"):
        overrides.append(f"algorithm.kl_ctrl.type={kl_ctrl['type']}")
    if kl_ctrl.get("kl_coef") is not None:
        overrides.append(f"algorithm.kl_ctrl.kl_coef={kl_ctrl['kl_coef']}")
    if kl_ctrl.get("horizon") is not None:
        overrides.append(f"algorithm.kl_ctrl.horizon={kl_ctrl['horizon']}")
    if kl_ctrl.get("target_kl") is not None:
        overrides.append(f"algorithm.kl_ctrl.target_kl={kl_ctrl['target_kl']}")

    rollout_correction = algorithm.get("rollout_correction", {})
    if rollout_correction.get("rollout_is") is not None:
        overrides.append(f"algorithm.rollout_correction.rollout_is={rollout_correction['rollout_is']}")
    if rollout_correction.get("rollout_is_threshold") is not None:
        overrides.append(
            f"algorithm.rollout_correction.rollout_is_threshold={q(rollout_correction['rollout_is_threshold'])}"
        )
    if rollout_correction.get("rollout_is_batch_normalize") is not None:
        overrides.append(
            "algorithm.rollout_correction.rollout_is_batch_normalize="
            f"{q(rollout_correction['rollout_is_batch_normalize'])}"
        )
    if rollout_correction.get("rollout_rs") is not None:
        overrides.append(f"algorithm.rollout_correction.rollout_rs={q(rollout_correction['rollout_rs'])}")
    if rollout_correction.get("rollout_rs_threshold") is not None:
        overrides.append(
            f"algorithm.rollout_correction.rollout_rs_threshold={q(rollout_correction['rollout_rs_threshold'])}"
        )
    if rollout_correction.get("bypass_mode") is not None:
        overrides.append(f"algorithm.rollout_correction.bypass_mode={q(rollout_correction['bypass_mode'])}")
    if rollout_correction.get("loss_type") is not None:
        overrides.append(f"algorithm.rollout_correction.loss_type={rollout_correction['loss_type']}")

    if lora_rank > 0:
        overrides.extend(
            [
                f"actor_rollout_ref.model.lora_rank={lora_rank}",
                f"actor_rollout_ref.model.lora_alpha={model.get('lora_alpha', 16)}",
                f"actor_rollout_ref.model.target_modules={q(model.get('target_modules', 'all-linear'))}",
            ]
        )
        if model.get("lora_dropout") is not None:
            overrides.append(f"+actor_rollout_ref.model.lora.dropout={model['lora_dropout']}")

    num_cpus = ray.get("num_cpus")
    if num_cpus is not None:
        overrides.append(f"ray_kwargs.ray_init.num_cpus={num_cpus}")

    object_store_memory = ray.get("object_store_memory")
    if object_store_memory is not None:
        overrides.append(f"+ray_kwargs.ray_init.object_store_memory={object_store_memory}")

    fsdp = trainer.get("fsdp", {})
    if "param_offload" in fsdp:
        overrides.append(f"actor_rollout_ref.actor.fsdp_config.param_offload={q(fsdp['param_offload'])}")
    if "optimizer_offload" in fsdp:
        overrides.append(f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={q(fsdp['optimizer_offload'])}")
    if "offload_policy" in fsdp:
        overrides.append(f"actor_rollout_ref.actor.fsdp_config.offload_policy={q(fsdp['offload_policy'])}")

    exclude_modules = model.get("exclude_modules")
    if lora_rank > 0 and exclude_modules:
        overrides.append(f"actor_rollout_ref.model.exclude_modules={q(exclude_modules)}")
    lora_config = model.get("lora", {})
    if lora_rank > 0:
        for key, value in lora_config.items():
            overrides.append(f"+actor_rollout_ref.model.lora.{key}={q(value)}")

    return overrides


def resolve_trainer_dispatch(config: dict, config_path: Path, root_dir: Path):
    from verl_post_training.adapters.trainer import resolve_trainer_adapter
    from verl_post_training.launch.dispatch import resolve_dispatch
    from verl_post_training.launch.load_config import TaskConfig

    model_id = str(config.get("model", {}).get("path") or "").strip()
    if not model_id:
        raise ValueError("RL config must define model.path for trainer dispatch.")

    task_config = TaskConfig.from_mapping(
        {
            "task_type": "chat_rl",
            "model_id": model_id,
            "trainer_backend": "verl",
            "dataset_adapter": "chat_rl",
            "input_manifest": str(config_path),
            "output_dir": str(root_dir / "outputs" / "rl" / "local_grpo"),
            "launcher": {"kind": "python_module", "module": "verl.trainer.main_ppo"},
            "resources": {"precision": "bf16", "devices": config.get("trainer", {}).get("n_gpus_per_node", 1)},
            "backend_config": {"config_file": str(config_path)},
        }
    )
    plan = resolve_dispatch(task_config)
    return plan, resolve_trainer_adapter(plan)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    root_dir = config_path.parents[2]
    config = yaml.safe_load(config_path.read_text())
    plan, adapter = resolve_trainer_dispatch(config, config_path=config_path, root_dir=root_dir)
    print(
        "Resolved trainer dispatch: "
        f"{adapter.adapter_key} handles {plan.task_type.value} for {plan.model_entry.model_family.value}"
    )
    overrides = build_overrides(config, root_dir=root_dir)
    record = adapter.run(
        plan,
        output_dir=root_dir / "outputs" / "rl" / "local_grpo",
        dry_run=args.dry_run,
        overrides=overrides,
    )
    cmd = list(record["argv"])
    print("Resolved VERL command:")
    print(" ".join(cmd))

    if args.dry_run:
        return 0

    env = os.environ.copy()
    src_path = root_dir / "RL/src"
    env["PYTHONPATH"] = str(src_path) if "PYTHONPATH" not in env else f"{src_path}:{env['PYTHONPATH']}"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return subprocess.call(cmd, cwd=root_dir, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
