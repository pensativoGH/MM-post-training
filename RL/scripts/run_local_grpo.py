#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


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
    trainer = config["trainer"]
    model = config["model"]
    rollout = config["rollout"]
    data = config["data"]
    reward = config["reward"]
    ray = config.get("ray", {})
    lora_rank = int(model.get("lora_rank", 0) or 0)

    overrides = [
        "algorithm.adv_estimator=grpo",
        "trainer.logger=['console']",
        f"trainer.project_name={trainer['project_name']}",
        f"trainer.experiment_name={trainer['experiment_name']}",
        f"trainer.nnodes={trainer['nnodes']}",
        f"trainer.n_gpus_per_node={trainer['n_gpus_per_node']}",
        f"trainer.total_epochs={trainer['total_epochs']}",
        f"trainer.save_freq={trainer['save_freq']}",
        f"trainer.test_freq={trainer['test_freq']}",
        f"trainer.val_before_train={q(trainer['val_before_train'])}",
        f"trainer.default_local_dir={root_dir / trainer['default_local_dir']}",
        f"trainer.device={trainer.get('device', 'cuda')}",
        "actor_rollout_ref.model.use_remove_padding=false",
        f"actor_rollout_ref.model.path={model['path']}",
        f"actor_rollout_ref.model.trust_remote_code={q(model.get('trust_remote_code', True))}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={q(model.get('enable_gradient_checkpointing', True))}",
        f"+actor_rollout_ref.model.override_config.attn_implementation={model.get('attn_implementation', 'sdpa')}",
        "actor_rollout_ref.actor.use_kl_loss=false",
        f"actor_rollout_ref.actor.freeze_vision_tower={q(model.get('freeze_vision_tower', False))}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={q(trainer.get('use_dynamic_bsz', False))}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={trainer['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={trainer['ppo_micro_batch_size_per_gpu']}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={trainer['ppo_max_token_len_per_gpu']}",
        f"actor_rollout_ref.actor.optim.lr={trainer['lr']}",
        f"actor_rollout_ref.actor.use_torch_compile={q(trainer.get('use_torch_compile', False))}",
        "critic.enable=false",
        f"data.train_files={root_dir / data['train_file']}",
        f"data.val_files={root_dir / data['val_file']}",
        "data.prompt_key=prompt",
        "data.reward_fn_key=data_source",
        f"data.max_prompt_length={data['max_prompt_length']}",
        f"data.max_response_length={data['max_response_length']}",
        f"data.train_batch_size={data['train_batch_size']}",
        f"data.val_batch_size={data['val_batch_size']}",
        f"data.return_raw_chat={q(data.get('return_raw_chat', True))}",
        f"data.return_multi_modal_inputs={q(data.get('return_multi_modal_inputs', True))}",
        f"data.image_key={data.get('image_key', 'images')}",
        f"data.video_key={data.get('video_key', 'videos')}",
        f"data.filter_overlong_prompts={q(data.get('filter_overlong_prompts', False))}",
        f"data.dataloader_num_workers={data.get('dataloader_num_workers', 0)}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout['tensor_model_parallel_size']}",
        f"actor_rollout_ref.rollout.data_parallel_size={rollout['data_parallel_size']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout['gpu_memory_utilization']}",
        f"actor_rollout_ref.rollout.temperature={rollout['temperature']}",
        f"actor_rollout_ref.rollout.top_p={rollout['top_p']}",
        f"actor_rollout_ref.rollout.top_k={rollout['top_k']}",
        f"actor_rollout_ref.rollout.n={rollout['n']}",
        f"actor_rollout_ref.rollout.prompt_length={data['max_prompt_length']}",
        f"actor_rollout_ref.rollout.response_length={data['max_response_length']}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={rollout['max_num_batched_tokens']}",
        f"actor_rollout_ref.rollout.max_num_seqs={rollout['max_num_seqs']}",
        f"actor_rollout_ref.rollout.max_model_len={rollout['max_model_len']}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={rollout['log_prob_micro_batch_size_per_gpu']}",
        f"actor_rollout_ref.rollout.enforce_eager={q(rollout['enforce_eager'])}",
        f"actor_rollout_ref.rollout.free_cache_engine={q(rollout['free_cache_engine'])}",
        f"actor_rollout_ref.rollout.load_format={rollout['load_format']}",
        f"actor_rollout_ref.rollout.disable_log_stats={q(rollout['disable_log_stats'])}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={q(rollout['enable_chunked_prefill'])}",
        f"actor_rollout_ref.rollout.enable_prefix_caching={q(rollout['enable_prefix_caching'])}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={rollout['val_temperature']}",
        f"actor_rollout_ref.rollout.val_kwargs.do_sample={q(False)}",
        f"actor_rollout_ref.rollout.agent.num_workers={rollout.get('agent_num_workers', 1)}",
        f"reward.custom_reward_function.path={root_dir / reward['custom_reward_function']['path']}",
        f"reward.custom_reward_function.name={reward['custom_reward_function']['name']}",
        f"reward.num_workers={reward.get('num_workers', 1)}",
    ]

    if lora_rank > 0:
        overrides.extend(
            [
                f"actor_rollout_ref.model.lora_rank={lora_rank}",
                f"actor_rollout_ref.model.lora_alpha={model.get('lora_alpha', 16)}",
                f"actor_rollout_ref.model.target_modules={q(model.get('target_modules', 'all-linear'))}",
            ]
        )

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


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    root_dir = config_path.parents[2]
    config = yaml.safe_load(config_path.read_text())
    overrides = build_overrides(config, root_dir=root_dir)

    cmd = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
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
