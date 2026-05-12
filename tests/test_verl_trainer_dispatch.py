"""M8 acceptance: dispatch resolves Qwen chat RL through the VERL trainer
adapter.

This file pins the explicit trainer-dispatch criterion (quoted from the
approved plan):

    explicit trainer-dispatch tests verify that ... the VERL adapter
    handles ``task_type=chat_rl`` for the seeded Qwen registry entry

Concretely, these tests pin:

* a YAML config with ``task_type=chat_rl``, ``trainer_backend=verl``,
  ``dataset_adapter=chat_rl``, and the seeded Qwen model id loads via
  the M2 loader and dispatches through
  ``verl_post_training.launch.dispatch.resolve_dispatch`` without raising
* the resolved dispatch plan advertises ``trainer_backend=verl`` and
  ``task_type=chat_rl``
* the trainer-adapter selector returns an adapter defined under
  ``verl_post_training.adapters.trainer.verl`` (not the V-JEPA2 or
  LLaMA-Factory trainer adapter)
* a chat_sft config does *not* resolve to the VERL trainer adapter — the
  selector must distinguish RL from SFT
* the VERL trainer adapter rejects an incompatible plan (chat_sft task
  type, or a non-Qwen / non-vlm_chat model_id) directly, so the
  rejection cannot be bypassed by a caller that builds a plan by hand

Tests must run without GPUs or upstream VERL. Dispatch and selector
wiring are the only concern; the adapter's inner launch path is not
exercised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


QWEN_MODEL_ID = "qwen3-vl-4b-instruct"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dispatch_module():
    import verl_post_training.launch.dispatch as module

    return module


@pytest.fixture(scope="module")
def load_config_module():
    import verl_post_training.launch.load_config as module

    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qwen_rl_config(
    *,
    model_id: str = QWEN_MODEL_ID,
    trainer_backend: str = "verl",
) -> dict[str, Any]:
    return {
        "task_type": "chat_rl",
        "model_id": model_id,
        "trainer_backend": trainer_backend,
        "dataset_adapter": "chat_rl",
        "input_manifest": "data/pipeline/rl_manifest.jsonl",
        "output_dir": "outputs/post_training/qwen_rl_run_001",
        "launcher": {
            "kind": "python_module",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }


def _qwen_sft_config() -> dict[str, Any]:
    return {
        "task_type": "chat_sft",
        "model_id": QWEN_MODEL_ID,
        "trainer_backend": "llamafactory",
        "dataset_adapter": "chat_sft",
        "input_manifest": "data/pipeline/train_manifest.jsonl",
        "output_dir": "outputs/post_training/qwen_sft_run_001",
        "launcher": {
            "kind": "torchrun",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }


def _resolve_loader(load_config_module):
    for attr in ("load_config", "load_task_config", "load", "from_yaml"):
        fn = getattr(load_config_module, attr, None)
        if callable(fn):
            return fn
    pytest.fail(
        "verl_post_training.launch.load_config must expose a callable named "
        "one of: load_config, load_task_config, load, from_yaml."
    )


def _resolve_dispatcher(dispatch_module):
    for attr in ("resolve_dispatch", "dispatch", "resolve", "plan", "build_plan"):
        fn = getattr(dispatch_module, attr, None)
        if callable(fn):
            return fn
    pytest.fail(
        "verl_post_training.launch.dispatch must expose a callable named "
        "one of: resolve_dispatch, dispatch, resolve, plan, build_plan."
    )


def _resolve_trainer_adapter_selector(dispatch_module):
    for attr in (
        "get_trainer_adapter",
        "select_trainer_adapter",
        "resolve_trainer_adapter",
        "trainer_adapter_for_plan",
        "build_trainer_adapter",
    ):
        fn = getattr(dispatch_module, attr, None)
        if callable(fn):
            return fn
    try:
        import verl_post_training.adapters.trainer as trainer_pkg
    except ModuleNotFoundError:
        trainer_pkg = None
    if trainer_pkg is not None:
        for attr in (
            "get_trainer_adapter",
            "select_trainer_adapter",
            "resolve_trainer_adapter",
            "trainer_adapter_for_plan",
            "build_trainer_adapter",
        ):
            fn = getattr(trainer_pkg, attr, None)
            if callable(fn):
                return fn
    pytest.fail(
        "Repo-level dispatch must expose a selector that returns a trainer "
        "adapter for a dispatch plan."
    )


def _trainer_backend_of(plan: object) -> str | None:
    if hasattr(plan, "trainer_backend"):
        value = plan.trainer_backend
    elif isinstance(plan, dict):
        value = plan.get("trainer_backend")
    else:
        return None
    if value is None:
        return None
    return getattr(value, "value", str(value))


def _task_type_of(plan: object) -> str | None:
    if hasattr(plan, "task_type"):
        value = plan.task_type
    elif isinstance(plan, dict):
        value = plan.get("task_type")
    else:
        return None
    if value is None:
        return None
    return getattr(value, "value", str(value))


def _write_yaml(tmp_path: Path, payload: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path


def _build_plan(load_config_module, dispatch_module, tmp_path, payload):
    loader = _resolve_loader(load_config_module)
    dispatcher = _resolve_dispatcher(dispatch_module)
    cfg = loader(_write_yaml(tmp_path, payload))
    return dispatcher(cfg)


def _adapter_instance_from_module(module) -> object:
    for attr in (
        "VERLTrainerAdapter",
        "VerlTrainerAdapter",
        "TrainerAdapter",
        "Adapter",
        "adapter",
    ):
        candidate = getattr(module, attr, None)
        if candidate is None:
            continue
        if isinstance(candidate, type):
            try:
                return candidate()
            except TypeError:
                continue
        return candidate
    pytest.fail(
        "verl_post_training.adapters.trainer.verl must expose a trainer "
        "adapter class or instance (e.g. VERLTrainerAdapter); "
        f"got module attributes: {sorted(vars(module))!r}"
    )


# ---------------------------------------------------------------------------
# Criterion: chat_rl + Qwen + verl resolves to the VERL trainer adapter
# ---------------------------------------------------------------------------


def test_qwen_chat_rl_config_loads_and_dispatches(
    load_config_module, dispatch_module, tmp_path
):
    plan = _build_plan(
        load_config_module, dispatch_module, tmp_path, _qwen_rl_config()
    )
    assert plan is not None, (
        "Dispatcher returned None for a known-good Qwen chat_rl config; "
        "the happy path must produce a usable dispatch plan."
    )


def test_qwen_chat_rl_dispatch_resolves_to_verl_trainer_backend(
    load_config_module, dispatch_module, tmp_path
):
    plan = _build_plan(
        load_config_module, dispatch_module, tmp_path, _qwen_rl_config()
    )
    assert _trainer_backend_of(plan) == "verl", (
        "Qwen chat_rl config must resolve to trainer_backend=verl; "
        f"got {_trainer_backend_of(plan)!r}"
    )
    assert _task_type_of(plan) == "chat_rl", (
        "Qwen chat_rl config must resolve to task_type=chat_rl; "
        f"got {_task_type_of(plan)!r}"
    )


def test_trainer_selector_returns_verl_adapter_for_chat_rl(
    load_config_module, dispatch_module, tmp_path
):
    plan = _build_plan(
        load_config_module, dispatch_module, tmp_path, _qwen_rl_config()
    )
    selector = _resolve_trainer_adapter_selector(dispatch_module)
    adapter = selector(plan)
    assert adapter is not None, (
        "Trainer-adapter selector returned None for a Qwen chat_rl + verl "
        "plan; the VERL trainer adapter must be reachable."
    )

    module_name = type(adapter).__module__ or ""
    qualname = type(adapter).__qualname__ or ""
    # Distinguish the VERL trainer adapter from the LLaMA-Factory or V-JEPA2
    # ones. We check for a module path containing `adapters.trainer.verl`
    # or a class name that starts with VERL/Verl (avoiding generic substring
    # matches that could collide with `vjepa2`).
    looks_like_verl_module = "adapters.trainer.verl" in module_name
    looks_like_verl_class = qualname.startswith(("VERL", "Verl"))
    assert looks_like_verl_module or looks_like_verl_class, (
        "Selector must return a VERL trainer adapter (expected the type "
        "to live in `verl_post_training.adapters.trainer.verl`); got "
        f"type {module_name}.{qualname}"
    )
    assert "vjepa2" not in module_name, (
        "Selector returned a V-JEPA2 trainer adapter for a chat_rl plan; "
        "this would be a cross-family regression."
    )
    assert "adapters.trainer.llamafactory" not in module_name, (
        "Selector returned the LLaMA-Factory trainer adapter for a chat_rl "
        "plan; the VERL adapter handles chat_rl."
    )


def test_chat_sft_does_not_resolve_to_verl_adapter(
    load_config_module, dispatch_module, tmp_path
):
    """Symmetric guard: a Qwen chat_sft plan must not land on the VERL
    trainer adapter. The selector must distinguish RL from SFT.
    """

    plan = _build_plan(
        load_config_module, dispatch_module, tmp_path, _qwen_sft_config()
    )
    selector = _resolve_trainer_adapter_selector(dispatch_module)
    adapter = selector(plan)
    if adapter is None:
        return
    module_name = type(adapter).__module__ or ""
    qualname = type(adapter).__qualname__ or ""
    assert "adapters.trainer.verl" not in module_name, (
        "Trainer-adapter selector returned the VERL trainer adapter for a "
        f"chat_sft plan; got {module_name}.{qualname}"
    )
    assert not qualname.startswith(("VERL", "Verl")), (
        "Trainer-adapter selector returned a VERL adapter type for a "
        f"chat_sft plan; got {module_name}.{qualname}"
    )


# ---------------------------------------------------------------------------
# Direct adapter rejection: caller cannot bypass dispatch
# ---------------------------------------------------------------------------


def test_verl_adapter_rejects_incompatible_task_type_directly():
    """If a caller bypasses the dispatcher and feeds a chat_sft plan to
    the VERL adapter directly, the adapter itself must refuse.
    """

    try:
        import verl_post_training.adapters.trainer.verl as module
    except ModuleNotFoundError as exc:
        pytest.fail(
            "M8 requires `verl_post_training.adapters.trainer.verl` to "
            f"exist; got ModuleNotFoundError: {exc!r}"
        )
    adapter = _adapter_instance_from_module(module)

    validate = None
    for attr in ("validate", "validate_plan", "check", "prepare", "plan", "build"):
        fn = getattr(adapter, attr, None)
        if callable(fn):
            validate = fn
            break
    if validate is None:
        pytest.fail(
            "VERL trainer adapter must expose a callable validate / "
            "prepare hook so capability errors surface even when callers "
            f"skip resolve_dispatch; got: {adapter!r}"
        )

    from verl_post_training.launch.dispatch import resolve_dispatch
    from verl_post_training.launch.load_config import TaskConfig

    good_cfg = TaskConfig.from_mapping(_qwen_rl_config())
    good_plan = resolve_dispatch(good_cfg)

    bad_plan = None
    try:
        import dataclasses

        if dataclasses.is_dataclass(good_plan):
            from verl_post_training.registry.schemas import TaskType

            bad_plan = dataclasses.replace(
                good_plan, task_type=TaskType.CHAT_SFT
            )
    except (TypeError, ValueError):
        bad_plan = None

    if bad_plan is None:
        pytest.skip(
            "Could not build a mutated plan to exercise the adapter's own "
            "rejection path; structural plan type is opaque."
        )

    with pytest.raises(Exception) as excinfo:
        validate(bad_plan)
    assert type(excinfo.value) is not Exception, (
        "adapter validate hook raised bare Exception for an incompatible "
        "plan; raise a typed compatibility error so callers can recover."
    )


def test_verl_adapter_rejects_non_vlm_chat_model_directly():
    """The VERL adapter must also refuse a plan whose registry entry is
    not a vlm_chat model (e.g. a video_encoder or world_model).
    """

    try:
        import verl_post_training.adapters.trainer.verl as module
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing module: {exc!r}")
    adapter = _adapter_instance_from_module(module)
    validate = None
    for attr in ("validate", "validate_plan", "check", "prepare", "plan", "build"):
        fn = getattr(adapter, attr, None)
        if callable(fn):
            validate = fn
            break
    if validate is None:
        pytest.fail("VERL trainer adapter must expose a validate hook.")

    from verl_post_training.launch.dispatch import resolve_dispatch
    from verl_post_training.launch.load_config import TaskConfig
    from verl_post_training.registry import get_model_entry

    good_cfg = TaskConfig.from_mapping(_qwen_rl_config())
    good_plan = resolve_dispatch(good_cfg)

    encoder_entry = get_model_entry("vjepa2-video-encoder-placeholder")

    bad_plan = None
    try:
        import dataclasses

        if dataclasses.is_dataclass(good_plan):
            bad_plan = dataclasses.replace(good_plan, model_entry=encoder_entry)
    except (TypeError, ValueError):
        bad_plan = None

    if bad_plan is None:
        pytest.skip(
            "Could not mutate the plan's model_entry; structural plan type "
            "is opaque."
        )

    with pytest.raises(Exception):
        validate(bad_plan)
