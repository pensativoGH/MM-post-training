"""M6B acceptance: a repo-level config with ``task_type=masked_video_prediction``,
``trainer_backend=vjepa2_native``, and a compatible V-JEPA2 model id dispatches
to the V-JEPA2 trainer adapter.

This file pins the first M6B acceptance criterion (quoted from the approved
plan):

    a repo-level config with ``task_type=masked_video_prediction``,
    ``trainer_backend=vjepa2_native``, and a compatible V-JEPA2 ``model_id``
    dispatches to the V-JEPA2 trainer adapter

Concretely, these tests pin:

* the registry exposes at least one V-JEPA2 model entry that advertises both
  the ``masked_video_prediction`` task type and the ``vjepa2_native`` trainer
  backend so the dispatch path is reachable end-to-end
* a YAML config with that combination loads via the M2 loader and resolves
  through ``verl_post_training.launch.dispatch.resolve_dispatch`` without
  raising
* the resolved dispatch plan advertises ``trainer_backend=vjepa2_native`` and
  ``task_type=masked_video_prediction``
* a trainer-adapter selector returns the V-JEPA2 trainer adapter exposed
  under ``verl_post_training.adapters.trainer.vjepa2``
* a chat dispatch (Qwen + LLaMA-Factory) does *not* resolve to the V-JEPA2
  trainer adapter — the selector must not widen across families

The tests must run without GPUs or upstream V-JEPA2 weights; we only assert
on dispatch wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


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


@pytest.fixture(scope="module")
def registry_module():
    import verl_post_training.registry as module

    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Return a callable that maps a dispatch plan to a trainer adapter."""

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
        ):
            fn = getattr(trainer_pkg, attr, None)
            if callable(fn):
                return fn

    pytest.fail(
        "Repo-level dispatch must expose a selector that returns a trainer "
        "adapter for a dispatch plan. Expected one of: get_trainer_adapter, "
        "select_trainer_adapter, resolve_trainer_adapter, "
        "trainer_adapter_for_plan, build_trainer_adapter on "
        "verl_post_training.launch.dispatch or "
        "verl_post_training.adapters.trainer."
    )


def _find_vjepa2_training_model_id() -> str:
    """Return the seeded model_id that advertises masked_video_prediction +
    vjepa2_native trainer support, failing the test if none exists.
    """

    from verl_post_training.registry import iter_entries
    from verl_post_training.registry.schemas import (
        TaskType,
        TrainerBackend,
    )

    candidates: list[str] = []
    for entry in iter_entries():
        if (
            TaskType.MASKED_VIDEO_PREDICTION in entry.supported_task_types
            and TrainerBackend.VJEPA2_NATIVE in entry.trainer_backends
        ):
            candidates.append(entry.model_id)

    if not candidates:
        pytest.fail(
            "M6B requires at least one V-JEPA2 registry entry that advertises "
            "both task_type=masked_video_prediction and "
            "trainer_backend=vjepa2_native; none found. The registry must be "
            "extended (or a new entry added) so dispatch can resolve a "
            "compatible V-JEPA2 model id for masked-video-prediction training."
        )

    return candidates[0]


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


def _model_entry_of(plan: object):
    return (
        getattr(plan, "model_entry", None)
        or getattr(plan, "registry_entry", None)
        or (plan.get("model_entry") if isinstance(plan, dict) else None)
    )


def _vjepa2_training_config(
    *,
    model_id: str,
    output_dir: str = "outputs/post_training/vjepa2_training_run_001",
    input_manifest: str = "data/pipeline/video_train_manifest.jsonl",
) -> dict[str, Any]:
    return {
        "task_type": "masked_video_prediction",
        "model_id": model_id,
        "trainer_backend": "vjepa2_native",
        "dataset_adapter": "vjepa2",
        "input_manifest": input_manifest,
        "output_dir": output_dir,
        "launcher": {
            "kind": "python_module",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }


def _write_yaml(tmp_path: Path, payload: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path


def _build_plan(load_config_module, dispatch_module, tmp_path, payload):
    loader = _resolve_loader(load_config_module)
    dispatcher = _resolve_dispatcher(dispatch_module)
    cfg = loader(_write_yaml(tmp_path, payload))
    return dispatcher(cfg)


# ---------------------------------------------------------------------------
# Criterion 1: dispatch resolves to the V-JEPA2 trainer adapter
# ---------------------------------------------------------------------------


def test_registry_exposes_vjepa2_masked_video_prediction_entry():
    """The control plane cannot resolve masked-video-prediction training
    without a registry entry that declares both the task and the trainer
    backend. Pin that contract explicitly.
    """

    model_id = _find_vjepa2_training_model_id()
    assert model_id, (
        "Expected at least one V-JEPA2 model entry with "
        "MASKED_VIDEO_PREDICTION + VJEPA2_NATIVE; got none."
    )


def test_vjepa2_training_config_loads_and_dispatches(
    load_config_module, dispatch_module, tmp_path
):
    """The repo-level YAML must load and dispatch without raising for the
    V-JEPA2 masked-video-prediction training path.
    """

    model_id = _find_vjepa2_training_model_id()
    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _vjepa2_training_config(model_id=model_id),
    )
    assert plan is not None, (
        "Dispatcher returned None for a known-good V-JEPA2 training config; "
        "the happy path must produce a usable dispatch plan."
    )


def test_vjepa2_training_dispatch_resolves_to_vjepa2_native_trainer_backend(
    load_config_module, dispatch_module, tmp_path
):
    model_id = _find_vjepa2_training_model_id()
    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _vjepa2_training_config(model_id=model_id),
    )

    assert _trainer_backend_of(plan) == "vjepa2_native", (
        "V-JEPA2 training config must resolve to trainer_backend="
        f"vjepa2_native; got {_trainer_backend_of(plan)!r}"
    )
    assert _task_type_of(plan) == "masked_video_prediction", (
        "V-JEPA2 training config must resolve to task_type="
        f"masked_video_prediction; got {_task_type_of(plan)!r}"
    )


def test_vjepa2_training_dispatch_points_at_vjepa2_registry_entry(
    load_config_module, dispatch_module, tmp_path
):
    """The dispatched plan must reference the V-JEPA2 registry entry so the
    trainer adapter selector can route correctly. The model_family is not
    required to be ``video_encoder`` (M6B may add a separate family or
    entry) — but the entry must at minimum echo the dispatched model_id.
    """

    model_id = _find_vjepa2_training_model_id()
    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _vjepa2_training_config(model_id=model_id),
    )

    entry = _model_entry_of(plan)
    assert entry is not None, (
        "Dispatch plan must expose the resolved registry entry so callers "
        "can route to the V-JEPA2 trainer adapter."
    )
    entry_model_id = getattr(entry, "model_id", None)
    assert entry_model_id == model_id, (
        "Resolved registry entry must echo the dispatched model_id; "
        f"got entry.model_id={entry_model_id!r}, expected {model_id!r}"
    )


def test_trainer_adapter_selector_returns_vjepa2_trainer_adapter(
    load_config_module, dispatch_module, tmp_path
):
    """Given a plan resolved from the V-JEPA2 training config, the trainer
    adapter selector must return the V-JEPA2 trainer adapter defined under
    ``verl_post_training.adapters.trainer.vjepa2``.
    """

    model_id = _find_vjepa2_training_model_id()
    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _vjepa2_training_config(model_id=model_id),
    )
    selector = _resolve_trainer_adapter_selector(dispatch_module)

    adapter = selector(plan)
    assert adapter is not None, (
        "Trainer adapter selector returned None for a "
        "masked_video_prediction + V-JEPA2 plan; the V-JEPA2 trainer "
        "adapter must be reachable."
    )

    module_name = type(adapter).__module__ or ""
    qualname = type(adapter).__qualname__ or ""
    assert (
        "adapters.trainer.vjepa2" in module_name
        or "vjepa2" in qualname.lower()
    ), (
        "Selector must return a V-JEPA2 trainer adapter (expected the type "
        "to live in `verl_post_training.adapters.trainer.vjepa2`); got "
        f"type {module_name}.{qualname}"
    )


def test_chat_dispatch_does_not_return_vjepa2_trainer_adapter(
    load_config_module, dispatch_module, tmp_path
):
    """Symmetric guard: a Qwen chat-SFT plan must not resolve to the
    V-JEPA2 trainer adapter. If it did, the selector would be ignoring the
    plan's trainer_backend / family.
    """

    payload = {
        "task_type": "chat_sft",
        "model_id": "qwen3-vl-4b-instruct",
        "trainer_backend": "llamafactory",
        "dataset_adapter": "chat_sft",
        "input_manifest": "data/pipeline/train_manifest.jsonl",
        "output_dir": "outputs/post_training/qwen_sft_check",
        "launcher": {
            "kind": "torchrun",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }
    plan = _build_plan(load_config_module, dispatch_module, tmp_path, payload)
    selector = _resolve_trainer_adapter_selector(dispatch_module)

    adapter = selector(plan)
    if adapter is None:
        # Acceptable: selector may refuse a chat plan. The negative assertion
        # only kicks in when an adapter is returned.
        return

    module_name = type(adapter).__module__ or ""
    qualname = type(adapter).__qualname__ or ""
    assert "adapters.trainer.vjepa2" not in module_name, (
        "Trainer selector returned the V-JEPA2 trainer adapter for a Qwen "
        "chat plan; this would be a cross-family regression. Got "
        f"{module_name}.{qualname}"
    )
    assert "vjepa2" not in qualname.lower(), (
        "Trainer selector returned a vjepa2-named adapter type for a Qwen "
        f"chat plan; got {module_name}.{qualname}"
    )
