"""M6B acceptance: the V-JEPA2 trainer adapter rejects incompatible
combinations *before* any backend process starts.

This file pins the second M6B acceptance criterion (quoted from the
approved plan):

    the trainer adapter rejects incompatible combinations, including a
    non-V-JEPA2 ``model_id`` or an inference-only task type, before any
    backend process starts

Concretely, these tests pin:

* a config with ``task_type=masked_video_prediction`` +
  ``trainer_backend=vjepa2_native`` + a non-V-JEPA2 ``model_id`` (e.g. the
  Qwen chat entry) fails with a typed compatibility error
* a config with ``trainer_backend=vjepa2_native`` + an inference-only task
  type (``embedding_inference``) fails with a typed compatibility error
* a config with ``task_type=masked_video_prediction`` +
  ``runtime_backend=vjepa2_native`` (i.e. running training under a runtime
  backend) fails — the V-JEPA2 trainer path must not be silently driven
  from the runtime side
* the failure path does not import a heavyweight backend module
  (``torch`` / ``vllm`` / ``llamafactory`` / ``verl``) before the
  compatibility gate
* the V-JEPA2 trainer adapter exposes a ``validate``/``prepare`` style
  hook that also rejects an incompatible plan when called directly (so the
  rejection cannot be bypassed by a caller that skips ``resolve_dispatch``)

Tests use ``tmp_path`` and ``monkeypatch`` to stay deterministic. No
backend subprocess is started.
"""

from __future__ import annotations

import sys
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
        return None
    for attr in (
        "get_trainer_adapter",
        "select_trainer_adapter",
        "resolve_trainer_adapter",
        "trainer_adapter_for_plan",
    ):
        fn = getattr(trainer_pkg, attr, None)
        if callable(fn):
            return fn
    return None


def _find_vjepa2_training_model_id() -> str:
    from verl_post_training.registry import iter_entries
    from verl_post_training.registry.schemas import (
        TaskType,
        TrainerBackend,
    )

    for entry in iter_entries():
        if (
            TaskType.MASKED_VIDEO_PREDICTION in entry.supported_task_types
            and TrainerBackend.VJEPA2_NATIVE in entry.trainer_backends
        ):
            return entry.model_id
    pytest.fail(
        "M6B requires at least one V-JEPA2 registry entry that advertises "
        "both masked_video_prediction and vjepa2_native; none found."
    )


def _write_yaml(tmp_path: Path, payload: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path


def _dispatch_from_yaml(load_config_module, dispatch_module, tmp_path, payload):
    loader = _resolve_loader(load_config_module)
    dispatcher = _resolve_dispatcher(dispatch_module)
    config = loader(_write_yaml(tmp_path, payload))
    return dispatcher(config)


def _assert_typed_compatibility_error(
    excinfo: pytest.ExceptionInfo[BaseException],
    *,
    must_mention: tuple[str, ...],
) -> None:
    exc = excinfo.value
    assert type(exc) is not Exception, (
        "Trainer dispatch raised a bare Exception for an incompatible "
        "combination; use a typed compatibility error so callers can "
        "distinguish it from unrelated bugs."
    )
    assert not isinstance(exc, ImportError), (
        "Trainer dispatch imported a backend module before rejecting an "
        f"incompatible combination; got ImportError: {exc!r}"
    )
    assert not isinstance(exc, ModuleNotFoundError), (
        "Trainer dispatch failed by attempting to import a backend module "
        f"before the compatibility gate; got ModuleNotFoundError: {exc!r}"
    )

    msg = str(exc)
    missing = [token for token in must_mention if token not in msg]
    assert not missing, (
        f"compatibility error must name {sorted(must_mention)}; "
        f"missing {missing} from message: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 2.a: non-V-JEPA2 model_id with the V-JEPA2 trainer backend fails
# ---------------------------------------------------------------------------


def test_non_vjepa2_model_with_vjepa2_trainer_fails(
    load_config_module, dispatch_module, tmp_path
):
    """Pointing the V-JEPA2 trainer backend at a Qwen chat model_id must
    fail with a typed compatibility error before any backend process starts.
    """

    payload = {
        "task_type": "masked_video_prediction",
        "model_id": "qwen3-vl-4b-instruct",
        "trainer_backend": "vjepa2_native",
        "dataset_adapter": "vjepa2",
        "input_manifest": "data/pipeline/video_train_manifest.jsonl",
        "output_dir": "outputs/post_training/vjepa2_bad_model",
        "launcher": {
            "kind": "python_module",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }

    with pytest.raises(Exception) as excinfo:
        _dispatch_from_yaml(
            load_config_module, dispatch_module, tmp_path, payload
        )

    _assert_typed_compatibility_error(
        excinfo, must_mention=("masked_video_prediction",)
    )


def test_unknown_model_with_vjepa2_trainer_fails(
    load_config_module, dispatch_module, tmp_path
):
    """An unknown model_id must fail before any backend process starts."""

    payload = {
        "task_type": "masked_video_prediction",
        "model_id": "definitely-not-a-real-model-9999",
        "trainer_backend": "vjepa2_native",
        "dataset_adapter": "vjepa2",
        "input_manifest": "data/pipeline/video_train_manifest.jsonl",
        "output_dir": "outputs/post_training/vjepa2_unknown_model",
        "launcher": {
            "kind": "python_module",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }

    with pytest.raises(Exception) as excinfo:
        _dispatch_from_yaml(
            load_config_module, dispatch_module, tmp_path, payload
        )

    assert "definitely-not-a-real-model-9999" in str(excinfo.value), (
        "compatibility error must reference the unknown model_id; got: "
        f"{excinfo.value!r}"
    )
    assert not isinstance(excinfo.value, ImportError), (
        "Dispatch imported a backend before rejecting an unknown model_id; "
        f"got: {excinfo.value!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 2.b: inference-only task type with the V-JEPA2 trainer fails
# ---------------------------------------------------------------------------


def test_inference_task_with_vjepa2_trainer_fails(
    load_config_module, dispatch_module, tmp_path
):
    """The V-JEPA2 trainer backend must reject an inference-only task type
    such as ``embedding_inference``. The trainer path is only for training.
    """

    payload = {
        "task_type": "embedding_inference",
        "model_id": "vjepa2-video-encoder-placeholder",
        "trainer_backend": "vjepa2_native",
        "dataset_adapter": "vjepa2",
        "input_manifest": "data/pipeline/video_manifest.jsonl",
        "output_dir": "outputs/post_training/vjepa2_inference_as_training",
        "launcher": {
            "kind": "python_module",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }

    with pytest.raises(Exception) as excinfo:
        _dispatch_from_yaml(
            load_config_module, dispatch_module, tmp_path, payload
        )

    _assert_typed_compatibility_error(
        excinfo, must_mention=("embedding_inference",)
    )


def test_chat_task_with_vjepa2_trainer_fails(
    load_config_module, dispatch_module, tmp_path
):
    """A chat task type with the V-JEPA2 trainer backend must also fail —
    the trainer adapter is for masked-video-prediction only.
    """

    payload = {
        "task_type": "chat_sft",
        "model_id": "qwen3-vl-4b-instruct",
        "trainer_backend": "vjepa2_native",
        "dataset_adapter": "chat_sft",
        "input_manifest": "data/pipeline/train_manifest.jsonl",
        "output_dir": "outputs/post_training/qwen_with_vjepa2_trainer",
        "launcher": {
            "kind": "torchrun",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }

    with pytest.raises(Exception) as excinfo:
        _dispatch_from_yaml(
            load_config_module, dispatch_module, tmp_path, payload
        )

    _assert_typed_compatibility_error(
        excinfo, must_mention=("vjepa2_native",)
    )


# ---------------------------------------------------------------------------
# Criterion 2.c: training task driven via runtime_backend fails
# ---------------------------------------------------------------------------


def test_masked_video_prediction_via_runtime_backend_fails(
    load_config_module, dispatch_module, tmp_path
):
    """Training must dispatch through the trainer adapter, not a runtime
    adapter. A masked-video-prediction config that sets only
    ``runtime_backend=vjepa2_native`` must be rejected so the trainer path
    is the only way to launch training.
    """

    model_id = _find_vjepa2_training_model_id()
    payload = {
        "task_type": "masked_video_prediction",
        "model_id": model_id,
        # No trainer_backend; instead try to drive training via runtime.
        "runtime_backend": "vjepa2_native",
        "dataset_adapter": "vjepa2",
        "input_manifest": "data/pipeline/video_train_manifest.jsonl",
        "output_dir": "outputs/post_training/vjepa2_runtime_as_trainer",
        "launcher": {
            "kind": "python_module",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }

    with pytest.raises(Exception):
        _dispatch_from_yaml(
            load_config_module, dispatch_module, tmp_path, payload
        )


# ---------------------------------------------------------------------------
# Criterion 2.d: failures land before backend modules are imported
# ---------------------------------------------------------------------------


def test_incompatible_training_dispatch_does_not_import_backend(
    load_config_module, dispatch_module, tmp_path, monkeypatch
):
    """Compatibility checks for the V-JEPA2 trainer must run before any
    heavyweight backend module is imported. We block the obvious candidates
    and confirm an incompatible config still fails cleanly — not with an
    ``ImportError`` from a backend module.
    """

    blocked_modules = ("vllm", "llamafactory", "verl", "torch")
    for name in blocked_modules:
        monkeypatch.setitem(sys.modules, name, None)

    payload = {
        "task_type": "masked_video_prediction",
        "model_id": "qwen3-vl-4b-instruct",  # not a V-JEPA2 entry
        "trainer_backend": "vjepa2_native",
        "dataset_adapter": "vjepa2",
        "input_manifest": "data/pipeline/video_train_manifest.jsonl",
        "output_dir": "outputs/post_training/vjepa2_blocked_imports",
        "launcher": {
            "kind": "python_module",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }

    with pytest.raises(Exception) as excinfo:
        _dispatch_from_yaml(
            load_config_module, dispatch_module, tmp_path, payload
        )

    assert not isinstance(excinfo.value, (ImportError, ModuleNotFoundError)), (
        "Trainer dispatch imported a backend module before rejecting an "
        f"incompatible combination; got: {excinfo.value!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 2.e: trainer adapter itself refuses an incompatible plan
# ---------------------------------------------------------------------------


def _resolve_trainer_adapter_module():
    """Import the V-JEPA2 trainer adapter module if it exists."""

    try:
        import verl_post_training.adapters.trainer.vjepa2 as module
    except ModuleNotFoundError as exc:
        pytest.fail(
            "M6B requires "
            "`verl_post_training.adapters.trainer.vjepa2` to exist; got "
            f"ModuleNotFoundError: {exc!r}"
        )
    return module


def _find_adapter_validate_hook(adapter):
    """Return the first callable validation hook the adapter exposes."""

    for attr in ("validate", "validate_plan", "check", "prepare", "plan", "build"):
        fn = getattr(adapter, attr, None)
        if callable(fn):
            return fn, attr
    return None, None


def test_vjepa2_trainer_adapter_rejects_incompatible_plan_directly(
    load_config_module, dispatch_module, tmp_path
):
    """If a caller bypasses ``resolve_dispatch`` and constructs a plan by
    hand, the V-JEPA2 trainer adapter itself must still refuse an
    incompatible combination. Otherwise the rejection sits only at the
    dispatcher and can be silently bypassed.
    """

    trainer_module = _resolve_trainer_adapter_module()

    # Find the adapter class/instance the module exposes.
    adapter = None
    for attr in (
        "VJEPA2TrainerAdapter",
        "VJepa2TrainerAdapter",
        "Adapter",
        "TrainerAdapter",
        "adapter",
    ):
        candidate = getattr(trainer_module, attr, None)
        if candidate is None:
            continue
        if isinstance(candidate, type):
            try:
                adapter = candidate()
            except TypeError:
                continue
            break
        adapter = candidate
        break

    if adapter is None:
        pytest.fail(
            "verl_post_training.adapters.trainer.vjepa2 must expose a "
            "trainer adapter class or instance (e.g. VJEPA2TrainerAdapter); "
            f"got module attributes: {sorted(vars(trainer_module))!r}"
        )

    hook, hook_name = _find_adapter_validate_hook(adapter)
    if hook is None:
        pytest.fail(
            "V-JEPA2 trainer adapter must expose a callable validate / "
            "prepare hook so capability errors surface even when callers "
            f"skip resolve_dispatch; got: {adapter!r}"
        )

    # Build a plan-shaped object by reaching through dispatch with a *good*
    # config, then handing the adapter a mutated copy that points at a chat
    # model. Use a real dispatch plan as the structural template.
    loader = _resolve_loader(load_config_module)
    dispatcher = _resolve_dispatcher(dispatch_module)
    model_id = _find_vjepa2_training_model_id()
    good_payload = {
        "task_type": "masked_video_prediction",
        "model_id": model_id,
        "trainer_backend": "vjepa2_native",
        "dataset_adapter": "vjepa2",
        "input_manifest": "data/pipeline/video_train_manifest.jsonl",
        "output_dir": "outputs/post_training/vjepa2_good_plan",
        "launcher": {
            "kind": "python_module",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }
    good_plan = dispatcher(loader(_write_yaml(tmp_path, good_payload)))

    # Construct a "bad" plan with the Qwen entry swapped in to simulate a
    # caller that bypassed compatibility. We try a few shape-tolerant
    # mutation strategies.
    from verl_post_training.registry import get_model_entry

    chat_entry = get_model_entry("qwen3-vl-4b-instruct")

    bad_plan = None
    if hasattr(good_plan, "_replace"):  # NamedTuple
        try:
            bad_plan = good_plan._replace(model_entry=chat_entry)
        except (AttributeError, ValueError):
            bad_plan = None

    if bad_plan is None:
        try:
            import dataclasses

            if dataclasses.is_dataclass(good_plan):
                bad_plan = dataclasses.replace(good_plan, model_entry=chat_entry)
        except (TypeError, ValueError):
            bad_plan = None

    if bad_plan is None and isinstance(good_plan, dict):
        bad_plan = dict(good_plan)
        bad_plan["model_entry"] = chat_entry

    if bad_plan is None:
        pytest.skip(
            "Could not build a mutated plan to exercise the adapter's own "
            "rejection path; structural plan type is opaque."
        )

    with pytest.raises(Exception) as excinfo:
        hook(bad_plan)

    assert type(excinfo.value) is not Exception, (
        f"adapter.{hook_name} raised bare Exception for an incompatible "
        "plan; raise a typed compatibility error so callers can recover."
    )
