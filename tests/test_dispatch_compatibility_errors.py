"""M2 acceptance: the dispatcher rejects unsupported model-family / backend
combinations with an explicit compatibility error *before* any backend
process starts.

This file pins the fourth M2 acceptance criterion (quoted from the approved
plan):

    dispatching an unsupported combination such as
    ``model_family=world_model`` with ``runtime_backend=openai_chat_vllm``
    fails with an explicit compatibility error

It also covers the symmetric trainer-side case (asking for a vlm_chat model
to be trained with a non-chat trainer backend) and an unknown ``model_id``
path so the dispatcher's failure surface is uniform.

The dispatcher is allowed to fail by:

- raising a registry-owned or repo-owned exception class (preferred)
- raising a typed standard-library exception (``ValueError``, ``LookupError``,
  ``TypeError``) whose message names both the unsupported family and the
  unsupported backend so the failure is debuggable

A bare ``Exception``, an ``ImportError`` from a backend module, or any
silent fallback (returning ``None`` / a Qwen entry / a partial plan) is a
test failure.

Tests use ``tmp_path`` and ``monkeypatch`` to remain deterministic and
self-contained. No backend subprocess is started.
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


def _resolve_loader(load_config_module):
    for attr in ("load_config", "load", "load_task_config", "from_yaml"):
        fn = getattr(load_config_module, attr, None)
        if callable(fn):
            return fn
    pytest.fail(
        "verl_post_training.launch.load_config must expose a callable "
        "named one of: load_config, load, load_task_config, from_yaml."
    )


def _resolve_dispatcher(dispatch_module):
    """Return the dispatch entry point.

    Plan-level naming is not pinned, so accept any of a few natural spellings.
    """

    for attr in ("dispatch", "resolve", "plan", "resolve_dispatch", "build_plan"):
        fn = getattr(dispatch_module, attr, None)
        if callable(fn):
            return fn
    pytest.fail(
        "verl_post_training.launch.dispatch must expose a callable named "
        "one of: dispatch, resolve, plan, resolve_dispatch, build_plan."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, payload: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path


def _dispatch_from_yaml(
    load_config_module, dispatch_module, tmp_path, payload
):
    loader = _resolve_loader(load_config_module)
    dispatcher = _resolve_dispatcher(dispatch_module)
    config = loader(_write_yaml(tmp_path, payload))
    return dispatcher(config)


def _assert_explicit_compatibility_error(
    excinfo: pytest.ExceptionInfo[BaseException],
    *,
    must_mention: tuple[str, ...],
) -> None:
    """Assert that the captured exception is a typed compatibility error
    whose message names the offending pieces.
    """

    exc = excinfo.value
    assert type(exc) is not Exception, (
        "dispatcher raised a bare Exception for an incompatible combination; "
        "use a typed compatibility error so callers can distinguish it from "
        "unrelated bugs."
    )
    assert not isinstance(exc, ImportError), (
        "dispatcher imported a backend module before rejecting an "
        f"incompatible combination; got ImportError: {exc!r}"
    )

    msg = str(exc)
    missing = [token for token in must_mention if token not in msg]
    assert not missing, (
        f"compatibility error must name {sorted(must_mention)}; "
        f"missing {missing} from message: {msg!r}"
    )


def _compatible_qwen_config() -> dict[str, Any]:
    """A known-good config that should *not* fail compatibility checks."""

    return {
        "task_type": "chat_sft",
        "model_id": "qwen3-vl-4b-instruct",
        "trainer_backend": "llamafactory",
        "dataset_adapter": "chat_sft",
        "input_manifest": "data/pipeline/train_manifest.jsonl",
        "output_dir": "outputs/post_training/qwen_sft_run_001",
        "launcher": {
            "kind": "torchrun",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {
            "precision": "bf16",
            "devices": 1,
        },
        "backend_config": {},
    }


# ---------------------------------------------------------------------------
# Criterion 4: incompatible combinations fail with an explicit error
# ---------------------------------------------------------------------------


def test_world_model_with_openai_chat_runtime_fails(
    load_config_module, dispatch_module, tmp_path
):
    """The acceptance criterion's named-and-shamed regression case:
    pointing a ``world_model`` model id at the ``openai_chat_vllm`` runtime
    must fail before any backend process starts.
    """

    payload = {
        "task_type": "world_model_rollout",
        # Seeded world_model placeholder from the M1 registry.
        "model_id": "dreamdojo-world-model-placeholder",
        "runtime_backend": "openai_chat_vllm",
        "dataset_adapter": "dreamdojo",
        "input_manifest": "data/pipeline/world_model_manifest.jsonl",
        "output_dir": "outputs/post_training/world_model_run_001",
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

    _assert_explicit_compatibility_error(
        excinfo, must_mention=("world_model", "openai_chat_vllm")
    )


def test_vlm_chat_with_non_chat_runtime_fails(
    load_config_module, dispatch_module, tmp_path
):
    """Symmetric regression: a chat model must not be dispatched to a
    non-chat runtime backend (here ``vjepa2_native``).
    """

    payload = {
        "task_type": "chat_rl",
        "model_id": "qwen3-vl-4b-instruct",
        "runtime_backend": "vjepa2_native",
        "dataset_adapter": "chat_rl",
        "input_manifest": "data/pipeline/rl_manifest.jsonl",
        "output_dir": "outputs/post_training/qwen_rl_bad_runtime",
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

    _assert_explicit_compatibility_error(
        excinfo, must_mention=("vjepa2_native",)
    )


def test_vlm_chat_with_non_chat_trainer_fails(
    load_config_module, dispatch_module, tmp_path
):
    """A chat model must not be dispatched to a non-chat trainer backend.

    The Qwen entry advertises trainer support for ``llamafactory`` and
    ``verl`` only; selecting ``vjepa2_native`` must fail before any backend
    process is started.
    """

    payload = {
        "task_type": "chat_sft",
        "model_id": "qwen3-vl-4b-instruct",
        "trainer_backend": "vjepa2_native",
        "dataset_adapter": "chat_sft",
        "input_manifest": "data/pipeline/train_manifest.jsonl",
        "output_dir": "outputs/post_training/qwen_sft_bad_trainer",
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

    _assert_explicit_compatibility_error(
        excinfo, must_mention=("vjepa2_native",)
    )


def test_unsupported_task_type_for_model_fails(
    load_config_module, dispatch_module, tmp_path
):
    """Asking the Qwen entry to perform a task type it does not declare
    (``masked_video_prediction``) must fail with a compatibility error that
    names the requested task type.
    """

    payload = {
        "task_type": "masked_video_prediction",
        "model_id": "qwen3-vl-4b-instruct",
        "trainer_backend": "llamafactory",
        "dataset_adapter": "chat_sft",
        "input_manifest": "data/pipeline/train_manifest.jsonl",
        "output_dir": "outputs/post_training/qwen_bad_task",
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

    _assert_explicit_compatibility_error(
        excinfo, must_mention=("masked_video_prediction",)
    )


def test_unknown_model_id_dispatch_fails(
    load_config_module, dispatch_module, tmp_path
):
    """Dispatching with a model_id not in the registry must fail before any
    backend code is invoked.
    """

    payload = _compatible_qwen_config()
    payload["model_id"] = "totally-unknown-model-id-9999"

    with pytest.raises(Exception) as excinfo:
        _dispatch_from_yaml(
            load_config_module, dispatch_module, tmp_path, payload
        )

    assert "totally-unknown-model-id-9999" in str(excinfo.value), (
        "compatibility error must reference the unknown model_id; got: "
        f"{excinfo.value!r}"
    )
    assert not isinstance(excinfo.value, ImportError), (
        "dispatcher imported a backend before rejecting an unknown model_id; "
        f"got: {excinfo.value!r}"
    )


def test_incompatible_dispatch_does_not_import_backend(
    load_config_module, dispatch_module, tmp_path, monkeypatch
):
    """Compatibility checks must run before backend modules are imported.

    We make the four most likely backend modules unimportable; the dispatcher
    must still reject an incompatible combination cleanly. If we observe an
    ``ImportError`` here, that proves the dispatcher reached into a backend
    before the compatibility gate.
    """

    blocked_modules = ("vllm", "llamafactory", "verl", "torch")
    for name in blocked_modules:
        monkeypatch.setitem(sys.modules, name, None)

    payload = {
        "task_type": "world_model_rollout",
        "model_id": "dreamdojo-world-model-placeholder",
        "runtime_backend": "openai_chat_vllm",
        "dataset_adapter": "dreamdojo",
        "input_manifest": "data/pipeline/world_model_manifest.jsonl",
        "output_dir": "outputs/post_training/world_model_blocked_imports",
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

    assert not isinstance(excinfo.value, ImportError), (
        "dispatcher imported a backend module before rejecting an "
        f"incompatible combination; got ImportError: {excinfo.value!r}"
    )


# ---------------------------------------------------------------------------
# Regression guard: compatible combinations must still resolve cleanly
# ---------------------------------------------------------------------------


def test_compatible_chat_dispatch_succeeds(
    load_config_module, dispatch_module, tmp_path
):
    """The negative tests above must not be one-sided: a known-good Qwen
    chat config must still resolve through the dispatcher without raising.
    """

    payload = _compatible_qwen_config()

    plan = _dispatch_from_yaml(
        load_config_module, dispatch_module, tmp_path, payload
    )

    assert plan is not None, (
        "dispatcher returned None for a known-good Qwen chat config; the "
        "happy path must produce a usable dispatch plan."
    )
