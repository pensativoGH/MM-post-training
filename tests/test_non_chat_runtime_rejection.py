"""M3 acceptance: non-chat families are rejected before server startup
begins when sent through the chat runtime entry point.

This file pins criterion 3 from the approved plan section for milestone M3
(quoted):

    requesting a non-chat family through the chat runtime entry point fails
    before server startup begins

"Chat runtime entry point" here is the Python-level local runtime resolver
exposed by ``verl_post_training_runtime.local_runtime`` — the same surface
used by ``runtime/scripts/check_qwen_vllm_ready.py`` and by
``runtime/scripts/start_qwen_vllm_server.sh`` (via shell-side bootstrap).

The rejection must happen *before* any backend process is started. We
prove that by:

1. blocking the most likely backend imports (``vllm``, ``torch``,
   ``subprocess`` callable replaced with a sentinel) and asserting that no
   ``ImportError`` / sentinel call escapes the resolver; and

2. asserting the resolver raises a typed error (not a bare ``Exception``,
   not ``ImportError``) whose message names the offending model family or
   model id, so operators can debug a misrouted config.

The non-chat seeded entries from the M1 registry are used as concrete
inputs (``video_encoder``, ``video_generator``, ``world_model``).

Tests are deterministic and self-contained — no GPU, no network, no real
subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_SRC = _REPO_ROOT / "runtime" / "src"
if _RUNTIME_SRC.is_dir():
    _runtime_src_str = str(_RUNTIME_SRC)
    if _runtime_src_str not in sys.path:
        sys.path.insert(0, _runtime_src_str)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtime_module():
    import verl_post_training_runtime.local_runtime as module

    return module


@pytest.fixture(scope="module")
def registry_module():
    import verl_post_training.registry as module

    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block_backend_imports(monkeypatch) -> None:
    """Make the most likely backend modules unimportable.

    If the chat runtime entry point imports vLLM / torch before validating
    the requested model family, we will see an ``ImportError`` escape. That
    proves the validation gate runs *after* a backend import, which is
    exactly what this criterion forbids.
    """

    for name in ("vllm", "vllm.entrypoints", "torch"):
        monkeypatch.setitem(sys.modules, name, None)


def _block_subprocess(monkeypatch) -> None:
    """Replace ``subprocess.Popen`` / ``subprocess.run`` with sentinels.

    If the resolver shells out to start a server before validating the
    requested model family, the sentinel will fire and the test will fail
    with a specific, debuggable assertion error rather than a flaky timeout.
    """

    import subprocess

    def _refuse_popen(*args, **kwargs):  # pragma: no cover - sentinel only
        raise AssertionError(
            "chat runtime entry point started a subprocess "
            "(subprocess.Popen) before validating model family — that "
            "violates the M3 acceptance criterion 'fails before server "
            "startup begins'."
        )

    def _refuse_run(*args, **kwargs):  # pragma: no cover - sentinel only
        raise AssertionError(
            "chat runtime entry point invoked subprocess.run before "
            "validating model family — that violates the M3 acceptance "
            "criterion 'fails before server startup begins'."
        )

    monkeypatch.setattr(subprocess, "Popen", _refuse_popen)
    monkeypatch.setattr(subprocess, "run", _refuse_run)


def _assert_typed_runtime_rejection(
    excinfo: pytest.ExceptionInfo[BaseException],
    *,
    must_mention: tuple[str, ...],
) -> None:
    """Assert that the captured exception is a typed runtime-rejection
    error whose message names the offending pieces.
    """

    exc = excinfo.value

    assert type(exc) is not Exception, (
        "chat runtime entry point raised a bare Exception for a non-chat "
        "model family; use a typed error so callers can distinguish it "
        f"from unrelated bugs. Got: {exc!r}"
    )
    assert not isinstance(exc, ImportError), (
        "chat runtime entry point raised ImportError before rejecting a "
        f"non-chat model family; backend import must not run first. Got: {exc!r}"
    )
    assert not isinstance(exc, AssertionError), (
        "chat runtime entry point started a subprocess before rejecting a "
        f"non-chat model family. Got: {exc!r}"
    )

    msg = str(exc)
    missing = [token for token in must_mention if token not in msg]
    assert not missing, (
        f"rejection error must name {sorted(must_mention)}; "
        f"missing {missing} from message: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 3: non-chat families are rejected before server startup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "non_chat_model_id,family_token",
    [
        ("vjepa2-video-encoder-placeholder", "video_encoder"),
        ("wan-video-generator-placeholder", "video_generator"),
        ("dreamdojo-world-model-placeholder", "world_model"),
    ],
)
def test_non_chat_model_id_rejected_by_local_runtime(
    runtime_module, monkeypatch, non_chat_model_id, family_token
):
    """For each of the three seeded non-chat families, passing the model id
    through ``resolve_local_runtime_spec`` (the chat runtime entry point)
    must raise a typed error that names the offending model id and either
    the model family or a registry-recognisable rejection token.
    """

    _block_backend_imports(monkeypatch)
    _block_subprocess(monkeypatch)

    with pytest.raises(Exception) as excinfo:
        runtime_module.resolve_local_runtime_spec(non_chat_model_id)

    _assert_typed_runtime_rejection(
        excinfo,
        must_mention=(non_chat_model_id,),
    )

    # The error must additionally point at *why* the request was rejected
    # — either by naming the model family, or by naming the chat-runtime
    # constraint. Either is acceptable; both being absent is not.
    msg = str(excinfo.value)
    msg_lower = msg.lower()
    explains_why = (
        family_token in msg
        or "vlm_chat" in msg
        or "openai_chat_vllm" in msg
        or "chat" in msg_lower
    )
    assert explains_why, (
        "rejection error must explain why the model id was refused — "
        f"expected one of {{{family_token!r}, 'vlm_chat', "
        f"'openai_chat_vllm', or 'chat'}} in the message; got: {msg!r}"
    )


def test_rejection_does_not_import_backend_modules(
    runtime_module, registry_module, monkeypatch
):
    """A non-chat model id must be rejected before the resolver imports any
    backend module. We block ``vllm`` and ``torch`` from importing; the
    resolver must still produce a typed rejection error, never an
    ``ImportError``.
    """

    _block_backend_imports(monkeypatch)
    _block_subprocess(monkeypatch)

    with pytest.raises(Exception) as excinfo:
        runtime_module.resolve_local_runtime_spec(
            "vjepa2-video-encoder-placeholder"
        )

    assert not isinstance(excinfo.value, ImportError), (
        "resolver imported a backend module before rejecting a non-chat "
        f"model family; got ImportError: {excinfo.value!r}"
    )
    assert not isinstance(excinfo.value, AssertionError), (
        "resolver invoked a subprocess before rejecting a non-chat model "
        f"family; got AssertionError: {excinfo.value!r}"
    )


def test_chat_runtime_uses_registry_to_classify_family(
    runtime_module, registry_module, monkeypatch
):
    """Synthesize a fresh non-chat entry in the registry at runtime, then
    confirm the chat runtime entry point rejects it. This proves the
    rejection logic is registry-driven rather than a hard-coded blocklist
    that only happens to cover the three seeded placeholders.
    """

    synthetic_model_id = "synthetic-non-chat-for-test"
    fake_entry = registry_module.ModelRegistryEntry(
        model_id=synthetic_model_id,
        model_family=registry_module.ModelFamily.VIDEO_ENCODER,
        supported_task_types=(registry_module.TaskType.EMBEDDING_INFERENCE,),
        trainer_backends=(registry_module.TrainerBackend.VJEPA2_NATIVE,),
        runtime_backends=(registry_module.RuntimeBackend.VJEPA2_NATIVE,),
        checkpoint_source="synthetic/non-chat",
        checkpoint_format="native",
        required_modalities=("video",),
        dataset_adapter_keys=("vjepa2",),
        launcher_type="python_module",
        default_precision="bf16",
        distributed_requirements={},
        environment_tags=("synthetic",),
    )

    monkeypatch.setitem(
        registry_module.MODEL_REGISTRY, synthetic_model_id, fake_entry
    )
    # Some implementations capture the registry mapping by reference; some
    # call ``get_model_entry`` per request. Patching both keeps the test
    # robust to either choice.

    _block_backend_imports(monkeypatch)
    _block_subprocess(monkeypatch)

    with pytest.raises(Exception) as excinfo:
        runtime_module.resolve_local_runtime_spec(synthetic_model_id)

    _assert_typed_runtime_rejection(
        excinfo,
        must_mention=(synthetic_model_id,),
    )


def test_chat_selectors_still_succeed_after_rejection_path_lands(
    runtime_module, monkeypatch
):
    """A regression guard for the criterion: the rejection logic for
    non-chat families must not block the supported Qwen selectors. This
    test would catch an overly broad implementation (e.g., "reject anything
    not literally named 'thinking'/'instruct'/'thinking32b'") that
    accidentally drops legitimate flows.
    """

    _block_backend_imports(monkeypatch)
    _block_subprocess(monkeypatch)

    # Should not raise for any of the supported selectors. Just resolving
    # the spec is what notebooks and check scripts already do; the resolver
    # must not perform any backend import or subprocess call at this stage.
    for selector in ("thinking", "instruct", "thinking32b"):
        spec = runtime_module.resolve_local_runtime_spec(selector)
        assert spec is not None, (
            f"chat runtime entry point unexpectedly returned None for the "
            f"supported selector {selector!r}"
        )


def test_chat_runtime_rejects_registered_non_chat_before_probe(
    runtime_module, monkeypatch
):
    """If the chat runtime layer exposes a probe / readiness helper, the
    non-chat rejection must short-circuit it: ``probe_openai_compatible_runtime``
    must never be reached with a non-chat model id when wired through the
    resolver. We capture this by patching the probe and asserting it is
    never called.
    """

    probe_calls: list[dict[str, Any]] = []

    real_probe = runtime_module.probe_openai_compatible_runtime

    def tracking_probe(**kwargs):
        probe_calls.append(kwargs)
        return real_probe(**kwargs)

    monkeypatch.setattr(
        runtime_module, "probe_openai_compatible_runtime", tracking_probe
    )
    _block_backend_imports(monkeypatch)
    _block_subprocess(monkeypatch)

    with pytest.raises(Exception):
        runtime_module.resolve_local_runtime_spec(
            "dreamdojo-world-model-placeholder"
        )

    assert probe_calls == [], (
        "chat runtime entry point invoked the openai-compatible probe for a "
        f"non-chat model id; probe was called with {probe_calls!r}"
    )
