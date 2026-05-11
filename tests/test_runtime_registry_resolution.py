"""M3 acceptance: the local runtime resolves ``openai_chat_vllm`` through
the repo-owned registry rather than hard-coded Qwen-only branches.

This file pins criterion 1 from the approved plan section for milestone M3
(quoted):

    when given the current supported Qwen selector or explicit Qwen model
    id, the runtime resolves ``openai_chat_vllm`` through the registry
    instead of hard-coded Qwen-only branches

To prove "through the registry" the resolved spec must carry registry-
derived metadata. Concretely, the resolved spec is expected to either:

- expose a ``runtime_backend`` field whose value equals
  ``openai_chat_vllm`` (the M1 ``RuntimeBackend`` enum value), or
- expose a ``model_entry`` / ``registry_entry`` field that is the
  ``ModelRegistryEntry`` returned by ``get_model_entry`` for a
  ``vlm_chat`` family.

The implementer may keep ``resolve_local_runtime_spec`` returning a dict and
just add the new key, or upgrade to a dataclass with attributes — both are
accepted. The point is that the runtime-backend identity is recoverable
from the resolved spec.

To prove the registry is *actually* consulted (and not just hard-coded), one
test patches ``verl_post_training.registry.get_model_entry`` and asserts
that resolving the explicit Qwen model id reaches that function before any
backend startup occurs.

Tests are deterministic and self-contained. No vLLM, no GPU, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

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


def _enum_value(value: Any) -> str:
    """Return a plain string for either an enum member or a raw string."""

    return getattr(value, "value", value)


def _runtime_backend_of(spec: object) -> str | None:
    """Recover the ``runtime_backend`` identifier from a runtime spec.

    Tolerates the two natural shapes the implementer may pick:

    1. Dict with a ``runtime_backend`` key (string or enum), or
    2. Dataclass / object with a ``runtime_backend`` attribute, or
    3. Dict / object exposing a ``model_entry`` / ``registry_entry`` whose
       ``runtime_backends`` tuple is a single element.
    """

    if isinstance(spec, dict):
        if "runtime_backend" in spec:
            return _enum_value(spec["runtime_backend"])
        entry = spec.get("model_entry") or spec.get("registry_entry")
        if entry is not None:
            backends = tuple(getattr(entry, "runtime_backends", ()))
            if len(backends) == 1:
                return _enum_value(backends[0])
        return None

    if hasattr(spec, "runtime_backend"):
        return _enum_value(getattr(spec, "runtime_backend"))
    entry = getattr(spec, "model_entry", None) or getattr(spec, "registry_entry", None)
    if entry is not None:
        backends = tuple(getattr(entry, "runtime_backends", ()))
        if len(backends) == 1:
            return _enum_value(backends[0])
    return None


def _model_entry_of(spec: object) -> object | None:
    """Recover a registry entry from a runtime spec if one is attached."""

    if isinstance(spec, dict):
        return spec.get("model_entry") or spec.get("registry_entry")
    return getattr(spec, "model_entry", None) or getattr(spec, "registry_entry", None)


def _registered_runtime_backend(spec: object) -> str:
    backend = _runtime_backend_of(spec)
    if backend is not None:
        return backend
    pytest.fail(
        "runtime spec must carry registry-derived metadata so callers can "
        "tell that the resolved runtime backend is `openai_chat_vllm`. "
        "Expected either a `runtime_backend` field (string or enum), or a "
        "`model_entry` / `registry_entry` field whose `runtime_backends` is "
        f"a single-element tuple. Got: {spec!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 1: known Qwen selectors resolve via the registry to
# ``openai_chat_vllm``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("selector", ["thinking", "instruct", "thinking32b"])
def test_known_selectors_resolve_to_openai_chat_vllm_through_registry(
    runtime_module, selector
):
    """Each of the historically supported selectors must now resolve to the
    registry-backed ``openai_chat_vllm`` runtime backend. This pins the
    "through the registry" half of the criterion: the resolved spec must
    carry the runtime backend identity, not just a base URL.
    """

    spec = runtime_module.resolve_local_runtime_spec(selector)
    assert _registered_runtime_backend(spec) == "openai_chat_vllm", (
        f"selector {selector!r} must resolve to runtime_backend "
        f"`openai_chat_vllm`; got {_runtime_backend_of(spec)!r}"
    )


def test_explicit_qwen_registry_model_id_resolves_to_openai_chat_vllm(
    runtime_module,
):
    """The criterion explicitly names "explicit Qwen model id" as a
    supported input. Passing the seeded registry id (``qwen3-vl-4b-instruct``)
    must resolve to the registry-backed openai_chat_vllm runtime backend.
    """

    spec = runtime_module.resolve_local_runtime_spec("qwen3-vl-4b-instruct")
    assert _registered_runtime_backend(spec) == "openai_chat_vllm", (
        "passing the seeded Qwen registry model_id must resolve through the "
        "registry to runtime_backend=openai_chat_vllm; got "
        f"{_runtime_backend_of(spec)!r}"
    )


def test_resolved_qwen_spec_carries_vlm_chat_family(runtime_module, registry_module):
    """If a registry entry is exposed on the spec, it must be a ``vlm_chat``
    entry: ``openai_chat_vllm`` is the chat-family runtime backend, so a
    resolved entry pointing anywhere else is a regression.
    """

    spec = runtime_module.resolve_local_runtime_spec("thinking")
    entry = _model_entry_of(spec)
    if entry is None:
        # Equivalent shape: a direct ``model_family`` field on the spec.
        family = None
        if isinstance(spec, dict):
            family = spec.get("model_family")
        else:
            family = getattr(spec, "model_family", None)
        if family is None:
            pytest.skip(
                "spec does not expose a registry entry or `model_family`; "
                "skipping family check — `runtime_backend` is asserted elsewhere"
            )
        assert _enum_value(family) == "vlm_chat", (
            f"resolved Qwen spec must declare model_family=vlm_chat; got {family!r}"
        )
        return

    assert _enum_value(entry.model_family) == "vlm_chat", (
        f"resolved Qwen spec must point at a vlm_chat registry entry; got "
        f"model_family={entry.model_family!r}"
    )


def test_resolution_actually_calls_into_the_registry(
    runtime_module, registry_module, monkeypatch
):
    """The acceptance criterion says "through the registry" — not "produces
    a value that happens to match the registry". This test monkeypatches
    ``get_model_entry`` so that *any* registry call is recorded, then
    resolves the seeded Qwen registry id and asserts the registry was
    actually consulted.
    """

    calls: list[str] = []
    real_get_model_entry = registry_module.get_model_entry

    def tracking_get_model_entry(model_id: str):
        calls.append(model_id)
        return real_get_model_entry(model_id)

    # The runtime layer is allowed to import either the symbol or the
    # module; patch the canonical location so re-imports see the wrapper.
    monkeypatch.setattr(
        registry_module, "get_model_entry", tracking_get_model_entry
    )
    monkeypatch.setattr(
        "verl_post_training.registry.model_registry.get_model_entry",
        tracking_get_model_entry,
        raising=False,
    )
    # If the runtime module captured ``get_model_entry`` at import time,
    # patch that reference too (best-effort; non-fatal if absent).
    if hasattr(runtime_module, "get_model_entry"):
        monkeypatch.setattr(
            runtime_module, "get_model_entry", tracking_get_model_entry
        )

    spec = runtime_module.resolve_local_runtime_spec("qwen3-vl-4b-instruct")

    assert "qwen3-vl-4b-instruct" in calls, (
        "resolving the seeded Qwen registry id must call into "
        "`verl_post_training.registry.get_model_entry` so the runtime is "
        "demonstrably registry-backed; observed calls: " f"{calls!r}"
    )
    # The resolved backend must still be openai_chat_vllm — guards against
    # a stub that calls the registry but ignores the result.
    assert _registered_runtime_backend(spec) == "openai_chat_vllm"


def test_runtime_backend_value_matches_m1_enum(runtime_module, registry_module):
    """Whatever the resolved spec advertises must match the canonical M1
    ``RuntimeBackend.OPENAI_CHAT_VLLM`` value. This guards against drift
    between the runtime layer and the registry enums.
    """

    spec = runtime_module.resolve_local_runtime_spec("thinking")
    backend = _registered_runtime_backend(spec)
    expected = registry_module.RuntimeBackend.OPENAI_CHAT_VLLM.value
    assert backend == expected, (
        f"resolved runtime backend {backend!r} must equal the M1 enum value "
        f"{expected!r}"
    )
