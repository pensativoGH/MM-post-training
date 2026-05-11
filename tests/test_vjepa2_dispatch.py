"""M6 acceptance: a repo-level config with ``task_type=embedding_inference``
and a V-JEPA2 model id dispatches to the encoder runtime adapter, and the
wrapper returns a normalized output with enough metadata for a smoke test
to assert success.

This file pins acceptance criteria 2 and 3 from the approved plan section
for milestone M6 (quoted):

    a repo-level config with ``task_type=embedding_inference`` and a
    V-JEPA2 model id dispatches to the encoder runtime adapter

    the wrapper returns normalized output containing enough metadata for a
    smoke test to assert success, including ``model_id``, ``task_type``,
    ``output_dir``, and a per-example result status

Concretely, these tests pin:

* a YAML config with ``task_type=embedding_inference`` and the seeded
  V-JEPA2 model id loads without raising and dispatches through
  ``verl_post_training.launch.dispatch.resolve_dispatch`` without raising
* the resolved dispatch plan advertises ``runtime_backend=vjepa2_native``
  and points at a ``video_encoder`` registry entry
* a runtime adapter selector resolves the plan to the encoder runtime
  adapter exposed under ``verl_post_training.adapters.runtime.encoder``
* invoking the encoder runtime adapter returns a normalized mapping that
  contains ``model_id``, ``task_type``, ``output_dir``, and a per-example
  result status block

The tests must run without GPUs or upstream V-JEPA2 weights: the encoder
runtime adapter is expected to expose a seam (``run`` / ``invoke`` / etc.)
that does not require importing the upstream package to *return* the
normalized envelope shape. We monkeypatch any inner inference call as
needed so the tests stay deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


# ---------------------------------------------------------------------------
# Constants / fixtures
# ---------------------------------------------------------------------------


VJEPA2_PLACEHOLDER_MODEL_ID = "vjepa2-video-encoder-placeholder"


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


def _vjepa2_inference_config(
    *,
    model_id: str = VJEPA2_PLACEHOLDER_MODEL_ID,
    output_dir: str = "outputs/post_training/vjepa2_inference_run_001",
    input_manifest: str = "data/pipeline/video_manifest.jsonl",
) -> dict[str, Any]:
    return {
        "task_type": "embedding_inference",
        "model_id": model_id,
        "runtime_backend": "vjepa2_native",
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


def _resolve_runtime_adapter_selector(dispatch_module):
    """Return a callable that maps a dispatch plan to a runtime adapter.

    The writer may choose a few reasonable names — we accept any of them.
    """

    for attr in (
        "get_runtime_adapter",
        "select_runtime_adapter",
        "resolve_runtime_adapter",
        "runtime_adapter_for_plan",
        "build_runtime_adapter",
    ):
        fn = getattr(dispatch_module, attr, None)
        if callable(fn):
            return fn

    # Allow the writer to expose the selector from the adapters subpackage too.
    try:
        import verl_post_training.adapters.runtime as runtime_pkg
    except ModuleNotFoundError:
        runtime_pkg = None
    if runtime_pkg is not None:
        for attr in (
            "get_runtime_adapter",
            "select_runtime_adapter",
            "resolve_runtime_adapter",
            "runtime_adapter_for_plan",
        ):
            fn = getattr(runtime_pkg, attr, None)
            if callable(fn):
                return fn

    pytest.fail(
        "Repo-level dispatch must expose a selector that returns a runtime "
        "adapter for a dispatch plan. Expected one of: get_runtime_adapter, "
        "select_runtime_adapter, resolve_runtime_adapter, "
        "runtime_adapter_for_plan, build_runtime_adapter on "
        "verl_post_training.launch.dispatch or "
        "verl_post_training.adapters.runtime."
    )


def _runtime_backend_of(plan: object) -> str | None:
    """Recover the runtime backend identifier from a dispatch plan."""

    if hasattr(plan, "runtime_backend"):
        value = plan.runtime_backend
    elif isinstance(plan, dict):
        value = plan.get("runtime_backend")
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


def _invoke_adapter(adapter, plan):
    """Invoke whatever entry-point the encoder runtime adapter exposes.

    Accept ``run``, ``invoke``, ``execute``, or ``__call__`` — the contract
    is just that the adapter is *callable* against a dispatch plan.
    """

    for attr in ("run", "invoke", "execute"):
        fn = getattr(adapter, attr, None)
        if callable(fn):
            return fn(plan)
    if callable(adapter):
        return adapter(plan)
    pytest.fail(
        "Encoder runtime adapter must expose a callable entry point "
        f"(run/invoke/execute/__call__); got: {adapter!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 2: config with embedding_inference + V-JEPA2 model dispatches
# to the encoder runtime adapter
# ---------------------------------------------------------------------------


def test_vjepa2_inference_config_loads_and_dispatches(
    load_config_module, dispatch_module, tmp_path
):
    """The repo-level YAML must load and dispatch without raising for the
    V-JEPA2 embedding inference path.
    """

    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _vjepa2_inference_config(),
    )
    assert plan is not None, (
        "Dispatcher returned None for a known-good V-JEPA2 inference "
        "config; the happy path must produce a usable dispatch plan."
    )


def test_vjepa2_dispatch_resolves_to_vjepa2_native_runtime_backend(
    load_config_module, dispatch_module, tmp_path
):
    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _vjepa2_inference_config(),
    )

    assert _runtime_backend_of(plan) == "vjepa2_native", (
        "V-JEPA2 inference config must resolve to runtime_backend="
        f"vjepa2_native; got {_runtime_backend_of(plan)!r}"
    )


def test_vjepa2_dispatch_points_at_video_encoder_family(
    load_config_module, dispatch_module, tmp_path
):
    """The dispatched plan must reference a ``video_encoder`` registry
    entry — selecting any other family for embedding inference would be a
    silent regression.
    """

    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _vjepa2_inference_config(),
    )
    entry = (
        getattr(plan, "model_entry", None)
        or getattr(plan, "registry_entry", None)
        or (plan.get("model_entry") if isinstance(plan, dict) else None)
    )
    assert entry is not None, (
        "Dispatch plan must expose the resolved registry entry so callers "
        "can route to the encoder runtime adapter."
    )
    family = getattr(entry, "model_family", None)
    family_value = getattr(family, "value", family)
    assert family_value == "video_encoder", (
        f"V-JEPA2 model must dispatch as model_family=video_encoder; "
        f"got {family_value!r}"
    )


def test_runtime_adapter_selector_returns_encoder_adapter(
    load_config_module, dispatch_module, tmp_path
):
    """Given a plan resolved from the V-JEPA2 inference config, the
    runtime-adapter selector must return the encoder runtime adapter
    defined under ``verl_post_training.adapters.runtime.encoder``.
    """

    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _vjepa2_inference_config(),
    )
    selector = _resolve_runtime_adapter_selector(dispatch_module)

    adapter = selector(plan)
    assert adapter is not None, (
        "Runtime adapter selector returned None for an embedding_inference "
        "+ V-JEPA2 plan; the encoder runtime adapter must be reachable."
    )

    module_name = type(adapter).__module__ or ""
    qualname = type(adapter).__qualname__ or ""
    # The encoder runtime adapter must live under the
    # `verl_post_training.adapters.runtime.encoder` module per the plan.
    assert (
        "adapters.runtime.encoder" in module_name
        or "encoder" in qualname.lower()
    ), (
        "Selector must return an encoder runtime adapter (expected the type "
        "to live in `verl_post_training.adapters.runtime.encoder`); got "
        f"type {module_name}.{qualname}"
    )


def test_non_vjepa2_model_does_not_resolve_to_encoder_runtime_adapter(
    load_config_module, dispatch_module, tmp_path, monkeypatch
):
    """Symmetric guard: a chat-model embedding_inference attempt must not
    silently land on the encoder runtime adapter.

    The Qwen entries do not advertise ``embedding_inference`` so dispatch
    itself should reject this config; this test pins that rejection so a
    future writer cannot bypass it by widening the encoder runtime adapter
    selector.
    """

    payload = _vjepa2_inference_config(model_id="qwen3-vl-4b-instruct")
    payload["runtime_backend"] = "openai_chat_vllm"
    with pytest.raises(Exception):
        _build_plan(
            load_config_module, dispatch_module, tmp_path, payload
        )


# ---------------------------------------------------------------------------
# Criterion 3: wrapper returns a normalized envelope with model_id /
# task_type / output_dir / per-example status
# ---------------------------------------------------------------------------


def _coerce_envelope(envelope: object) -> dict[str, Any]:
    """Allow the wrapper to return a dataclass or a dict; coerce to mapping."""

    if isinstance(envelope, dict):
        return envelope
    if hasattr(envelope, "__dict__"):
        return dict(vars(envelope))
    raise AssertionError(
        f"V-JEPA2 wrapper return value must be a mapping or dataclass; got: "
        f"{envelope!r}"
    )


def _ensure_per_example_status(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the per-example result block under one of a few reasonable keys."""

    for key in ("per_example", "examples", "results", "outputs", "items"):
        value = envelope.get(key)
        if isinstance(value, list) and value:
            return value
    pytest.fail(
        "V-JEPA2 wrapper envelope must include a per-example result list "
        "under one of: per_example, examples, results, outputs, items. "
        f"Got envelope keys: {sorted(envelope.keys())!r}"
    )


def test_vjepa2_runtime_adapter_returns_normalized_envelope(
    load_config_module, dispatch_module, tmp_path, monkeypatch
):
    """Invoking the encoder runtime adapter against a dispatch plan must
    return an envelope that surfaces ``model_id``, ``task_type``,
    ``output_dir``, and a per-example status block.

    The acceptance criterion is about the *envelope*, not GPU correctness —
    so we monkeypatch any underlying inference seam to avoid loading the
    upstream V-JEPA2 model.
    """

    # Build a fake input manifest the adapter can iterate over.
    manifest_path = tmp_path / "video_manifest.jsonl"
    manifest_path.write_text(
        '{"example_id": "vid_000", "media_paths": ["s3://bucket/a.mp4"], "modality": "video"}\n'
        '{"example_id": "vid_001", "media_paths": ["s3://bucket/b.mp4"], "modality": "video"}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "encoder_out"
    output_dir.mkdir()

    payload = _vjepa2_inference_config(
        input_manifest=str(manifest_path),
        output_dir=str(output_dir),
    )
    plan = _build_plan(
        load_config_module, dispatch_module, tmp_path, payload
    )
    selector = _resolve_runtime_adapter_selector(dispatch_module)
    adapter = selector(plan)

    # Best-effort: stub any inner inference hook so we never need GPU/weights.
    encoder_module = sys.modules.get(
        "verl_post_training.adapters.runtime.encoder"
    )
    if encoder_module is not None:
        for stub_name in (
            "_run_encoder",
            "run_encoder",
            "_encode_example",
            "encode_example",
            "_invoke_upstream",
        ):
            if hasattr(encoder_module, stub_name):
                monkeypatch.setattr(
                    encoder_module,
                    stub_name,
                    lambda *args, **kwargs: {"status": "ok"},
                )

    envelope = _invoke_adapter(adapter, plan)
    envelope_map = _coerce_envelope(envelope)

    for required_key in ("model_id", "task_type", "output_dir"):
        assert required_key in envelope_map, (
            f"V-JEPA2 wrapper envelope must include {required_key!r}; "
            f"got keys: {sorted(envelope_map.keys())!r}"
        )

    assert envelope_map["model_id"] == VJEPA2_PLACEHOLDER_MODEL_ID, (
        f"envelope.model_id must echo the dispatched model_id; got "
        f"{envelope_map['model_id']!r}"
    )
    task_type_val = envelope_map["task_type"]
    task_type_str = getattr(task_type_val, "value", task_type_val)
    assert task_type_str == "embedding_inference", (
        f"envelope.task_type must be 'embedding_inference'; got "
        f"{task_type_str!r}"
    )
    assert str(envelope_map["output_dir"]) == str(output_dir), (
        f"envelope.output_dir must match the configured output_dir "
        f"{str(output_dir)!r}; got {envelope_map['output_dir']!r}"
    )

    per_example = _ensure_per_example_status(envelope_map)
    assert len(per_example) >= 1, (
        "Per-example status list must contain at least one entry for a "
        "non-empty input manifest."
    )
    for item in per_example:
        item_map = item if isinstance(item, dict) else dict(vars(item))
        assert "status" in item_map, (
            "Each per-example record must carry a `status` field so smoke "
            f"tests can assert success; got record keys: {sorted(item_map.keys())!r}"
        )
