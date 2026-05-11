"""M7 acceptance: a repo-level config with ``task_type=generation_inference``
and a Wan model id dispatches to the video-generation runtime adapter.

This file pins the second M7 acceptance criterion (quoted from the
approved plan):

    a repo-level config with ``task_type=generation_inference`` and a
    Wan model id dispatches to the video generation runtime adapter

Concretely, these tests pin:

* a YAML config with ``task_type=generation_inference`` and the seeded
  Wan model id loads without raising and dispatches through
  ``verl_post_training.launch.dispatch.resolve_dispatch`` without raising
* the resolved dispatch plan advertises ``runtime_backend=wan_native``
  and points at a ``video_generator`` registry entry
* a runtime-adapter selector resolves the plan to the video-generation
  runtime adapter exposed under
  ``verl_post_training.adapters.runtime.video_generation``
* a chat-family model attempting ``generation_inference`` is rejected
  before reaching the video-generation runtime adapter

The tests must run without GPUs or upstream Wan2.2 weights: dispatch is
a control-plane concern only, and any inner inference seam in the
runtime adapter is monkeypatched in the artifact-contract tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


# ---------------------------------------------------------------------------
# Constants / fixtures
# ---------------------------------------------------------------------------


WAN_PLACEHOLDER_MODEL_ID = "wan-video-generator-placeholder"


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


def _wan_inference_config(
    *,
    model_id: str = WAN_PLACEHOLDER_MODEL_ID,
    output_dir: str = "outputs/post_training/wan_inference_run_001",
    input_manifest: str = "data/pipeline/wan_conditioning_manifest.jsonl",
    runtime_backend: str = "wan_native",
) -> dict[str, Any]:
    return {
        "task_type": "generation_inference",
        "model_id": model_id,
        "runtime_backend": runtime_backend,
        "dataset_adapter": "wan",
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
    """Return a callable that maps a dispatch plan to a runtime adapter."""

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
# Criterion 2: generation_inference + Wan model dispatches to the
# video-generation runtime adapter
# ---------------------------------------------------------------------------


def test_wan_inference_config_loads_and_dispatches(
    load_config_module, dispatch_module, tmp_path
):
    """The repo-level YAML must load and dispatch without raising for the
    Wan generation_inference path.
    """

    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _wan_inference_config(),
    )
    assert plan is not None, (
        "Dispatcher returned None for a known-good Wan inference config; "
        "the happy path must produce a usable dispatch plan."
    )


def test_wan_dispatch_resolves_to_wan_native_runtime_backend(
    load_config_module, dispatch_module, tmp_path
):
    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _wan_inference_config(),
    )

    assert _runtime_backend_of(plan) == "wan_native", (
        "Wan inference config must resolve to runtime_backend=wan_native; "
        f"got {_runtime_backend_of(plan)!r}"
    )


def test_wan_dispatch_resolves_to_generation_inference_task_type(
    load_config_module, dispatch_module, tmp_path
):
    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _wan_inference_config(),
    )

    assert _task_type_of(plan) == "generation_inference", (
        "Wan inference config must resolve to task_type=generation_inference; "
        f"got {_task_type_of(plan)!r}"
    )


def test_wan_dispatch_points_at_video_generator_family(
    load_config_module, dispatch_module, tmp_path
):
    """The dispatched plan must reference a ``video_generator`` registry
    entry — selecting any other family for generation_inference would be
    a silent regression that lets non-generative models reach the video
    generation runtime adapter.
    """

    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _wan_inference_config(),
    )
    entry = _model_entry_of(plan)
    assert entry is not None, (
        "Dispatch plan must expose the resolved registry entry so callers "
        "can route to the video-generation runtime adapter."
    )
    family = getattr(entry, "model_family", None)
    family_value = getattr(family, "value", family)
    assert family_value == "video_generator", (
        "Wan model must dispatch as model_family=video_generator; "
        f"got {family_value!r}"
    )


def test_runtime_adapter_selector_returns_video_generation_adapter(
    load_config_module, dispatch_module, tmp_path
):
    """Given a plan resolved from the Wan inference config, the
    runtime-adapter selector must return the video-generation runtime
    adapter defined under
    ``verl_post_training.adapters.runtime.video_generation``.
    """

    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _wan_inference_config(),
    )
    selector = _resolve_runtime_adapter_selector(dispatch_module)

    adapter = selector(plan)
    assert adapter is not None, (
        "Runtime adapter selector returned None for a generation_inference "
        "+ Wan plan; the video-generation runtime adapter must be reachable."
    )

    module_name = type(adapter).__module__ or ""
    qualname = type(adapter).__qualname__ or ""
    # The video-generation runtime adapter must live under
    # `verl_post_training.adapters.runtime.video_generation` per the plan.
    assert (
        "adapters.runtime.video_generation" in module_name
        or "video_generation" in qualname.lower()
        or "videogeneration" in qualname.lower()
    ), (
        "Selector must return a video-generation runtime adapter (expected "
        "the type to live in "
        "`verl_post_training.adapters.runtime.video_generation`); got "
        f"type {module_name}.{qualname}"
    )


def test_chat_model_does_not_resolve_to_video_generation_runtime_adapter(
    load_config_module, dispatch_module, tmp_path
):
    """Symmetric guard: a chat model attempting generation_inference must
    not silently land on the video-generation runtime adapter.

    The Qwen entries do not advertise ``generation_inference`` so dispatch
    itself should reject this config; this test pins that rejection so a
    future writer cannot bypass it by widening the video-generation
    runtime adapter selector.
    """

    payload = _wan_inference_config(model_id="qwen3-vl-4b-instruct")
    payload["runtime_backend"] = "openai_chat_vllm"
    with pytest.raises(Exception):
        _build_plan(
            load_config_module, dispatch_module, tmp_path, payload
        )


def test_video_encoder_model_does_not_resolve_to_video_generation_runtime_adapter(
    load_config_module, dispatch_module, tmp_path
):
    """A video-encoder model (V-JEPA2 family) must not be acceptable as
    a video-generation backend. Dispatch must reject this combination
    rather than letting the encoder reach the generation runtime path.
    """

    payload = _wan_inference_config(model_id="vjepa2-video-encoder-placeholder")
    payload["runtime_backend"] = "vjepa2_native"
    with pytest.raises(Exception):
        _build_plan(
            load_config_module, dispatch_module, tmp_path, payload
        )


def test_wan_dispatch_rejects_mismatched_runtime_backend(
    load_config_module, dispatch_module, tmp_path
):
    """If the user pairs the Wan model id with a non-wan runtime backend
    (e.g. ``openai_chat_vllm``), dispatch must fail. Otherwise the
    runtime-adapter selector would have to make an arbitrary choice.
    """

    payload = _wan_inference_config(runtime_backend="openai_chat_vllm")
    with pytest.raises(Exception):
        _build_plan(
            load_config_module, dispatch_module, tmp_path, payload
        )


def test_wan_registry_entry_advertises_generation_inference_and_wan_native(
    registry_module,
):
    """Sanity guard against a regression in the seeded registry. The Wan
    placeholder entry must advertise both ``generation_inference`` and
    ``wan_native`` so dispatch can resolve. If this guard breaks, the
    registry seed itself has drifted and dispatch will appear to fail
    for unrelated reasons.
    """

    from verl_post_training.registry.schemas import RuntimeBackend, TaskType

    entry = registry_module.get_model_entry(WAN_PLACEHOLDER_MODEL_ID)
    assert TaskType.GENERATION_INFERENCE in entry.supported_task_types, (
        f"{WAN_PLACEHOLDER_MODEL_ID!r} must advertise "
        f"generation_inference; got {entry.supported_task_types!r}"
    )
    assert RuntimeBackend.WAN_NATIVE in entry.runtime_backends, (
        f"{WAN_PLACEHOLDER_MODEL_ID!r} must advertise wan_native runtime "
        f"backend; got {entry.runtime_backends!r}"
    )
