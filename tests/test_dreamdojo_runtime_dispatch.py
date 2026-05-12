"""M8 acceptance: a repo-level config with a DreamDojo model id dispatches
to a normalized world-model runtime adapter rather than a chat runtime
path.

This file pins the second M8 acceptance criterion (quoted from the
approved plan):

    a repo-level config with a DreamDojo model id dispatches to a
    normalized world-model runtime adapter rather than a chat runtime path

Concretely, these tests pin:

* a YAML config with ``task_type=world_model_rollout`` and the seeded
  DreamDojo model id loads via the M2 loader and dispatches through
  ``verl_post_training.launch.dispatch.resolve_dispatch`` without raising
* the resolved dispatch plan advertises ``runtime_backend=dreamdojo``
  and points at a ``world_model`` registry entry
* a runtime-adapter selector resolves the plan to a world-model runtime
  adapter defined under ``verl_post_training.adapters.runtime.world_model``
* the selector does **not** return the OpenAI-chat / vLLM runtime adapter
  for a DreamDojo plan
* a chat config pointed at a DreamDojo model id fails — DreamDojo must
  never route through the chat runtime path

Tests must run without GPUs, upstream DreamDojo, or vLLM. Dispatch is a
control-plane concern only; the runtime adapter's inner execution seam is
not exercised here (it may be deferred per the M8 deferral clause — see
``test_capability_error_messages.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


# ---------------------------------------------------------------------------
# Constants / fixtures
# ---------------------------------------------------------------------------


DREAMDOJO_PLACEHOLDER_MODEL_ID = "dreamdojo-world-model-placeholder"


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


def _dreamdojo_rollout_config(
    *,
    model_id: str = DREAMDOJO_PLACEHOLDER_MODEL_ID,
    output_dir: str = "outputs/post_training/dreamdojo_rollout_run_001",
    input_manifest: str = "data/pipeline/dreamdojo_trajectory_manifest.jsonl",
    runtime_backend: str = "dreamdojo",
) -> dict[str, Any]:
    return {
        "task_type": "world_model_rollout",
        "model_id": model_id,
        "runtime_backend": runtime_backend,
        "dataset_adapter": "dreamdojo",
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
            "build_runtime_adapter",
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
# Criterion: DreamDojo dispatch reaches the world-model runtime adapter
# ---------------------------------------------------------------------------


def test_dreamdojo_runtime_config_loads_and_dispatches(
    load_config_module, dispatch_module, tmp_path
):
    """The repo-level YAML must load and dispatch without raising for the
    DreamDojo world_model_rollout path.
    """

    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _dreamdojo_rollout_config(),
    )
    assert plan is not None, (
        "Dispatcher returned None for a known-good DreamDojo rollout "
        "config; the happy path must produce a usable dispatch plan."
    )


def test_dreamdojo_dispatch_resolves_to_dreamdojo_runtime_backend(
    load_config_module, dispatch_module, tmp_path
):
    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _dreamdojo_rollout_config(),
    )

    assert _runtime_backend_of(plan) == "dreamdojo", (
        "DreamDojo rollout config must resolve to runtime_backend=dreamdojo; "
        f"got {_runtime_backend_of(plan)!r}"
    )
    assert _task_type_of(plan) == "world_model_rollout", (
        "DreamDojo rollout config must resolve to task_type="
        f"world_model_rollout; got {_task_type_of(plan)!r}"
    )


def test_dreamdojo_dispatch_points_at_world_model_family(
    load_config_module, dispatch_module, tmp_path
):
    """The dispatched plan must reference a ``world_model`` registry entry
    — selecting any other family for ``world_model_rollout`` would be a
    silent regression that lets non-world-model models reach the
    world-model runtime adapter.
    """

    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _dreamdojo_rollout_config(),
    )
    entry = _model_entry_of(plan)
    assert entry is not None, (
        "Dispatch plan must expose the resolved registry entry so callers "
        "can route to the world-model runtime adapter."
    )
    family = getattr(entry, "model_family", None)
    family_value = getattr(family, "value", family)
    assert family_value == "world_model", (
        "DreamDojo model must dispatch as model_family=world_model; "
        f"got {family_value!r}"
    )


def test_runtime_adapter_selector_returns_world_model_adapter(
    load_config_module, dispatch_module, tmp_path
):
    """Given a plan resolved from the DreamDojo rollout config, the
    runtime-adapter selector must return a world-model runtime adapter
    defined under ``verl_post_training.adapters.runtime.world_model``.
    """

    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _dreamdojo_rollout_config(),
    )
    selector = _resolve_runtime_adapter_selector(dispatch_module)

    adapter = selector(plan)
    assert adapter is not None, (
        "Runtime adapter selector returned None for a world_model_rollout "
        "+ DreamDojo plan; the world-model runtime adapter must be "
        "reachable."
    )

    module_name = type(adapter).__module__ or ""
    qualname = type(adapter).__qualname__ or ""
    assert (
        "adapters.runtime.world_model" in module_name
        or "world_model" in qualname.lower()
        or "worldmodel" in qualname.lower()
        or "dreamdojo" in module_name.lower()
        or "dreamdojo" in qualname.lower()
    ), (
        "Selector must return a world-model runtime adapter (expected "
        "the type to live in "
        "`verl_post_training.adapters.runtime.world_model`); got "
        f"type {module_name}.{qualname}"
    )


def test_runtime_adapter_selector_does_not_return_chat_runtime_for_dreamdojo(
    load_config_module, dispatch_module, tmp_path
):
    """Symmetric guard: the DreamDojo plan must never resolve to the
    OpenAI/vLLM chat runtime adapter. If the selector ever returns a
    type whose module name includes ``openai_chat_vllm`` for a DreamDojo
    plan, the world-model boundary has been silently widened.
    """

    plan = _build_plan(
        load_config_module,
        dispatch_module,
        tmp_path,
        _dreamdojo_rollout_config(),
    )
    selector = _resolve_runtime_adapter_selector(dispatch_module)
    adapter = selector(plan)

    module_name = type(adapter).__module__ or ""
    qualname = type(adapter).__qualname__ or ""
    assert "openai_chat_vllm" not in module_name, (
        "Runtime selector returned an openai_chat_vllm adapter for a "
        f"DreamDojo plan; got {module_name}.{qualname}"
    )
    assert "openaichat" not in qualname.lower(), (
        "Runtime selector returned an openai-chat adapter for a DreamDojo "
        f"plan; got {module_name}.{qualname}"
    )
    assert "chat_vllm" not in qualname.lower(), (
        "Runtime selector returned a chat_vllm adapter for a DreamDojo "
        f"plan; got {module_name}.{qualname}"
    )


def test_dreamdojo_model_with_openai_chat_runtime_fails(
    load_config_module, dispatch_module, tmp_path
):
    """Pairing the DreamDojo model id with the openai_chat_vllm runtime
    backend must fail — DreamDojo must not be reachable via the chat
    runtime path.
    """

    payload = _dreamdojo_rollout_config(runtime_backend="openai_chat_vllm")
    with pytest.raises(Exception):
        _build_plan(
            load_config_module, dispatch_module, tmp_path, payload
        )


def test_dreamdojo_runtime_adapter_module_exists():
    """The plan names the world-model runtime adapter module explicitly;
    pin its presence so the writer does not put DreamDojo behind an
    unrelated runtime adapter.
    """

    try:
        import verl_post_training.adapters.runtime.world_model as module
    except ModuleNotFoundError as exc:
        pytest.fail(
            "M8 requires "
            "`verl_post_training.adapters.runtime.world_model` to exist; "
            f"got ModuleNotFoundError: {exc!r}"
        )
    assert any(
        callable(getattr(module, attr, None))
        or isinstance(getattr(module, attr, None), type)
        for attr in dir(module)
        if not attr.startswith("_")
    ), (
        "verl_post_training.adapters.runtime.world_model must expose a "
        "runtime adapter class or instance; got module with no exports."
    )


def test_dreamdojo_registry_entry_advertises_world_model_rollout_and_dreamdojo(
    registry_module,
):
    """Sanity guard against a regression in the seeded registry. The
    DreamDojo placeholder entry must advertise both
    ``world_model_rollout`` and ``dreamdojo`` so dispatch can resolve.
    """

    from verl_post_training.registry.schemas import RuntimeBackend, TaskType

    entry = registry_module.get_model_entry(DREAMDOJO_PLACEHOLDER_MODEL_ID)
    assert TaskType.WORLD_MODEL_ROLLOUT in entry.supported_task_types, (
        f"{DREAMDOJO_PLACEHOLDER_MODEL_ID!r} must advertise "
        f"world_model_rollout; got {entry.supported_task_types!r}"
    )
    assert RuntimeBackend.DREAMDOJO in entry.runtime_backends, (
        f"{DREAMDOJO_PLACEHOLDER_MODEL_ID!r} must advertise dreamdojo "
        f"runtime backend; got {entry.runtime_backends!r}"
    )
