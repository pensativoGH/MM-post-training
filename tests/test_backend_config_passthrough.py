"""M2 acceptance: ``backend_config`` is preserved verbatim as
backend-specific passthrough data and is never promoted into top-level
dispatch logic.

This file pins the third M2 acceptance criterion (quoted from the approved
plan):

    ``backend_config`` keys are preserved and passed through without being
    promoted into top-level dispatch logic

Operationally, that means three things:

1. **Preserved verbatim.** Whatever the user wrote under ``backend_config``
   in YAML — including arbitrary scalar / list / nested-dict values, and
   keys that share names with normalized top-level fields — survives the
   load+dispatch round trip unchanged.

2. **Not promoted.** A key that appears *only* under ``backend_config``
   does not implicitly become a top-level field. For example,
   ``backend_config.task_type: chat_rl`` must not silently override the
   real top-level ``task_type``.

3. **Not validated.** Unknown / arbitrary keys under ``backend_config`` do
   not trigger schema-validation errors at the top-level config layer.
   Backend-specific validation, if any, belongs to the backend.

Tests use ``tmp_path`` so each invocation is hermetic and do not touch the
network, the GPU, or any backend subprocess.
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
def load_config_module():
    import verl_post_training.launch.load_config as module

    return module


@pytest.fixture(scope="module")
def dispatch_module():
    import verl_post_training.launch.dispatch as module

    return module


def _resolve_loader(load_config_module):
    for attr in ("load_config", "load", "load_task_config", "from_yaml"):
        fn = getattr(load_config_module, attr, None)
        if callable(fn):
            return fn
    pytest.fail(
        "verl_post_training.launch.load_config must expose a callable named "
        "one of: load_config, load, load_task_config, from_yaml."
    )


def _resolve_dispatcher(dispatch_module):
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


def _get_field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def _coerce_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _extract_backend_config(obj: Any) -> Any:
    """Pull ``backend_config`` off either the loaded config object *or* the
    dispatch plan, with light shape-tolerance for nested wrappers.
    """

    if obj is None:
        return None
    direct = _get_field(obj, "backend_config")
    if direct is not None:
        return direct
    # Some plan shapes nest the user-facing config under ``config``,
    # ``task_config``, or ``request``. Try those before giving up.
    for nested in ("config", "task_config", "request"):
        inner = _get_field(obj, nested)
        if inner is not None:
            via_inner = _get_field(inner, "backend_config")
            if via_inner is not None:
                return via_inner
    return None


def _base_qwen_config() -> dict[str, Any]:
    return {
        "task_type": "chat_sft",
        "model_id": "qwen3-vl-4b-instruct",
        "trainer_backend": "llamafactory",
        "dataset_adapter": "chat_sft",
        "input_manifest": "data/pipeline/train_manifest.jsonl",
        "output_dir": "outputs/post_training/qwen_passthrough_run",
        "launcher": {
            "kind": "torchrun",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }


# ---------------------------------------------------------------------------
# Criterion 3: backend_config is preserved verbatim and not promoted
# ---------------------------------------------------------------------------


def test_backend_config_round_trips_through_loader(load_config_module, tmp_path):
    loader = _resolve_loader(load_config_module)
    payload = _base_qwen_config()
    payload["backend_config"] = {
        "config_file": "SFT/configs/qwen_sft.yaml",
        "extra_args": ["--seed", "42"],
        "nested": {
            "deepspeed_stage": 2,
            "gradient_checkpointing": True,
        },
        "scalar_int": 7,
        "scalar_float": 3.14,
        "scalar_bool": False,
        "scalar_null": None,
    }
    path = _write_yaml(tmp_path, payload)

    config = loader(path)
    backend_config = _extract_backend_config(config)

    assert backend_config is not None, (
        "loader dropped backend_config; the M2 contract requires it to "
        "survive as opaque passthrough data."
    )

    # Tolerate dataclass-wrapped backend_config objects by mapping them back
    # to a dict for comparison.
    if not isinstance(backend_config, dict) and hasattr(backend_config, "__dict__"):
        backend_config = dict(backend_config.__dict__)

    assert backend_config == payload["backend_config"], (
        "backend_config did not round-trip through the loader unchanged; "
        f"expected {payload['backend_config']!r}, got {backend_config!r}"
    )


def test_backend_config_round_trips_through_dispatch(
    load_config_module, dispatch_module, tmp_path
):
    loader = _resolve_loader(load_config_module)
    dispatcher = _resolve_dispatcher(dispatch_module)
    payload = _base_qwen_config()
    payload["backend_config"] = {
        "config_file": "SFT/configs/qwen_sft.yaml",
        "extra_args": ["--lr", "1e-5"],
    }
    plan = dispatcher(loader(_write_yaml(tmp_path, payload)))

    backend_config = _extract_backend_config(plan)
    assert backend_config is not None, (
        "dispatch plan dropped backend_config; backends downstream of "
        "dispatch must still be able to read it."
    )
    if not isinstance(backend_config, dict) and hasattr(backend_config, "__dict__"):
        backend_config = dict(backend_config.__dict__)
    assert backend_config == payload["backend_config"], (
        "backend_config did not survive dispatch unchanged; "
        f"expected {payload['backend_config']!r}, got {backend_config!r}"
    )


def test_unknown_backend_config_keys_are_accepted(load_config_module, tmp_path):
    """The schema must not validate the *contents* of ``backend_config``.

    Unknown keys, including keys that look like enum values or top-level
    field names, must not trigger a validation error.
    """

    loader = _resolve_loader(load_config_module)
    payload = _base_qwen_config()
    payload["backend_config"] = {
        "totally_made_up_key": "totally_made_up_value",
        "another_key": 12345,
        "task_type": "chat_rl",  # shadow of a top-level field name
        "trainer_backend": "definitely_not_a_real_backend",
    }
    path = _write_yaml(tmp_path, payload)

    # Must not raise — the contract says backend_config is opaque to the
    # top-level schema.
    config = loader(path)
    assert config is not None


def test_backend_config_keys_do_not_shadow_top_level_fields(
    load_config_module, tmp_path
):
    """A ``backend_config.task_type`` value must not silently overwrite the
    real top-level ``task_type``.

    This is the regression the acceptance criterion's "without being
    promoted into top-level dispatch logic" clause guards against.
    """

    loader = _resolve_loader(load_config_module)
    payload = _base_qwen_config()
    payload["backend_config"] = {
        "task_type": "chat_rl",  # different from top-level value below
        "trainer_backend": "verl",  # different from top-level value below
        "model_id": "qwen3-vl-9000-fictional",
    }
    path = _write_yaml(tmp_path, payload)
    config = loader(path)

    assert _coerce_value(_get_field(config, "task_type")) == "chat_sft", (
        "top-level task_type must not be silently overridden by a "
        "backend_config.task_type key"
    )
    assert _coerce_value(_get_field(config, "trainer_backend")) == "llamafactory", (
        "top-level trainer_backend must not be silently overridden by a "
        "backend_config.trainer_backend key"
    )
    assert _get_field(config, "model_id") == "qwen3-vl-4b-instruct", (
        "top-level model_id must not be silently overridden by a "
        "backend_config.model_id key"
    )


def test_missing_backend_config_is_tolerated(load_config_module, tmp_path):
    """Omitting ``backend_config`` entirely should not break loading — the
    config contract names it as a passthrough field, not a hard requirement
    with mandatory contents. A loader is allowed either to default it to an
    empty mapping or to leave it as ``None``; both are acceptable as long as
    the load itself does not raise.
    """

    loader = _resolve_loader(load_config_module)
    payload = _base_qwen_config()
    payload.pop("backend_config", None)
    path = _write_yaml(tmp_path, payload)

    config = loader(path)
    assert config is not None

    backend_config = _extract_backend_config(config)
    # Empty dict or None both express "user supplied nothing". Neither must
    # cause a downstream crash for the load step itself.
    assert backend_config in (None, {}, {} or None), (
        "missing backend_config should default to an empty mapping or "
        f"remain None; got {backend_config!r}"
    )


def test_backend_config_is_isolated_from_other_configs(
    load_config_module, tmp_path
):
    """Loading two configs that differ only in ``backend_config`` must yield
    two independent backend_config payloads — no shared mutable default,
    no accidental aliasing, no leakage between loads.
    """

    loader = _resolve_loader(load_config_module)

    payload_a = _base_qwen_config()
    payload_a["backend_config"] = {"flag": "alpha"}
    payload_b = _base_qwen_config()
    payload_b["backend_config"] = {"flag": "beta"}

    path_a = tmp_path / "config_a.yaml"
    path_b = tmp_path / "config_b.yaml"
    path_a.write_text(yaml.safe_dump(payload_a, sort_keys=False))
    path_b.write_text(yaml.safe_dump(payload_b, sort_keys=False))

    config_a = loader(path_a)
    config_b = loader(path_b)

    bc_a = _extract_backend_config(config_a)
    bc_b = _extract_backend_config(config_b)

    if not isinstance(bc_a, dict) and hasattr(bc_a, "__dict__"):
        bc_a = dict(bc_a.__dict__)
    if not isinstance(bc_b, dict) and hasattr(bc_b, "__dict__"):
        bc_b = dict(bc_b.__dict__)

    assert bc_a == {"flag": "alpha"}, (
        f"backend_config for config_a did not round-trip; got {bc_a!r}"
    )
    assert bc_b == {"flag": "beta"}, (
        f"backend_config for config_b did not round-trip; got {bc_b!r}"
    )


def test_backend_config_does_not_appear_at_top_level(
    load_config_module, tmp_path
):
    """Sanity check: keys placed under ``backend_config`` must stay there
    and not be visible as top-level attributes on the loaded config.

    Otherwise the dispatcher could accidentally pick up backend-specific
    options when making cross-family decisions.
    """

    loader = _resolve_loader(load_config_module)
    payload = _base_qwen_config()
    payload["backend_config"] = {
        "deepspeed_stage": 3,
        "lora_target": "all-linear",
    }
    path = _write_yaml(tmp_path, payload)
    config = loader(path)

    for key in ("deepspeed_stage", "lora_target"):
        # If a top-level attribute with that name exists, it must not
        # silently mirror the backend_config value.
        if hasattr(config, key):
            assert getattr(config, key) != payload["backend_config"][key], (
                f"backend_config.{key} was promoted to a top-level attribute "
                "with the same value; this violates the M2 isolation rule."
            )
        if isinstance(config, dict):
            assert key not in config, (
                f"backend_config.{key} was promoted to a top-level dict "
                "key; this violates the M2 isolation rule."
            )
