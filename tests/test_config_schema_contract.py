"""M2 acceptance: the repo-level config loader accepts a valid YAML config
that declares every required top-level field, exposes a validated in-memory
structure to callers, and rejects unknown enum values *before* any backend
code is invoked.

This file pins two of the four M2 acceptance criteria:

1. *Valid YAML with top-level dispatch fields loads into a validated
   in-memory config object.*

   Required fields per the approved plan and design doc (section 7.5):
   ``task_type``, ``model_id``, ``dataset_adapter``, ``input_manifest``,
   ``output_dir``, ``launcher``, ``resources``, and either
   ``trainer_backend`` or ``runtime_backend``.

2. *Unknown enum values are rejected before any backend code runs.*

   The dispatch entry point must not import or invoke a backend (vLLM,
   LLaMA-Factory, VERL, V-JEPA2, ...) when the supplied ``task_type``,
   ``trainer_backend``, or ``runtime_backend`` is not in the approved enum
   set. Validation must fail at the loader layer.

The two other M2 acceptance criteria live in the sibling test files
``test_dispatch_compatibility_errors.py`` and
``test_backend_config_passthrough.py``.

These tests use ``tmp_path`` and ``monkeypatch`` to remain deterministic and
self-contained; they do not touch the network, the GPU, or any
backend-owned subprocess.
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
def load_config_module():
    """Import the M2 loader module.

    Plain import: a missing M2 implementation must surface as a *failure*
    here so the two-agent loop sees the red signal.
    """

    import verl_post_training.launch.load_config as module

    return module


def _resolve_loader(load_config_module):
    """Return the YAML-path-to-config-object loader.

    The plan does not pin the exact name, so we accept any of a few natural
    spellings. The first match wins.
    """

    for attr in ("load_config", "load", "load_task_config", "from_yaml"):
        fn = getattr(load_config_module, attr, None)
        if callable(fn):
            return fn
    pytest.fail(
        "verl_post_training.launch.load_config must expose a callable named "
        "one of: load_config, load, load_task_config, from_yaml."
    )


# ---------------------------------------------------------------------------
# Canonical valid configs
# ---------------------------------------------------------------------------


def _valid_trainer_config() -> dict[str, Any]:
    """A complete, syntactically valid trainer-side config.

    Mirrors the YAML shape approved in design doc section 7.5 but reuses the
    seeded Qwen entry from the M1 registry so the loader can resolve it
    without inventing a new model entry.
    """

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
        "backend_config": {
            "config_file": "SFT/configs/qwen_sft.yaml",
            "extra_args": [],
        },
    }


def _valid_runtime_config() -> dict[str, Any]:
    """A complete, syntactically valid runtime-side config.

    Uses ``runtime_backend`` instead of ``trainer_backend`` to pin the
    acceptance criterion that *either* field satisfies the schema.
    """

    return {
        "task_type": "chat_rl",
        "model_id": "qwen3-vl-4b-instruct",
        "runtime_backend": "openai_chat_vllm",
        "dataset_adapter": "chat_rl",
        "input_manifest": "data/pipeline/rl_manifest.jsonl",
        "output_dir": "outputs/post_training/qwen_rl_run_001",
        "launcher": {
            "kind": "python_module",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {
            "precision": "bf16",
            "devices": 1,
        },
        "backend_config": {},
    }


def _write_yaml(tmp_path: Path, payload: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path


# ---------------------------------------------------------------------------
# Criterion 1: a valid YAML loads into a validated in-memory structure
# ---------------------------------------------------------------------------


def _coerce_value(value: Any) -> Any:
    """Strip enum / Path wrappers so equality assertions read naturally."""

    if hasattr(value, "value"):  # enum-like
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _get_field(config_obj: Any, name: str) -> Any:
    """Read a top-level field off the loaded config object.

    Accepts dataclass-style attribute access *or* dict-style indexing so the
    writer can pick whichever shape they prefer.
    """

    if hasattr(config_obj, name):
        return getattr(config_obj, name)
    if isinstance(config_obj, dict):
        return config_obj.get(name)
    pytest.fail(
        f"loaded config object does not expose top-level field {name!r}; "
        f"got type {type(config_obj).__name__}"
    )


def test_loader_accepts_valid_trainer_config(load_config_module, tmp_path):
    loader = _resolve_loader(load_config_module)
    payload = _valid_trainer_config()
    path = _write_yaml(tmp_path, payload)

    config = loader(path)

    assert config is not None, "loader returned None for a valid config"
    assert _coerce_value(_get_field(config, "task_type")) == "chat_sft"
    assert _get_field(config, "model_id") == "qwen3-vl-4b-instruct"
    assert _coerce_value(_get_field(config, "trainer_backend")) == "llamafactory"


def test_loader_accepts_valid_runtime_config(load_config_module, tmp_path):
    loader = _resolve_loader(load_config_module)
    payload = _valid_runtime_config()
    path = _write_yaml(tmp_path, payload)

    config = loader(path)

    assert config is not None, "loader returned None for a valid config"
    assert _coerce_value(_get_field(config, "task_type")) == "chat_rl"
    assert _coerce_value(_get_field(config, "runtime_backend")) == "openai_chat_vllm"


@pytest.mark.parametrize(
    "field_name",
    [
        "task_type",
        "model_id",
        "dataset_adapter",
        "input_manifest",
        "output_dir",
        "launcher",
        "resources",
    ],
)
def test_loaded_config_exposes_required_top_level_fields(
    load_config_module, tmp_path, field_name
):
    """Each of the seven non-conditional top-level fields must be reachable
    on the validated in-memory structure.

    ``trainer_backend``/``runtime_backend`` are tested separately because the
    plan permits *either* one (but not neither) — those are pinned in
    ``test_either_trainer_or_runtime_backend_is_required``.
    """

    loader = _resolve_loader(load_config_module)
    path = _write_yaml(tmp_path, _valid_trainer_config())
    config = loader(path)

    if hasattr(config, field_name):
        value = getattr(config, field_name)
    elif isinstance(config, dict):
        assert field_name in config, (
            f"loaded config dict is missing required field {field_name!r}; "
            f"got keys {sorted(config)}"
        )
        value = config[field_name]
    else:
        pytest.fail(
            f"loaded config object does not expose required field "
            f"{field_name!r}; got type {type(config).__name__}"
        )

    assert value not in (None, "", [], {}), (
        f"loaded config field {field_name!r} must preserve the value from "
        f"the YAML; got {value!r}"
    )


@pytest.mark.parametrize(
    "backend_field,backend_value",
    [
        ("trainer_backend", "llamafactory"),
        ("runtime_backend", "openai_chat_vllm"),
    ],
)
def test_either_trainer_or_runtime_backend_is_accepted(
    load_config_module, tmp_path, backend_field, backend_value
):
    loader = _resolve_loader(load_config_module)
    payload = _valid_trainer_config()
    payload.pop("trainer_backend", None)
    payload.pop("runtime_backend", None)
    payload[backend_field] = backend_value
    path = _write_yaml(tmp_path, payload)

    config = loader(path)
    assert config is not None, (
        f"loader rejected an otherwise-valid config that declared "
        f"{backend_field}={backend_value!r}"
    )
    assert _coerce_value(_get_field(config, backend_field)) == backend_value


def test_loader_rejects_config_missing_both_backend_fields(
    load_config_module, tmp_path
):
    """The plan requires *either* ``trainer_backend`` or ``runtime_backend`` —
    declaring neither must be a validation error.
    """

    loader = _resolve_loader(load_config_module)
    payload = _valid_trainer_config()
    payload.pop("trainer_backend", None)
    payload.pop("runtime_backend", None)
    path = _write_yaml(tmp_path, payload)

    with pytest.raises(Exception) as excinfo:
        loader(path)

    assert type(excinfo.value) is not Exception, (
        "loader raised a bare Exception; callers cannot distinguish missing-"
        "backend from unrelated bugs. Use a typed validation error."
    )


@pytest.mark.parametrize(
    "missing_field",
    [
        "task_type",
        "model_id",
        "dataset_adapter",
        "input_manifest",
        "output_dir",
        "launcher",
        "resources",
    ],
)
def test_loader_rejects_config_missing_required_field(
    load_config_module, tmp_path, missing_field
):
    loader = _resolve_loader(load_config_module)
    payload = _valid_trainer_config()
    payload.pop(missing_field)
    path = _write_yaml(tmp_path, payload)

    with pytest.raises(Exception) as excinfo:
        loader(path)

    msg = str(excinfo.value)
    assert missing_field in msg, (
        f"validation error must reference the missing field name "
        f"{missing_field!r}; got: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 2: unknown enum values are rejected before backend code runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("task_type", "totally_unknown_task"),
        ("trainer_backend", "totally_unknown_trainer"),
    ],
)
def test_unknown_enum_value_is_rejected(
    load_config_module, tmp_path, field_name, bad_value
):
    loader = _resolve_loader(load_config_module)
    payload = _valid_trainer_config()
    payload[field_name] = bad_value
    path = _write_yaml(tmp_path, payload)

    with pytest.raises(Exception) as excinfo:
        loader(path)

    msg = str(excinfo.value)
    assert bad_value in msg or field_name in msg, (
        "validation error must reference the offending field name or value; "
        f"got: {msg!r}"
    )
    assert type(excinfo.value) is not Exception, (
        "loader raised a bare Exception for an unknown enum value; use a "
        "typed validation error so dispatch callers can distinguish enum "
        "violations from unrelated bugs."
    )


def test_unknown_runtime_backend_is_rejected(load_config_module, tmp_path):
    """Pinned separately because ``runtime_backend`` is only present in the
    runtime-style config — the parametrized test above exercises the trainer
    config path.
    """

    loader = _resolve_loader(load_config_module)
    payload = _valid_runtime_config()
    payload["runtime_backend"] = "totally_unknown_runtime"
    path = _write_yaml(tmp_path, payload)

    with pytest.raises(Exception):
        loader(path)


def test_unknown_enum_value_rejection_does_not_import_backend(
    load_config_module, tmp_path, monkeypatch
):
    """Validation must reject unknown enum values *before* any backend code
    is invoked. We assert this by making the backend modules unimportable
    (set to ``None`` in ``sys.modules``) and confirming the loader still
    fails — but with a validation error, not an ImportError.
    """

    blocked_modules = (
        "vllm",
        "llamafactory",
        "verl",
        "torch",
    )
    for name in blocked_modules:
        monkeypatch.setitem(sys.modules, name, None)

    loader = _resolve_loader(load_config_module)
    payload = _valid_trainer_config()
    payload["task_type"] = "totally_unknown_task"
    path = _write_yaml(tmp_path, payload)

    with pytest.raises(Exception) as excinfo:
        loader(path)

    # If the loader had pulled in a backend before validating, we'd see an
    # ImportError from one of the blocked modules. A clean validation error
    # proves the rejection happens before any backend import.
    assert not isinstance(excinfo.value, ImportError), (
        "loader imported a backend module before rejecting an unknown enum "
        f"value; got ImportError: {excinfo.value!r}"
    )


def test_loader_is_idempotent_on_valid_input(load_config_module, tmp_path):
    """Loading the same valid YAML twice should not mutate the file or
    introduce hidden state — important so config validation is deterministic
    in CI.
    """

    loader = _resolve_loader(load_config_module)
    payload = _valid_trainer_config()
    path = _write_yaml(tmp_path, payload)
    before = path.read_text()

    first = loader(path)
    second = loader(path)

    after = path.read_text()
    assert before == after, "loader mutated the YAML source file"
    assert _coerce_value(_get_field(first, "task_type")) == _coerce_value(
        _get_field(second, "task_type")
    )
    assert _get_field(first, "model_id") == _get_field(second, "model_id")
