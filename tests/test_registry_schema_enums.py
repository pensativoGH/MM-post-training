"""M1 acceptance: enum values for `model_family`, `task_type`,
`trainer_backend`, and `runtime_backend` are importable from repo-owned schema
modules.

These tests pin the contract that other milestones depend on. They do not
constrain how the enums are spelled internally (e.g. `Enum` vs `StrEnum`), only
that:

- the four enum types live in `verl_post_training.registry.schemas`
- each enum exposes the canonical lowercase string values used elsewhere in
  the approved plan (so callers can do `ModelFamily("vlm_chat")`)
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def schemas_module():
    # Plain import so a missing M1 implementation surfaces as a *failure*,
    # not a skipped test. This is the red signal the two-agent loop relies on.
    import verl_post_training.registry.schemas as schemas

    return schemas


def _enum_string_values(enum_cls) -> set[str]:
    """Return the set of canonical string values an enum exposes.

    Accepts either ``Enum`` members whose ``.value`` is already a string or
    ``StrEnum`` members. Raises ``AssertionError`` if a member's value cannot
    be interpreted as the canonical string identifier.
    """

    values: set[str] = set()
    for member in enum_cls:
        value = getattr(member, "value", None)
        if isinstance(value, str):
            values.add(value)
        elif isinstance(member, str):
            values.add(str(member))
        else:
            raise AssertionError(
                f"Enum member {member!r} of {enum_cls!r} does not expose a "
                "string identifier; downstream milestones cannot key on it."
            )
    return values


def test_model_family_enum_is_importable(schemas_module):
    assert hasattr(schemas_module, "ModelFamily"), (
        "schemas must export a ModelFamily enum"
    )


def test_task_type_enum_is_importable(schemas_module):
    assert hasattr(schemas_module, "TaskType"), (
        "schemas must export a TaskType enum"
    )


def test_trainer_backend_enum_is_importable(schemas_module):
    assert hasattr(schemas_module, "TrainerBackend"), (
        "schemas must export a TrainerBackend enum"
    )


def test_runtime_backend_enum_is_importable(schemas_module):
    assert hasattr(schemas_module, "RuntimeBackend"), (
        "schemas must export a RuntimeBackend enum"
    )


def test_model_family_enum_covers_required_families(schemas_module):
    values = _enum_string_values(schemas_module.ModelFamily)
    required = {"vlm_chat", "video_encoder", "video_generator", "world_model"}
    missing = required - values
    assert not missing, (
        f"ModelFamily enum is missing required values: {sorted(missing)}; "
        f"got {sorted(values)}"
    )


def test_model_family_enum_round_trips_from_string(schemas_module):
    family = schemas_module.ModelFamily("vlm_chat")
    assert getattr(family, "value", family) == "vlm_chat"


def test_task_type_enum_covers_chat_baselines(schemas_module):
    values = _enum_string_values(schemas_module.TaskType)
    required = {"chat_sft", "chat_rl"}
    missing = required - values
    assert not missing, (
        f"TaskType enum is missing chat baseline values needed by the seeded "
        f"Qwen entry: {sorted(missing)}; got {sorted(values)}"
    )


def test_trainer_backend_enum_covers_chat_baselines(schemas_module):
    values = _enum_string_values(schemas_module.TrainerBackend)
    required = {"llamafactory", "verl"}
    missing = required - values
    assert not missing, (
        f"TrainerBackend enum is missing chat baseline values needed by the "
        f"seeded Qwen entry: {sorted(missing)}; got {sorted(values)}"
    )


def test_runtime_backend_enum_covers_chat_baseline(schemas_module):
    values = _enum_string_values(schemas_module.RuntimeBackend)
    required = {"openai_chat_vllm"}
    missing = required - values
    assert not missing, (
        f"RuntimeBackend enum is missing the chat baseline value needed by "
        f"the seeded Qwen entry: {sorted(missing)}; got {sorted(values)}"
    )


def test_enums_are_importable_without_backend_imports(monkeypatch):
    """Schemas must not pull in backend-specific code at import time.

    Other milestones import these enums in light orchestration contexts that
    must not require VERL, LLaMA-Factory, vLLM, etc. to be installed. We
    detect a regression here by making those modules unimportable and
    confirming the schema module still loads cleanly.
    """

    blocked = ("verl", "llamafactory", "vllm", "torch")
    import sys

    for name in blocked:
        monkeypatch.setitem(sys.modules, name, None)

    # Force a fresh import so the import hook above takes effect even if a
    # previous test cached the module.
    for name in list(sys.modules):
        if name.startswith("verl_post_training.registry.schemas"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    import importlib

    module = importlib.import_module("verl_post_training.registry.schemas")
    assert hasattr(module, "ModelFamily")
    assert hasattr(module, "TaskType")
    assert hasattr(module, "TrainerBackend")
    assert hasattr(module, "RuntimeBackend")
