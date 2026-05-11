"""M1 acceptance: the repo-owned model registry covers the four required model
families and exposes the full ``ModelRegistryEntry`` field contract for each
seeded or placeholder entry.

This test file pins three of the acceptance criteria for M1:

1. Importing the registry returns at least the ``vlm_chat``, ``video_encoder``,
   ``video_generator``, and ``world_model`` families.
2. At least one seeded Qwen chat entry declares every required field on the
   ``ModelRegistryEntry`` contract with the expected non-empty types.
3. Placeholder / fixture entries for ``video_encoder``, ``video_generator``,
   and ``world_model`` satisfy the same field-level contract — no required
   field is omitted just because a model family is not yet integrated.

The fourth and fifth acceptance criteria are pinned in the sibling test files
``test_registry_lookup_errors.py`` and ``test_registry_schema_enums.py``.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Contract definitions
# ---------------------------------------------------------------------------

#: Fields the approved plan mandates on every ``ModelRegistryEntry``.
REQUIRED_FIELDS: tuple[str, ...] = (
    "model_id",
    "model_family",
    "supported_task_types",
    "trainer_backends",
    "runtime_backends",
    "checkpoint_source",
    "checkpoint_format",
    "required_modalities",
    "dataset_adapter_keys",
    "launcher_type",
    "default_precision",
    "distributed_requirements",
    "environment_tags",
)

#: Fields that must be non-empty *tuples* (or, more permissively, ordered
#: sequences). Tuples are preferred so registry entries are hashable and
#: immutable in downstream call sites.
NON_EMPTY_SEQUENCE_FIELDS: tuple[str, ...] = (
    "supported_task_types",
    "trainer_backends",
    "runtime_backends",
    "required_modalities",
    "dataset_adapter_keys",
    "environment_tags",
)

#: Fields that must be non-empty strings.
NON_EMPTY_STRING_FIELDS: tuple[str, ...] = (
    "checkpoint_source",
    "checkpoint_format",
    "launcher_type",
    "default_precision",
)

#: Families that must appear in the registry per the M1 acceptance criteria.
REQUIRED_FAMILIES: frozenset[str] = frozenset(
    {"vlm_chat", "video_encoder", "video_generator", "world_model"}
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry_module():
    import verl_post_training.registry.model_registry as registry

    return registry


@pytest.fixture(scope="module")
def schemas_module():
    import verl_post_training.registry.schemas as schemas

    return schemas


def _family_value(family) -> str:
    """Extract the canonical string id for a model family.

    Accepts either an enum member (with ``.value``) or a plain string so the
    test does not over-constrain the implementation choice.
    """

    value = getattr(family, "value", family)
    assert isinstance(value, str), (
        f"model_family must be a string identifier, got {family!r}"
    )
    return value


def _resolve_lookup(registry_module):
    """Return a callable ``(model_id) -> entry`` for the registry.

    The plan does not prescribe the exact API name, so we accept any of a few
    natural conventions. Tests fail explicitly if none is exported.
    """

    for attr in ("get_model_entry", "get_entry", "lookup", "resolve"):
        fn = getattr(registry_module, attr, None)
        if callable(fn):
            return fn
    pytest.fail(
        "registry.model_registry must expose a lookup callable named one of: "
        "get_model_entry, get_entry, lookup, resolve."
    )


def _resolve_iter_entries(registry_module):
    """Return an iterable of all registered ``ModelRegistryEntry`` values.

    Accepts ``iter_entries`` / ``all_entries`` / a module-level ``REGISTRY``
    mapping or sequence so the writer is not forced into a single API shape.
    """

    for attr in ("iter_entries", "all_entries", "entries"):
        fn = getattr(registry_module, attr, None)
        if callable(fn):
            return list(fn())
    registry_obj = getattr(registry_module, "REGISTRY", None)
    if registry_obj is None:
        registry_obj = getattr(registry_module, "MODEL_REGISTRY", None)
    if registry_obj is not None:
        if hasattr(registry_obj, "values"):
            return list(registry_obj.values())
        return list(registry_obj)
    pytest.fail(
        "registry.model_registry must expose either an iterable callable "
        "(iter_entries / all_entries / entries) or a REGISTRY mapping."
    )


# ---------------------------------------------------------------------------
# Criterion 1: required model families are present
# ---------------------------------------------------------------------------


def test_registry_exposes_all_required_families(registry_module):
    """The four families named in the approved plan must all be reachable."""

    entries = _resolve_iter_entries(registry_module)
    assert entries, "registry must contain at least one entry"

    families_present = {_family_value(entry.model_family) for entry in entries}
    missing = REQUIRED_FAMILIES - families_present
    assert not missing, (
        f"registry is missing required model families: {sorted(missing)}; "
        f"got {sorted(families_present)}"
    )


# ---------------------------------------------------------------------------
# Criterion 2: at least one Qwen chat entry exposes every required field
# ---------------------------------------------------------------------------


def _select_vlm_chat_entries(registry_module):
    return [
        entry
        for entry in _resolve_iter_entries(registry_module)
        if _family_value(entry.model_family) == "vlm_chat"
    ]


def test_at_least_one_qwen_chat_entry_is_seeded(registry_module):
    chat_entries = _select_vlm_chat_entries(registry_module)
    assert chat_entries, (
        "registry must seed at least one vlm_chat entry to preserve the "
        "current Qwen workflow"
    )
    qwen_entries = [
        entry for entry in chat_entries if "qwen" in entry.model_id.lower()
    ]
    assert qwen_entries, (
        "at least one seeded vlm_chat entry must represent the existing Qwen "
        "path; got model_ids: "
        f"{[entry.model_id for entry in chat_entries]}"
    )


@pytest.mark.parametrize("field_name", REQUIRED_FIELDS)
def test_qwen_chat_entry_declares_required_field(registry_module, field_name):
    chat_entries = _select_vlm_chat_entries(registry_module)
    assert chat_entries, "vlm_chat must be seeded before per-field assertions"
    qwen_entries = [
        entry for entry in chat_entries if "qwen" in entry.model_id.lower()
    ]
    assert qwen_entries, "Qwen chat entry must be seeded"
    entry = qwen_entries[0]
    assert hasattr(entry, field_name), (
        f"Qwen vlm_chat entry is missing required field {field_name!r}; "
        f"available attributes: {dir(entry)}"
    )


def test_qwen_chat_entry_field_types(registry_module):
    chat_entries = _select_vlm_chat_entries(registry_module)
    qwen_entries = [
        entry for entry in chat_entries if "qwen" in entry.model_id.lower()
    ]
    assert qwen_entries, "Qwen chat entry must be seeded"
    entry = qwen_entries[0]

    # Non-empty tuple-like fields
    for field_name in NON_EMPTY_SEQUENCE_FIELDS:
        value = getattr(entry, field_name)
        assert isinstance(value, tuple), (
            f"Qwen entry field {field_name!r} must be a tuple (got "
            f"{type(value).__name__}); the approved plan requires hashable "
            "immutable sequences so registry entries can be cached and keyed."
        )
        assert len(value) > 0, (
            f"Qwen entry field {field_name!r} must be non-empty"
        )

    # Non-empty string fields
    for field_name in NON_EMPTY_STRING_FIELDS:
        value = getattr(entry, field_name)
        assert isinstance(value, str), (
            f"Qwen entry field {field_name!r} must be a string (got "
            f"{type(value).__name__})"
        )
        assert value.strip(), (
            f"Qwen entry field {field_name!r} must be a non-empty string"
        )

    # distributed_requirements: dict, may be empty
    distributed = entry.distributed_requirements
    assert isinstance(distributed, dict), (
        "Qwen entry field 'distributed_requirements' must be a dict (got "
        f"{type(distributed).__name__}); the approved plan allows it to be "
        "empty but it must always be a mapping."
    )


def test_qwen_chat_entry_uses_chat_backends(registry_module, schemas_module):
    """The seeded Qwen entry must route to the chat backends declared by the
    M1 plan: LLaMA-Factory for SFT, VERL for RL, and the OpenAI-compatible
    vLLM runtime for serving. This pins criterion 2 against backend drift.
    """

    chat_entries = _select_vlm_chat_entries(registry_module)
    qwen_entries = [
        entry for entry in chat_entries if "qwen" in entry.model_id.lower()
    ]
    assert qwen_entries, "Qwen chat entry must be seeded"
    entry = qwen_entries[0]

    trainer_values = {_family_value(b) for b in entry.trainer_backends}
    runtime_values = {_family_value(b) for b in entry.runtime_backends}
    task_values = {_family_value(t) for t in entry.supported_task_types}

    assert {"llamafactory", "verl"}.issubset(trainer_values), (
        "Qwen entry must route SFT through LLaMA-Factory and RL through VERL "
        "per the approved plan; got trainer_backends="
        f"{sorted(trainer_values)}"
    )
    assert "openai_chat_vllm" in runtime_values, (
        "Qwen entry must declare openai_chat_vllm as a supported runtime "
        f"backend; got runtime_backends={sorted(runtime_values)}"
    )
    assert {"chat_sft", "chat_rl"}.issubset(task_values), (
        "Qwen entry must support both chat_sft and chat_rl task types; got "
        f"supported_task_types={sorted(task_values)}"
    )


# ---------------------------------------------------------------------------
# Criterion 3: placeholder entries for the non-chat families honor the same
#              field-level contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family_name", sorted({"video_encoder", "video_generator", "world_model"})
)
def test_placeholder_family_has_at_least_one_entry(registry_module, family_name):
    entries = [
        entry
        for entry in _resolve_iter_entries(registry_module)
        if _family_value(entry.model_family) == family_name
    ]
    assert entries, (
        f"registry must contain at least one placeholder or fixture entry "
        f"for the {family_name!r} family"
    )


@pytest.mark.parametrize(
    "family_name", sorted({"video_encoder", "video_generator", "world_model"})
)
def test_placeholder_entry_satisfies_field_contract(registry_module, family_name):
    entries = [
        entry
        for entry in _resolve_iter_entries(registry_module)
        if _family_value(entry.model_family) == family_name
    ]
    assert entries, f"missing placeholder entry for {family_name}"
    entry = entries[0]

    for field_name in REQUIRED_FIELDS:
        assert hasattr(entry, field_name), (
            f"placeholder entry for {family_name!r} is missing required "
            f"field {field_name!r}; the approved plan forbids omitting "
            "required fields just because integration is incomplete."
        )

    for field_name in NON_EMPTY_SEQUENCE_FIELDS:
        value = getattr(entry, field_name)
        assert isinstance(value, tuple), (
            f"{family_name} placeholder field {field_name!r} must be a "
            f"tuple, got {type(value).__name__}"
        )
        assert len(value) > 0, (
            f"{family_name} placeholder field {field_name!r} must be "
            "non-empty; placeholder entries should declare intended "
            "task/backend keys so dispatch errors are descriptive."
        )

    for field_name in NON_EMPTY_STRING_FIELDS:
        value = getattr(entry, field_name)
        assert isinstance(value, str) and value.strip(), (
            f"{family_name} placeholder field {field_name!r} must be a "
            f"non-empty string, got {value!r}"
        )

    assert isinstance(entry.distributed_requirements, dict), (
        f"{family_name} placeholder field 'distributed_requirements' must be "
        f"a dict, got {type(entry.distributed_requirements).__name__}"
    )


def test_placeholder_entries_use_distinct_model_ids(registry_module):
    """Each placeholder family should have its own model_id so dispatch
    can target it deterministically rather than collapsing to a single shared
    stub.
    """

    entries = _resolve_iter_entries(registry_module)
    ids = [entry.model_id for entry in entries]
    assert len(ids) == len(set(ids)), (
        f"model_id values must be unique across registry entries; got {ids}"
    )
