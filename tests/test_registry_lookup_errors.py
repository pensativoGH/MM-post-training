"""M1 acceptance: requesting an unknown ``model_id`` must fail explicitly
instead of silently falling back to Qwen defaults.

This file pins the negative path of the registry contract. The positive path
(known model_ids resolve to the right entry) is covered in
``test_model_registry.py``.

Acceptance criterion (quoted from the approved plan):

    requesting an unknown ``model_id`` raises a typed error or returns a
    clearly named failure value; it must not silently fall back to Qwen
    defaults

We accept either of the two failure styles the plan allows (typed exception
*or* sentinel/None return value), but we reject any behavior that returns a
Qwen-shaped entry for an unknown id.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def registry_module():
    import verl_post_training.registry.model_registry as registry

    return registry


def _resolve_lookup(registry_module):
    for attr in ("get_model_entry", "get_entry", "lookup", "resolve"):
        fn = getattr(registry_module, attr, None)
        if callable(fn):
            return fn
    pytest.fail(
        "registry.model_registry must expose a lookup callable named one of: "
        "get_model_entry, get_entry, lookup, resolve."
    )


def _family_value(family) -> str:
    return getattr(family, "value", family)


# A model_id that must never be present in the registry. Chosen to be
# implausibly distinct from any real checkpoint name.
UNKNOWN_MODEL_ID = "definitely-not-a-real-model-id-xyz-9999"


def _looks_like_qwen_entry(value) -> bool:
    """Heuristic: does the returned value resemble the seeded Qwen entry?

    Used to catch the explicit anti-pattern the plan forbids: silently
    returning Qwen defaults when an unknown model_id is requested.
    """

    if value is None:
        return False
    model_id = getattr(value, "model_id", None)
    if isinstance(model_id, str) and "qwen" in model_id.lower():
        return True
    family = getattr(value, "model_family", None)
    if family is not None and _family_value(family) == "vlm_chat":
        # A vlm_chat entry returned for an unknown id is the exact regression
        # the acceptance criterion forbids.
        return True
    return False


def test_unknown_model_id_does_not_return_qwen_defaults(registry_module):
    """The single most important assertion in this file.

    If the lookup raises, that satisfies the criterion (the next two tests
    pin the shape of the exception). If it returns a value, that value must
    not be a Qwen-shaped entry.
    """

    lookup = _resolve_lookup(registry_module)
    try:
        result = lookup(UNKNOWN_MODEL_ID)
    except Exception:
        # An exception is an acceptable failure mode; details checked in
        # `test_unknown_model_id_raises_typed_error`.
        return

    assert not _looks_like_qwen_entry(result), (
        "registry lookup must not silently fall back to a Qwen-shaped entry "
        f"for an unknown model_id; got {result!r}"
    )


def test_unknown_model_id_signals_failure(registry_module):
    """The lookup must signal failure either by raising or by returning a
    sentinel/None — never by returning a populated entry.
    """

    lookup = _resolve_lookup(registry_module)
    raised: BaseException | None = None
    try:
        result = lookup(UNKNOWN_MODEL_ID)
    except BaseException as exc:  # noqa: BLE001 -- test wants any exception
        raised = exc
        result = None

    if raised is not None:
        # Failure mode 1: raised exception. Validated by the next test.
        return

    # Failure mode 2: returned a clearly-named failure value.
    if result is None:
        return
    # If a non-None object is returned, it must clearly indicate failure —
    # e.g. a NamedTuple/dataclass with `.found is False`, or a "not-found"
    # sentinel. We accept either of the following conventions.
    found_flag = getattr(result, "found", None)
    if found_flag is False:
        return
    if isinstance(result, bool) and result is False:
        return

    pytest.fail(
        "registry lookup of an unknown model_id must either raise or return a "
        f"clearly-named failure value; got {result!r}"
    )


def test_unknown_model_id_raises_typed_error(registry_module):
    """When the lookup raises (the preferred style per the plan), the
    exception must be a registry-defined error type — not a bare KeyError or
    a generic Exception that downstream callers cannot distinguish from
    unrelated bugs.

    Accepts any of:
      - a custom exception class exported by the registry module (e.g.
        ``ModelNotFoundError``, ``UnknownModelIdError``, ``RegistryLookupError``)
      - ``LookupError`` (or a subclass like ``KeyError``) wrapped so the
        message clearly references the unknown model_id

    Skips cleanly if the implementation chose the sentinel-return style.
    """

    lookup = _resolve_lookup(registry_module)
    try:
        lookup(UNKNOWN_MODEL_ID)
    except BaseException as exc:
        # Generic ``Exception`` (i.e. ``type(exc) is Exception``) is not
        # specific enough; we want a typed error.
        assert type(exc) is not Exception, (
            "registry lookup raised a bare Exception; downstream callers "
            "cannot distinguish 'unknown model' from unrelated runtime bugs. "
            "Raise a typed error (e.g. LookupError subclass or a registry-"
            "owned exception class) instead."
        )

        # The exception message must reference the unknown model_id so
        # logs and error surfaces are debuggable.
        assert UNKNOWN_MODEL_ID in str(exc), (
            "registry lookup error message must include the unknown "
            f"model_id {UNKNOWN_MODEL_ID!r}; got: {exc!r}"
        )

        # Prefer either LookupError (covers KeyError) or a registry-owned
        # exception class. We test "registry-owned" loosely: defined in the
        # same package as the lookup.
        if isinstance(exc, LookupError):
            return
        owning_pkg = type(exc).__module__.split(".")[0]
        assert owning_pkg == "verl_post_training", (
            "registry lookup raised a third-party exception type "
            f"({type(exc).__module__}.{type(exc).__name__}); raise either a "
            "LookupError subclass or a verl_post_training-owned exception."
        )
        return

    # No exception raised -> the lookup is using the sentinel-return style,
    # which is also acceptable. The previous test ensures that style does not
    # silently return Qwen defaults.
    pytest.skip(
        "registry lookup returned a value instead of raising; sentinel-style "
        "failure is allowed and is validated by "
        "test_unknown_model_id_signals_failure"
    )


def test_known_qwen_id_still_resolves(registry_module):
    """Regression guard: the unknown-id behavior above must not break the
    happy path. The seeded Qwen entry must still be looked up successfully.
    """

    lookup = _resolve_lookup(registry_module)

    qwen_ids: list[str] = []
    for attr in ("iter_entries", "all_entries", "entries"):
        fn = getattr(registry_module, attr, None)
        if callable(fn):
            qwen_ids = [
                entry.model_id
                for entry in fn()
                if "qwen" in entry.model_id.lower()
            ]
            break
    else:
        registry_obj = getattr(
            registry_module, "REGISTRY", None
        ) or getattr(registry_module, "MODEL_REGISTRY", None)
        if registry_obj is not None:
            items = (
                registry_obj.values()
                if hasattr(registry_obj, "values")
                else registry_obj
            )
            qwen_ids = [
                entry.model_id
                for entry in items
                if "qwen" in entry.model_id.lower()
            ]

    assert qwen_ids, (
        "no Qwen entry available to test happy-path lookup; check that the "
        "registry is properly seeded before pinning the negative path."
    )

    entry = lookup(qwen_ids[0])
    # If the lookup uses the sentinel-style API, peel the value out before
    # asserting on it.
    if hasattr(entry, "found") and getattr(entry, "value", None) is not None:
        entry = entry.value
    assert getattr(entry, "model_id", None) == qwen_ids[0], (
        f"expected lookup({qwen_ids[0]!r}) to resolve to the seeded Qwen "
        f"entry; got {entry!r}"
    )


def test_lookup_does_not_mutate_registry(registry_module):
    """Looking up an unknown id must not write anything to the registry —
    no caching of placeholder defaults, no silent insert.
    """

    def _snapshot():
        for attr in ("iter_entries", "all_entries", "entries"):
            fn = getattr(registry_module, attr, None)
            if callable(fn):
                return tuple(entry.model_id for entry in fn())
        registry_obj = getattr(
            registry_module, "REGISTRY", None
        ) or getattr(registry_module, "MODEL_REGISTRY", None)
        if registry_obj is not None:
            items = (
                registry_obj.values()
                if hasattr(registry_obj, "values")
                else registry_obj
            )
            return tuple(entry.model_id for entry in items)
        pytest.fail("registry does not expose an iteration surface")

    before = _snapshot()
    lookup = _resolve_lookup(registry_module)
    try:
        lookup(UNKNOWN_MODEL_ID)
    except Exception:
        pass
    after = _snapshot()
    assert before == after, (
        "unknown-id lookup mutated the registry; got "
        f"before={before} after={after}"
    )
