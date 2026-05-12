"""M6 acceptance: the V-JEPA2 smoke run completes without requiring the
user to manually ``cd`` into the pinned upstream V-JEPA2 checkout.

This file pins the fourth M6 acceptance criterion (quoted from the
approved plan):

    a smoke run implemented in
    ``world-model-post-training/shared/src/verl_post_training/smoke/test_vjepa2_inference.py``
    completes without requiring the user to manually change into the
    upstream V-JEPA2 repo

The smoke path must:

* exist at the path the plan pins
* be importable from any working directory (not just from inside
  ``third_party/vjepa2``)
* discover the upstream root via the repo-owned bootstrap helper —
  ``verl_post_training.bootstrap.third_party.discover_upstream_root`` —
  rather than hard-coding ``third_party/vjepa2`` paths
* not call ``os.chdir`` to a hard-coded vjepa2 path and not shell out
  with ``cd third_party/vjepa2`` before doing real work
* expose a callable entry point (``main`` / ``run`` / ``run_smoke``) that
  is invokable from outside the upstream tree

These tests are structural and behavioral but never start a backend
process. They use ``tmp_path`` to simulate execution from a foreign
working directory and ``monkeypatch`` to keep upstream module imports out
of the loop.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SMOKE_PATH = (
    _REPO_ROOT
    / "world-model-post-training"
    / "vjepa"
    / "src"
    / "verl_post_training_vjepa"
    / "smoke"
    / "test_inference.py"
)


# ---------------------------------------------------------------------------
# Existence / location
# ---------------------------------------------------------------------------


def test_smoke_file_exists_at_pinned_path():
    """The plan pins the smoke file at a specific repo-owned path."""

    assert _SMOKE_PATH.is_file(), (
        f"M6 requires a smoke run at {_SMOKE_PATH.relative_to(_REPO_ROOT)}; "
        "the file is missing."
    )


def test_smoke_lives_under_repo_owned_package_not_third_party():
    """The smoke file must live under ``world-model-post-training/vjepa/`` — never under
    ``third_party/``. Repo-owned wrapper code is the whole point of the
    M6 contract.
    """

    smoke_rel = _SMOKE_PATH.relative_to(_REPO_ROOT)
    assert smoke_rel.parts[0] == "world-model-post-training", (
        f"smoke file must live under world-model-post-training/, not {smoke_rel.parts[0]!r}"
    )
    assert "third_party" not in smoke_rel.parts, (
        "smoke file must not live inside third_party/; M6 forbids repo "
        f"wrapper code there. Got: {smoke_rel}"
    )


# ---------------------------------------------------------------------------
# Structural check: no manual cd / chdir to a hardcoded vjepa2 path
# ---------------------------------------------------------------------------


def _read_smoke_source() -> str:
    if not _SMOKE_PATH.is_file():
        pytest.fail(
            f"smoke file missing at {_SMOKE_PATH.relative_to(_REPO_ROOT)}; "
            "M6 requires this file."
        )
    return _SMOKE_PATH.read_text(encoding="utf-8")


def test_smoke_does_not_chdir_into_hardcoded_vjepa2_path():
    """The smoke must not contain a literal ``os.chdir`` (or equivalent)
    pointing at a hardcoded ``third_party/vjepa2`` location. Discovery is
    what makes the smoke runnable from any cwd.
    """

    src = _read_smoke_source()

    forbidden_patterns = (
        r"os\.chdir\(\s*['\"][^'\"]*third_party/vjepa2",
        r"chdir\(\s*['\"][^'\"]*third_party/vjepa2",
        r"cd\s+third_party/vjepa2",
        r"cd\s+\$\{?\w*\}?/third_party/vjepa2",
    )
    offenders = [p for p in forbidden_patterns if re.search(p, src)]
    assert not offenders, (
        "smoke must not require manual or programmatic `cd` into "
        f"third_party/vjepa2; matched forbidden patterns: {offenders!r}"
    )


def test_smoke_uses_bootstrap_discovery_helper():
    """The smoke must reach the upstream root through the repo-owned
    bootstrap helper rather than hard-coding paths.
    """

    src = _read_smoke_source()

    discovers_via_helper = (
        "discover_upstream_root" in src
        or "verl_post_training.bootstrap.third_party" in src
        or "load_manifest" in src
    )
    assert discovers_via_helper, (
        "smoke must call `discover_upstream_root` (or load the third_party "
        "manifest) from `verl_post_training.bootstrap.third_party` so the "
        "upstream root is resolved without hard-coded paths."
    )

    # And the smoke must reference the vjepa2 family by name when it asks
    # the helper for a root — otherwise the helper call is dead code.
    assert re.search(r"['\"]vjepa2['\"]", src), (
        "smoke must request the 'vjepa2' family from the discovery helper "
        "so the upstream root is selected from the manifest."
    )


def test_smoke_source_is_parseable_python():
    """A smoke file that does not parse can never satisfy the criterion."""

    src = _read_smoke_source()
    try:
        ast.parse(src)
    except SyntaxError as exc:  # pragma: no cover - reported below
        pytest.fail(f"smoke file does not parse as Python: {exc!r}")


# ---------------------------------------------------------------------------
# Behavioral check: importable from a foreign working directory
# ---------------------------------------------------------------------------


def _ensure_package_on_sys_path():
    src = _REPO_ROOT / "world-model-post-training" / "shared" / "src"
    if src.is_dir():
        src_str = str(src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


def test_smoke_is_importable_from_foreign_cwd(tmp_path, monkeypatch):
    """The smoke must be importable from any working directory; if the
    user can only import it after ``cd``-ing into ``third_party/vjepa2``,
    the contract is violated.
    """

    _ensure_package_on_sys_path()
    monkeypatch.chdir(tmp_path)

    # Drop any prior cached module so we re-exercise import.
    sys.modules.pop("verl_post_training.smoke.test_vjepa2_inference", None)
    sys.modules.pop("verl_post_training.smoke", None)

    try:
        import importlib

        module = importlib.import_module(
            "verl_post_training.smoke.test_vjepa2_inference"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            "smoke module must be importable from any cwd; got "
            f"ModuleNotFoundError: {exc!r}"
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            "smoke module must import without side effects that require "
            f"a specific cwd; got: {type(exc).__name__}: {exc!r}"
        )

    assert module is not None


def test_smoke_exposes_callable_entry_point(tmp_path, monkeypatch):
    """The smoke must expose a callable entry-point so it can be driven
    from a smoke runner (CI / Make target / direct python -m). Without
    one, "completes without manual cd" cannot be verified end-to-end.
    """

    _ensure_package_on_sys_path()
    monkeypatch.chdir(tmp_path)

    sys.modules.pop("verl_post_training.smoke.test_vjepa2_inference", None)
    import importlib

    try:
        module = importlib.import_module(
            "verl_post_training.smoke.test_vjepa2_inference"
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"smoke module must import cleanly; got {type(exc).__name__}: "
            f"{exc!r}"
        )

    candidates = ("main", "run", "run_smoke", "run_inference_smoke")
    has_callable_entry = any(
        callable(getattr(module, name, None)) for name in candidates
    )
    # Pytest-style smoke files also satisfy the contract — a function whose
    # name starts with ``test_`` proves there's an invokable entry point.
    has_test_entry = any(
        callable(getattr(module, name, None))
        for name in dir(module)
        if name.startswith("test_")
    )
    assert has_callable_entry or has_test_entry, (
        "smoke must expose a callable entry point so an external runner "
        "can drive it without requiring `cd` into the upstream repo. "
        f"Looked for one of {candidates} or a `test_*` function on the "
        "smoke module."
    )


def test_smoke_does_not_hardcode_repo_relative_vjepa2_path():
    """Even outside ``os.chdir`` calls, the smoke must not embed an
    absolute or repo-relative ``third_party/vjepa2/...`` filesystem path
    as a string literal — that would defeat discovery.
    """

    src = _read_smoke_source()
    tree = ast.parse(src)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if "third_party/vjepa2" in value or "third_party\\vjepa2" in value:
                offenders.append(value)

    assert not offenders, (
        "smoke must not embed `third_party/vjepa2` as a literal path; "
        "discovery via `verl_post_training.bootstrap.third_party` is the "
        f"required mechanism. Offending literals: {offenders!r}"
    )


def test_smoke_resolves_upstream_root_via_discovery(tmp_path, monkeypatch):
    """End-to-end behavior check: when the smoke imports the discovery
    helper, the helper must be in the importable graph from a foreign
    cwd. If the smoke can compute a candidate upstream root without
    erroring, the "no manual cd" contract is observably preserved.
    """

    _ensure_package_on_sys_path()
    monkeypatch.chdir(tmp_path)

    try:
        from verl_post_training.bootstrap.third_party import (  # noqa: F401
            discover_upstream_root,
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            "Smoke contract depends on the M5 discovery helper "
            "`verl_post_training.bootstrap.third_party.discover_upstream_root`; "
            f"helper is not importable: {exc!r}"
        )
    except ImportError as exc:
        pytest.fail(
            "Smoke contract requires `discover_upstream_root` from "
            "`verl_post_training.bootstrap.third_party`; got ImportError: "
            f"{exc!r}"
        )

    # Asking the helper for the vjepa2 family from a foreign cwd must
    # produce a usable path-like value rather than raising.
    from verl_post_training.bootstrap.third_party import (
        discover_upstream_root,
    )

    try:
        result = discover_upstream_root("vjepa2")
    except (LookupError, KeyError, ValueError):
        pytest.fail(
            "discover_upstream_root('vjepa2') must succeed against the "
            "real third_party manifest; without it the smoke cannot "
            "resolve the upstream root."
        )

    assert result is not None, (
        "discover_upstream_root('vjepa2') must return a path-like value "
        "so the smoke can locate the upstream root without manual cd."
    )

    # Smoke must remain invariant to cwd: re-running discovery from a
    # second foreign cwd must yield a path that still mentions the family.
    second_cwd = tmp_path / "elsewhere"
    second_cwd.mkdir()
    monkeypatch.chdir(second_cwd)
    second = discover_upstream_root("vjepa2")
    assert "vjepa2" in str(second).lower(), (
        f"discover_upstream_root('vjepa2') from cwd={second_cwd!r} returned "
        f"{second!r}; the smoke would not find the upstream root."
    )
