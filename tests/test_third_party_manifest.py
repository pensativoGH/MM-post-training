"""M5 acceptance: pinned ``third_party/`` bootstrap manifest.

These tests pin the structural half of milestone M5:

- the YAML manifest exists at
  ``post_training/configs/third_party/manifest.yaml``
- it declares one top-level mapping per upstream family (``vjepa2``,
  ``wan22``, ``dreamdojo``)
- every entry carries at minimum ``repo_dir``, ``remote_url``,
  ``pinned_revision``, and ``bootstrap_kind``
- the manifest is loadable through the repo-owned bootstrap module
- no business-logic Python modules live under ``third_party/`` (wrapper
  code must stay outside that boundary)

The status/discovery halves of M5 are covered by
``test_third_party_revision_status.py`` and ``test_wrapper_discovery.py``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _REPO_ROOT / "post_training" / "configs" / "third_party" / "manifest.yaml"

REQUIRED_FAMILIES = ("vjepa2", "wan22", "dreamdojo")
REQUIRED_FIELDS = ("repo_dir", "remote_url", "pinned_revision", "bootstrap_kind")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_raw_manifest() -> dict:
    if not _MANIFEST_PATH.is_file():
        pytest.fail(
            f"Expected manifest at {_MANIFEST_PATH.relative_to(_REPO_ROOT)}; "
            "M5 requires this file to exist."
        )
    with _MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert isinstance(raw, Mapping), (
        f"manifest.yaml must deserialize to a top-level mapping; got {type(raw).__name__}"
    )
    return dict(raw)


def _entry_as_mapping(entry: object) -> Mapping[str, object]:
    """Coerce a manifest entry to a mapping for field access.

    The writer may model entries as plain dicts or as a dataclass; both
    shapes are accepted as long as the required fields are reachable.
    """

    if isinstance(entry, Mapping):
        return entry
    if hasattr(entry, "__dict__"):
        return {key: value for key, value in vars(entry).items()}
    # Fallback: try attribute access for each required field.
    return {field: getattr(entry, field) for field in REQUIRED_FIELDS if hasattr(entry, field)}


def _manifest_as_mapping(manifest: object) -> Mapping[str, object]:
    """Coerce a loaded manifest object into a family-keyed mapping."""

    if isinstance(manifest, Mapping):
        return manifest
    for attr in ("entries", "upstreams", "families", "third_parties"):
        value = getattr(manifest, attr, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        if isinstance(value, Mapping):
            return value
    raise AssertionError(
        f"Cannot adapt loaded manifest to a family-keyed mapping: {manifest!r}"
    )


# ---------------------------------------------------------------------------
# Manifest file structure
# ---------------------------------------------------------------------------


def test_manifest_file_exists_at_repo_owned_path():
    """The YAML manifest must live where the plan pins it."""

    assert _MANIFEST_PATH.is_file(), (
        f"Expected manifest at {_MANIFEST_PATH.relative_to(_REPO_ROOT)}; "
        "M5 requires this file."
    )


def test_manifest_declares_all_required_families():
    raw = _read_raw_manifest()
    missing = sorted(set(REQUIRED_FAMILIES) - set(raw.keys()))
    assert not missing, (
        f"manifest.yaml is missing required upstream families: {missing}. "
        f"Found top-level keys: {sorted(raw.keys())!r}"
    )


@pytest.mark.parametrize("family", REQUIRED_FAMILIES)
def test_manifest_entry_declares_required_fields(family):
    raw = _read_raw_manifest()
    assert family in raw, f"manifest.yaml is missing entry for {family!r}"

    entry = raw[family]
    assert isinstance(entry, Mapping), (
        f"manifest entry {family!r} must be a mapping; got {type(entry).__name__}"
    )

    missing = [field for field in REQUIRED_FIELDS if field not in entry]
    assert not missing, (
        f"manifest entry {family!r} is missing required field(s): {missing}. "
        f"Entry keys: {sorted(entry.keys())!r}"
    )

    for field in REQUIRED_FIELDS:
        value = entry[field]
        assert isinstance(value, str) and value.strip(), (
            f"manifest entry {family!r} field {field!r} must be a non-empty "
            f"string; got {value!r}"
        )


@pytest.mark.parametrize("family", REQUIRED_FAMILIES)
def test_manifest_entry_pinned_revision_is_recorded(family):
    """`pinned_revision` is the core of the contract — it must be an
    explicit commit SHA or release tag, not a placeholder like ``HEAD`` or
    ``main`` that floats.
    """

    raw = _read_raw_manifest()
    pinned = raw[family]["pinned_revision"]
    assert isinstance(pinned, str) and pinned.strip(), (
        f"{family}.pinned_revision must be a non-empty string"
    )
    forbidden = {"head", "main", "master", "trunk", "latest"}
    assert pinned.strip().lower() not in forbidden, (
        f"{family}.pinned_revision must be an explicit revision, not the "
        f"floating ref {pinned!r}"
    )


# ---------------------------------------------------------------------------
# Loadable through the repo-owned bootstrap module
# ---------------------------------------------------------------------------


def test_bootstrap_module_load_manifest_returns_all_families():
    """The wrapper code surface that M5 promises must be able to load the
    manifest. This is the API later milestones (M6, M6B, M7, M8) will use.
    """

    from verl_post_training.bootstrap.third_party import load_manifest

    manifest = load_manifest()
    families = _manifest_as_mapping(manifest)

    missing = sorted(set(REQUIRED_FAMILIES) - set(families.keys()))
    assert not missing, (
        f"load_manifest() result is missing families {missing}. "
        f"Found: {sorted(families.keys())!r}"
    )

    for family in REQUIRED_FAMILIES:
        entry = _entry_as_mapping(families[family])
        for field in REQUIRED_FIELDS:
            assert field in entry, (
                f"load_manifest()[{family!r}] does not surface field {field!r}; "
                f"available: {sorted(entry.keys())!r}"
            )


def test_bootstrap_module_load_manifest_accepts_custom_path(tmp_path):
    """The loader must accept an explicit manifest path so tests, smoke
    runs, and CI fixtures can swap in alternate manifests without mutating
    the repo-owned one.
    """

    from verl_post_training.bootstrap.third_party import load_manifest

    custom = tmp_path / "manifest.yaml"
    custom.write_text(
        "vjepa2:\n"
        "  repo_dir: tmp/vjepa2\n"
        "  remote_url: https://example.com/vjepa2.git\n"
        "  pinned_revision: 0000000000000000000000000000000000000000\n"
        "  bootstrap_kind: git\n"
        "wan22:\n"
        "  repo_dir: tmp/wan22\n"
        "  remote_url: https://example.com/wan22.git\n"
        "  pinned_revision: 1111111111111111111111111111111111111111\n"
        "  bootstrap_kind: git\n"
        "dreamdojo:\n"
        "  repo_dir: tmp/dreamdojo\n"
        "  remote_url: https://example.com/dreamdojo.git\n"
        "  pinned_revision: 2222222222222222222222222222222222222222\n"
        "  bootstrap_kind: git\n",
        encoding="utf-8",
    )

    manifest = load_manifest(custom)
    families = _manifest_as_mapping(manifest)
    assert set(families.keys()) >= set(REQUIRED_FAMILIES)
    entry = _entry_as_mapping(families["vjepa2"])
    assert entry["repo_dir"] == "tmp/vjepa2"
    assert entry["pinned_revision"] == "0000000000000000000000000000000000000000"


# ---------------------------------------------------------------------------
# No business-logic Python under third_party/
# ---------------------------------------------------------------------------


def test_no_repo_owned_python_modules_inside_third_party():
    """Repo-specific wrapper logic must live outside ``third_party/``.

    The simplest test is to ask git itself: are there any ``.py`` files
    tracked under ``third_party/``? Upstream checkouts are expected to be
    gitignored, so the only ``.py`` files git would see are repo-owned —
    which the criterion forbids.
    """

    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files", "--", "third_party"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        pytest.skip("git executable not available; cannot enforce structural check")
        return

    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    py_files = [path for path in tracked if path.endswith(".py")]
    assert not py_files, (
        "third_party/ must not contain repo-owned Python modules — wrapper "
        f"code belongs under post_training/src/. Found: {py_files!r}"
    )
