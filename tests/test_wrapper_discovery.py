"""M5 acceptance: wrapper code discovers upstream roots from the manifest.

This pins the discovery half of the third_party contract:

    wrapper code can discover each upstream root from the manifest rather
    than from hard-coded absolute or relative paths in user-facing scripts

The expected surface is a function under
``verl_post_training.bootstrap.third_party`` (e.g. ``discover_upstream_root``)
that takes a family name, consults the manifest, and returns the resolved
filesystem path for that upstream checkout.

The "not hard-coded" guarantee is exercised by pointing discovery at a
tmp_path manifest and asserting the resolved path follows that manifest —
not a value compiled into Python.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest


REQUIRED_FAMILIES = ("vjepa2", "wan22", "dreamdojo")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover(name: str, *, manifest_path: Path | None = None, base_dir: Path | None = None):
    """Invoke the writer's discovery helper across a few reasonable shapes."""

    from verl_post_training.bootstrap import third_party as module

    func = getattr(module, "discover_upstream_root", None)
    if func is None:
        pytest.fail(
            "verl_post_training.bootstrap.third_party must expose "
            "`discover_upstream_root(name, ...)` so wrappers can locate the "
            "pinned upstream checkout without hard-coded paths."
        )

    kwargs_candidates = []
    if manifest_path is None and base_dir is None:
        kwargs_candidates.append({})
    if manifest_path is not None:
        kwargs_candidates.append({"manifest_path": manifest_path, "repo_root": base_dir})
        kwargs_candidates.append({"manifest_path": manifest_path, "base_dir": base_dir})
        kwargs_candidates.append({"manifest_path": manifest_path})
    if base_dir is not None and manifest_path is None:
        kwargs_candidates.append({"repo_root": base_dir})
        kwargs_candidates.append({"base_dir": base_dir})

    last_err: TypeError | None = None
    for kwargs in kwargs_candidates:
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            return func(name, **clean_kwargs)
        except TypeError as exc:
            last_err = exc
            continue

    raise AssertionError(
        "discover_upstream_root must accept overrides for the manifest path "
        "and base directory so tests and smoke runs can pin a self-contained "
        f"scenario. Last TypeError: {last_err!r}"
    )


def _write_custom_manifest(
    tmp_path: Path,
    *,
    repo_dirs: Mapping[str, str],
) -> Path:
    """Write a manifest with caller-controlled ``repo_dir`` values."""

    lines: list[str] = []
    for family, repo_dir in repo_dirs.items():
        lines.append(f"{family}:")
        lines.append(f"  repo_dir: {repo_dir}")
        lines.append(f"  remote_url: https://example.com/{family}.git")
        lines.append(f"  pinned_revision: {'0' * 40}")
        lines.append("  bootstrap_kind: git")
    path = tmp_path / "manifest.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Real repo-owned manifest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", REQUIRED_FAMILIES)
def test_discover_upstream_root_for_each_required_family(family):
    """Discovery must succeed for every family the manifest declares."""

    result = _discover(family)
    assert result is not None, (
        f"discover_upstream_root({family!r}) must return a path-like value, "
        "not None"
    )
    path = Path(result)
    # The discovered path string should mention the family name in some
    # form — guarding against a stub that returns a single shared root.
    family_aliases = {family, family.replace("22", "")}
    text = str(path).lower()
    assert any(alias in text for alias in family_aliases), (
        f"discover_upstream_root({family!r}) returned {path!r}; expected the "
        f"path to reference the family name (one of {family_aliases!r})."
    )


def test_discover_upstream_root_matches_real_manifest_repo_dir():
    """The discovered path must mirror the manifest's ``repo_dir`` value
    (joined to the repo root for relative entries). This proves discovery
    reads the manifest rather than relying on a duplicate constant.
    """

    from verl_post_training.bootstrap.third_party import load_manifest

    manifest = load_manifest()
    # Reduce manifest to a family-keyed mapping
    if not isinstance(manifest, Mapping):
        for attr in ("entries", "upstreams", "families", "third_parties"):
            candidate = getattr(manifest, attr, None)
            if candidate is not None:
                manifest = candidate() if callable(candidate) else candidate
                break
    assert isinstance(manifest, Mapping)

    entry = manifest["vjepa2"]
    if not isinstance(entry, Mapping):
        entry = {key: getattr(entry, key) for key in ("repo_dir",) if hasattr(entry, key)}
    repo_dir = entry["repo_dir"]
    discovered = Path(_discover("vjepa2"))

    expected_tail = Path(repo_dir)
    discovered_str = str(discovered)
    assert discovered_str.endswith(str(expected_tail)) or expected_tail == discovered, (
        f"discover_upstream_root('vjepa2') = {discovered!r} does not match "
        f"manifest repo_dir={repo_dir!r}; discovery must be derived from "
        "manifest data."
    )


# ---------------------------------------------------------------------------
# Custom manifest — proves no hard-coded paths
# ---------------------------------------------------------------------------


def test_discover_follows_custom_manifest_repo_dir(tmp_path):
    """Pointing discovery at a custom manifest must change the resolved
    path. If the result is independent of the manifest contents, then the
    paths are hard-coded somewhere — exactly what the criterion forbids.
    """

    custom_manifest = _write_custom_manifest(
        tmp_path,
        repo_dirs={
            "vjepa2": "alt/vjepa2_checkout",
            "wan22": "alt/wan22_checkout",
            "dreamdojo": "alt/dreamdojo_checkout",
        },
    )

    base = tmp_path
    discovered_vjepa = Path(_discover("vjepa2", manifest_path=custom_manifest, base_dir=base))
    discovered_wan = Path(_discover("wan22", manifest_path=custom_manifest, base_dir=base))
    discovered_dd = Path(_discover("dreamdojo", manifest_path=custom_manifest, base_dir=base))

    for discovered, repo_dir in (
        (discovered_vjepa, "alt/vjepa2_checkout"),
        (discovered_wan, "alt/wan22_checkout"),
        (discovered_dd, "alt/dreamdojo_checkout"),
    ):
        assert str(discovered).endswith(repo_dir), (
            f"custom manifest set repo_dir={repo_dir!r}, but discovery "
            f"returned {discovered!r}. Discovery must follow manifest data."
        )

    # Cross-check distinctness: different manifest entries must yield
    # different discovered paths.
    assert (
        discovered_vjepa != discovered_wan
        and discovered_wan != discovered_dd
        and discovered_vjepa != discovered_dd
    ), (
        "discover_upstream_root must return distinct paths for distinct "
        f"families; got {discovered_vjepa!r}, {discovered_wan!r}, {discovered_dd!r}"
    )


def test_discover_unknown_family_raises(tmp_path):
    """Asking for a family the manifest does not declare must fail
    explicitly rather than silently returning a default or repo-root path.
    """

    custom_manifest = _write_custom_manifest(
        tmp_path,
        repo_dirs={
            "vjepa2": "alt/vjepa2_checkout",
            "wan22": "alt/wan22_checkout",
            "dreamdojo": "alt/dreamdojo_checkout",
        },
    )

    with pytest.raises((LookupError, KeyError, ValueError)):
        _discover(
            "totally-not-an-upstream",
            manifest_path=custom_manifest,
            base_dir=tmp_path,
        )


def test_discover_resolves_under_supplied_base_dir(tmp_path):
    """When ``repo_dir`` is relative, discovery must join it against the
    supplied base directory. This is the mechanism that lets the bootstrap
    helpers live anywhere on disk without leaking absolute paths into the
    manifest.
    """

    custom_manifest = _write_custom_manifest(
        tmp_path,
        repo_dirs={
            "vjepa2": "checkouts/vjepa2",
            "wan22": "checkouts/wan22",
            "dreamdojo": "checkouts/dreamdojo",
        },
    )

    discovered = Path(
        _discover("vjepa2", manifest_path=custom_manifest, base_dir=tmp_path)
    )

    expected = tmp_path / "checkouts" / "vjepa2"
    assert discovered == expected or discovered.resolve() == expected.resolve(), (
        f"relative repo_dir must resolve under base_dir; expected {expected!r}, "
        f"got {discovered!r}"
    )
