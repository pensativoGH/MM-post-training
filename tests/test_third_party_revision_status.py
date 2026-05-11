"""M5 acceptance: bootstrap status reports absent, pinned, or mismatched.

This file pins the runtime-state half of the third_party contract:

    the bootstrap path can report whether each upstream checkout is
    absent, present at the pinned revision, or present at a mismatched
    revision

The repo-owned helper under ``verl_post_training.bootstrap.third_party``
must expose a ``check_revision_status`` (or equivalent) function that
accepts a family name from the manifest plus an override for the manifest
location and base directory the upstream is rooted at, and returns one of
three discriminable states.

Tests are deterministic: each scenario builds a tmp git repo so the test
does not depend on whether the real upstream was bootstrapped on the host.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_available() -> bool:
    try:
        subprocess.run(
            ["git", "--version"], capture_output=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def _init_git_repo_with_commit(path: Path) -> str:
    """Initialize a git repo at ``path`` with a single commit; return SHA."""

    path.mkdir(parents=True, exist_ok=True)
    env_args = [
        "-c", "user.email=test@example.invalid",
        "-c", "user.name=test",
        "-c", "commit.gpgsign=false",
        "-c", "init.defaultBranch=main",
    ]
    subprocess.run(
        ["git", *env_args, "init", "-q"],
        cwd=path, check=True, capture_output=True,
    )
    (path / "README").write_text("upstream stub\n", encoding="utf-8")
    subprocess.run(
        ["git", *env_args, "add", "."],
        cwd=path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", *env_args, "commit", "-q", "-m", "init"],
        cwd=path, check=True, capture_output=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _write_manifest(
    tmp_path: Path,
    *,
    family: str,
    repo_dir: str,
    pinned_revision: str,
    remote_url: str = "https://example.com/upstream.git",
    bootstrap_kind: str = "git",
) -> Path:
    """Write a one-entry manifest pointing at a tmp_path-relative repo_dir."""

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"{family}:\n"
        f"  repo_dir: {repo_dir}\n"
        f"  remote_url: {remote_url}\n"
        f"  pinned_revision: {pinned_revision}\n"
        f"  bootstrap_kind: {bootstrap_kind}\n",
        encoding="utf-8",
    )
    return manifest


def _status_name(value: object) -> str:
    """Normalize a status return value to a comparable lowercase string.

    The writer may return an enum (``BootstrapStatus.PINNED``) or a plain
    string (``"pinned"``); both shapes are accepted.
    """

    if value is None:
        raise AssertionError("check_revision_status returned None")
    raw = getattr(value, "value", None)
    if raw is None:
        raw = getattr(value, "name", None)
    if raw is None:
        raw = value
    return str(raw).strip().lower()


def _check_status(name: str, *, manifest_path: Path, base_dir: Path):
    """Call the writer's status helper, tolerating reasonable signature shapes."""

    from verl_post_training.bootstrap import third_party as module

    func = getattr(module, "check_revision_status", None)
    if func is None:
        pytest.fail(
            "verl_post_training.bootstrap.third_party must expose "
            "`check_revision_status` so callers can tell whether an upstream "
            "checkout is absent, pinned, or mismatched."
        )

    # Try a few reasonable keyword shapes so the writer has room to pick
    # naming without breaking the contract.
    last_err: TypeError | None = None
    for kwargs in (
        {"manifest_path": manifest_path, "base_dir": base_dir},
        {"manifest_path": manifest_path, "repo_root": base_dir},
        {"manifest": manifest_path, "base_dir": base_dir},
    ):
        try:
            return func(name, **kwargs)
        except TypeError as exc:
            last_err = exc
            continue
    raise AssertionError(
        "check_revision_status must accept a manifest path override and a "
        "base directory override so tests can pin a self-contained scenario. "
        f"Last TypeError: {last_err!r}"
    )


@pytest.fixture(autouse=True)
def _require_git():
    if not _git_available():
        pytest.skip("git is not available on this host")


# ---------------------------------------------------------------------------
# absent
# ---------------------------------------------------------------------------


def test_status_absent_when_repo_dir_does_not_exist(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        family="vjepa2",
        repo_dir="checkouts/vjepa2",
        pinned_revision="0" * 40,
    )
    # Intentionally do NOT create checkouts/vjepa2

    status = _check_status("vjepa2", manifest_path=manifest, base_dir=tmp_path)
    assert _status_name(status) == "absent", (
        f"missing upstream directory must report status=absent; got {status!r}"
    )


# ---------------------------------------------------------------------------
# pinned
# ---------------------------------------------------------------------------


def test_status_pinned_when_head_matches_manifest_revision(tmp_path):
    repo_dir_rel = "checkouts/vjepa2"
    sha = _init_git_repo_with_commit(tmp_path / repo_dir_rel)

    manifest = _write_manifest(
        tmp_path,
        family="vjepa2",
        repo_dir=repo_dir_rel,
        pinned_revision=sha,
    )

    status = _check_status("vjepa2", manifest_path=manifest, base_dir=tmp_path)
    assert _status_name(status) == "pinned", (
        f"HEAD matches manifest pinned_revision={sha!r}; expected status=pinned, "
        f"got {status!r}"
    )


# ---------------------------------------------------------------------------
# mismatched
# ---------------------------------------------------------------------------


def test_status_mismatched_when_head_differs_from_manifest_revision(tmp_path):
    repo_dir_rel = "checkouts/wan22"
    _init_git_repo_with_commit(tmp_path / repo_dir_rel)

    # Pin to a deliberately unrelated SHA so HEAD != pinned_revision.
    wrong_sha = "deadbeef" * 5  # 40 hex chars
    manifest = _write_manifest(
        tmp_path,
        family="wan22",
        repo_dir=repo_dir_rel,
        pinned_revision=wrong_sha,
    )

    status = _check_status("wan22", manifest_path=manifest, base_dir=tmp_path)
    assert _status_name(status) == "mismatched", (
        f"HEAD differs from manifest pinned_revision; expected status=mismatched, "
        f"got {status!r}"
    )


def test_three_status_values_are_distinguishable(tmp_path):
    """Sanity check: the three states are not aliases of each other."""

    # absent
    manifest_absent = _write_manifest(
        tmp_path / "absent",
        family="dreamdojo",
        repo_dir="checkouts/dreamdojo",
        pinned_revision="0" * 40,
    )
    (tmp_path / "absent").mkdir(parents=True, exist_ok=True)
    absent = _status_name(
        _check_status(
            "dreamdojo", manifest_path=manifest_absent, base_dir=tmp_path / "absent"
        )
    )

    # pinned
    pinned_root = tmp_path / "pinned"
    pinned_root.mkdir(parents=True, exist_ok=True)
    sha = _init_git_repo_with_commit(pinned_root / "checkouts/dreamdojo")
    manifest_pinned = _write_manifest(
        pinned_root,
        family="dreamdojo",
        repo_dir="checkouts/dreamdojo",
        pinned_revision=sha,
    )
    pinned = _status_name(
        _check_status(
            "dreamdojo", manifest_path=manifest_pinned, base_dir=pinned_root
        )
    )

    # mismatched
    mm_root = tmp_path / "mm"
    mm_root.mkdir(parents=True, exist_ok=True)
    _init_git_repo_with_commit(mm_root / "checkouts/dreamdojo")
    manifest_mm = _write_manifest(
        mm_root,
        family="dreamdojo",
        repo_dir="checkouts/dreamdojo",
        pinned_revision="cafef00d" * 5,
    )
    mismatched = _status_name(
        _check_status(
            "dreamdojo", manifest_path=manifest_mm, base_dir=mm_root
        )
    )

    assert {absent, pinned, mismatched} == {"absent", "pinned", "mismatched"}, (
        "the three status values must be distinguishable; got "
        f"absent={absent!r}, pinned={pinned!r}, mismatched={mismatched!r}"
    )
