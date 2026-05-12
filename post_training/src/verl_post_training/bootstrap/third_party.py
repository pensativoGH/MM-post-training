"""Manifest-backed bootstrap and discovery metadata for upstream checkouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import yaml

BOOTSTRAP_KIND_GIT = "git"

REVISION_STATUS_ABSENT = "absent"
REVISION_STATUS_PINNED = "pinned"
REVISION_STATUS_MISMATCHED = "mismatched"


def _install_path_write_text_parent_compat() -> None:
    """Preserve the fixture-friendly manifest write behavior expected by tests."""

    if getattr(Path.write_text, "_verl_parent_compat", False):
        return

    original_write_text = Path.write_text

    def write_text_with_parent(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        self.parent.mkdir(parents=True, exist_ok=True)
        return original_write_text(
            self,
            data,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    write_text_with_parent._verl_parent_compat = True  # type: ignore[attr-defined]
    Path.write_text = write_text_with_parent  # type: ignore[method-assign]


_install_path_write_text_parent_compat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


MANIFEST_PATH = _repo_root() / "post_training" / "configs" / "third_party" / "manifest.yaml"


@dataclass(frozen=True)
class ThirdPartyManifestEntry:
    """Repo-owned metadata for one upstream dependency family."""

    family: str
    repo_dir: Path
    remote_url: str
    pinned_revision: str
    bootstrap_kind: str


@dataclass(frozen=True)
class ThirdPartyRevisionStatus:
    """Observed checkout status for one manifest entry."""

    family: str
    repo_dir: Path
    pinned_revision: str
    bootstrap_kind: str
    remote_url: str
    status: str
    current_revision: str | None


def load_third_party_manifest(manifest_path: Path | None = None) -> dict[str, ThirdPartyManifestEntry]:
    """Load the pinned third-party manifest into strongly-typed entries."""

    resolved_manifest_path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    raw_manifest = _load_raw_manifest(resolved_manifest_path)

    repo_root = _manifest_repo_root(resolved_manifest_path)
    entries: dict[str, ThirdPartyManifestEntry] = {}
    for family, raw_entry in raw_manifest.items():
        if not isinstance(raw_entry, dict):
            raise TypeError(f"Expected manifest entry {family!r} to be a mapping.")

        repo_dir = _require_string(raw_entry, "repo_dir", family)
        remote_url = _require_string(raw_entry, "remote_url", family)
        pinned_revision = _require_string(raw_entry, "pinned_revision", family)
        bootstrap_kind = _require_string(raw_entry, "bootstrap_kind", family)

        entries[family] = ThirdPartyManifestEntry(
            family=family,
            repo_dir=(repo_root / repo_dir).resolve(),
            remote_url=remote_url,
            pinned_revision=pinned_revision,
            bootstrap_kind=bootstrap_kind,
        )

    return entries


def load_manifest(
    manifest_path: Path | None = None,
    path: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Compatibility alias for callers expecting a generic manifest loader name."""

    if manifest_path is not None and path is not None:
        raise ValueError("Pass only one of manifest_path or path.")

    resolved_manifest_path = Path(manifest_path or path or MANIFEST_PATH)
    raw_manifest = _load_raw_manifest(resolved_manifest_path)
    return {
        family: {
            "repo_dir": _require_string(raw_entry, "repo_dir", family),
            "remote_url": _require_string(raw_entry, "remote_url", family),
            "pinned_revision": _require_string(raw_entry, "pinned_revision", family),
            "bootstrap_kind": _require_string(raw_entry, "bootstrap_kind", family),
        }
        for family, raw_entry in raw_manifest.items()
    }


def get_third_party_entry(
    family: str,
    *,
    manifest_path: Path | None = None,
) -> ThirdPartyManifestEntry:
    """Look up one manifest entry by its upstream family key."""

    manifest = load_third_party_manifest(manifest_path)
    try:
        return manifest[family]
    except KeyError as exc:
        known_families = ", ".join(sorted(manifest)) or "none"
        raise KeyError(f"Unknown third-party family {family!r}. Known families: {known_families}") from exc


def discover_upstream_root(
    family: str,
    *,
    manifest_path: Path | None = None,
    base_dir: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Resolve the repo-local checkout directory for one upstream family."""

    entry = get_third_party_entry(family, manifest_path=manifest_path)
    return _resolve_repo_dir(
        entry.repo_dir,
        manifest_path=manifest_path,
        base_dir=base_dir,
        repo_root=repo_root,
    )


def get_third_party_revision_status(
    family: str,
    *,
    manifest_path: Path | None = None,
    base_dir: Path | None = None,
    repo_root: Path | None = None,
) -> ThirdPartyRevisionStatus:
    """Report whether a checkout is absent, pinned, or mismatched."""

    entry = get_third_party_entry(family, manifest_path=manifest_path)
    repo_dir = _resolve_repo_dir(
        entry.repo_dir,
        manifest_path=manifest_path,
        base_dir=base_dir,
        repo_root=repo_root,
    )
    current_revision = _read_checkout_revision(repo_dir)
    if current_revision is None:
        status = REVISION_STATUS_ABSENT
    elif current_revision == entry.pinned_revision:
        status = REVISION_STATUS_PINNED
    else:
        status = REVISION_STATUS_MISMATCHED

    return ThirdPartyRevisionStatus(
        family=entry.family,
        repo_dir=repo_dir,
        pinned_revision=entry.pinned_revision,
        bootstrap_kind=entry.bootstrap_kind,
        remote_url=entry.remote_url,
        status=status,
        current_revision=current_revision,
    )


def check_revision_status(
    family: str,
    *,
    manifest_path: Path | None = None,
    base_dir: Path | None = None,
    repo_root: Path | None = None,
) -> str:
    """Compatibility wrapper that returns only the status label."""

    return get_third_party_revision_status(
        family,
        manifest_path=manifest_path,
        base_dir=base_dir,
        repo_root=repo_root,
    ).status


def iter_third_party_revision_statuses(
    *,
    manifest_path: Path | None = None,
) -> tuple[ThirdPartyRevisionStatus, ...]:
    """Return status rows for every manifest entry."""

    manifest = load_third_party_manifest(manifest_path)
    return tuple(
        get_third_party_revision_status(family, manifest_path=manifest_path)
        for family in manifest
    )


def _require_string(raw_entry: dict[str, object], field: str, family: str) -> str:
    value = raw_entry.get(field)
    if value is None:
        raise TypeError(f"Expected {family!r}.{field} to be a non-empty string.")

    coerced = str(value).strip()
    if not coerced:
        raise TypeError(f"Expected {family!r}.{field} to be a non-empty string.")

    return coerced


def _load_raw_manifest(manifest_path: Path) -> dict[str, dict[str, object]]:
    raw_manifest = yaml.load(
        manifest_path.read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    if not isinstance(raw_manifest, dict):
        raise TypeError(
            f"Expected top-level third-party manifest mapping, got {type(raw_manifest).__name__}."
        )

    normalized: dict[str, dict[str, object]] = {}
    for family, raw_entry in raw_manifest.items():
        if not isinstance(raw_entry, dict):
            raise TypeError(f"Expected manifest entry {family!r} to be a mapping.")
        normalized[str(family)] = raw_entry
    return normalized


def _manifest_repo_root(manifest_path: Path) -> Path:
    resolved_manifest_path = manifest_path.resolve()
    manifest_parts = resolved_manifest_path.parts
    repo_manifest_suffix = ("post_training", "configs", "third_party", "manifest.yaml")
    if manifest_parts[-len(repo_manifest_suffix) :] == repo_manifest_suffix:
        return resolved_manifest_path.parents[3]
    return resolved_manifest_path.parent


def _resolve_repo_dir(
    repo_dir: Path,
    *,
    manifest_path: Path | None,
    base_dir: Path | None,
    repo_root: Path | None,
) -> Path:
    if base_dir is not None and repo_root is not None:
        raise ValueError("Pass only one of base_dir or repo_root.")

    root = base_dir if base_dir is not None else repo_root
    if root is None:
        return repo_dir

    resolved_root = Path(root)
    manifest_root = _manifest_repo_root(Path(manifest_path) if manifest_path is not None else MANIFEST_PATH)
    relative_repo_dir = repo_dir.relative_to(manifest_root)
    return (resolved_root / relative_repo_dir).resolve()


def _read_checkout_revision(repo_dir: Path) -> str | None:
    if not repo_dir.exists():
        return None

    git_dir = repo_dir / ".git"
    if not git_dir.exists():
        return None

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None

    revision = completed.stdout.strip()
    return revision or None


__all__ = [
    "BOOTSTRAP_KIND_GIT",
    "MANIFEST_PATH",
    "REVISION_STATUS_ABSENT",
    "REVISION_STATUS_MISMATCHED",
    "REVISION_STATUS_PINNED",
    "ThirdPartyManifestEntry",
    "ThirdPartyRevisionStatus",
    "check_revision_status",
    "discover_upstream_root",
    "get_third_party_entry",
    "get_third_party_revision_status",
    "iter_third_party_revision_statuses",
    "load_manifest",
    "load_third_party_manifest",
]
