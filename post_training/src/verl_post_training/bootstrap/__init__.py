"""Repo-owned third-party bootstrap helpers."""

from .third_party import (
    BOOTSTRAP_KIND_GIT,
    REVISION_STATUS_ABSENT,
    REVISION_STATUS_MISMATCHED,
    REVISION_STATUS_PINNED,
    MANIFEST_PATH,
    ThirdPartyManifestEntry,
    ThirdPartyRevisionStatus,
    discover_upstream_root,
    get_third_party_entry,
    get_third_party_revision_status,
    iter_third_party_revision_statuses,
    load_third_party_manifest,
)

__all__ = [
    "BOOTSTRAP_KIND_GIT",
    "MANIFEST_PATH",
    "REVISION_STATUS_ABSENT",
    "REVISION_STATUS_MISMATCHED",
    "REVISION_STATUS_PINNED",
    "ThirdPartyManifestEntry",
    "ThirdPartyRevisionStatus",
    "discover_upstream_root",
    "get_third_party_entry",
    "get_third_party_revision_status",
    "iter_third_party_revision_statuses",
    "load_third_party_manifest",
]
