#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POST_TRAINING_PYTHONPATH="${ROOT_DIR}/world-model-post-training/shared/src:${ROOT_DIR}/world-model-post-training/vjepa/src:${ROOT_DIR}/world-model-post-training/wan/src:${ROOT_DIR}/world-model-post-training/dreamdojo/src"
export PYTHONPATH="${POST_TRAINING_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
from __future__ import annotations

from pathlib import Path
import subprocess

from verl_post_training.bootstrap.third_party import (
    BOOTSTRAP_KIND_GIT,
    MANIFEST_PATH,
    REVISION_STATUS_ABSENT,
    get_third_party_entry,
    iter_third_party_revision_statuses,
    load_third_party_manifest,
)


def ensure_checkout(family: str) -> None:
    entry = get_third_party_entry(family)
    if entry.bootstrap_kind != BOOTSTRAP_KIND_GIT:
        raise SystemExit(f"Unsupported bootstrap_kind for {family}: {entry.bootstrap_kind}")

    if not entry.repo_dir.exists():
        entry.repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", entry.remote_url, str(entry.repo_dir)],
            check=True,
        )

    subprocess.run(["git", "fetch", "--all", "--tags"], cwd=entry.repo_dir, check=True)
    subprocess.run(["git", "checkout", entry.pinned_revision], cwd=entry.repo_dir, check=True)


manifest = load_third_party_manifest(MANIFEST_PATH)
for family in manifest:
    ensure_checkout(family)

for status in iter_third_party_revision_statuses(manifest_path=MANIFEST_PATH):
    summary = status.status
    if status.status == REVISION_STATUS_ABSENT:
        summary = f"{summary} ({status.repo_dir})"
    print(f"{status.family}: {summary}")
PY
