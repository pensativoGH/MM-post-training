from __future__ import annotations

from pathlib import Path
import subprocess

import yaml

from verl_post_training.adapters.dataset import (
    DreamDojoDatasetAdapter,
    VJEPA2DatasetAdapter,
    WanDatasetAdapter,
)
from verl_post_training.bootstrap.third_party import (
    MANIFEST_PATH,
    REVISION_STATUS_ABSENT,
    REVISION_STATUS_MISMATCHED,
    REVISION_STATUS_PINNED,
    discover_upstream_root,
    get_third_party_revision_status,
    load_third_party_manifest,
)


def test_manifest_declares_required_upstream_entries() -> None:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert set(manifest) >= {"vjepa2", "wan22", "dreamdojo"}
    for family in ("vjepa2", "wan22", "dreamdojo"):
        assert set(manifest[family]) >= {
            "repo_dir",
            "remote_url",
            "pinned_revision",
            "bootstrap_kind",
        }


def test_revision_status_reports_absent_pinned_and_mismatched(tmp_path: Path) -> None:
    repo_dir = tmp_path / "third_party" / "vjepa2"
    repo_dir.mkdir(parents=True)

    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True, text=True)
    (repo_dir / "README.md").write_text("bootstrap smoke\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_dir, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "initial",
        ],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    pinned_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    manifest_path = tmp_path / "post_training" / "configs" / "third_party" / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "vjepa2": {
                    "repo_dir": "third_party/vjepa2",
                    "remote_url": "https://example.invalid/vjepa2.git",
                    "pinned_revision": pinned_revision,
                    "bootstrap_kind": "git",
                },
                "wan22": {
                    "repo_dir": "third_party/wan22",
                    "remote_url": "https://example.invalid/wan22.git",
                    "pinned_revision": "deadbeef",
                    "bootstrap_kind": "git",
                },
                "dreamdojo": {
                    "repo_dir": "third_party/dreamdojo",
                    "remote_url": "https://example.invalid/dreamdojo.git",
                    "pinned_revision": "cafebabe",
                    "bootstrap_kind": "git",
                },
            }
        ),
        encoding="utf-8",
    )

    pinned = get_third_party_revision_status("vjepa2", manifest_path=manifest_path)
    mismatched = get_third_party_revision_status("wan22", manifest_path=manifest_path)
    absent = get_third_party_revision_status("dreamdojo", manifest_path=manifest_path)

    assert pinned.status == REVISION_STATUS_PINNED
    assert pinned.current_revision == pinned_revision
    assert mismatched.status == REVISION_STATUS_ABSENT
    assert absent.status == REVISION_STATUS_ABSENT

    wan_repo = tmp_path / "third_party" / "wan22"
    wan_repo.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=wan_repo, check=True, capture_output=True, text=True)
    (wan_repo / "README.md").write_text("wan\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=wan_repo, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "initial",
        ],
        cwd=wan_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    mismatched = get_third_party_revision_status("wan22", manifest_path=manifest_path)
    assert mismatched.status == REVISION_STATUS_MISMATCHED


def test_wrappers_discover_upstream_roots_from_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "post_training" / "configs" / "third_party" / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "vjepa2": {
                    "repo_dir": "vendor/vjepa2-src",
                    "remote_url": "https://example.invalid/vjepa2.git",
                    "pinned_revision": "1234",
                    "bootstrap_kind": "git",
                },
                "wan22": {
                    "repo_dir": "vendor/wan22-src",
                    "remote_url": "https://example.invalid/wan22.git",
                    "pinned_revision": "2345",
                    "bootstrap_kind": "git",
                },
                "dreamdojo": {
                    "repo_dir": "vendor/dreamdojo-src",
                    "remote_url": "https://example.invalid/dreamdojo.git",
                    "pinned_revision": "3456",
                    "bootstrap_kind": "git",
                },
            }
        ),
        encoding="utf-8",
    )

    expected = load_third_party_manifest(manifest_path)

    assert discover_upstream_root("vjepa2", manifest_path=manifest_path) == expected["vjepa2"].repo_dir
    assert VJEPA2DatasetAdapter().discover_upstream_root(manifest_path=manifest_path) == expected["vjepa2"].repo_dir
    assert WanDatasetAdapter().discover_upstream_root(manifest_path=manifest_path) == expected["wan22"].repo_dir
    assert DreamDojoDatasetAdapter().discover_upstream_root(manifest_path=manifest_path) == expected["dreamdojo"].repo_dir
