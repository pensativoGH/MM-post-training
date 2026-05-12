"""Repo-root import shim for the in-tree ``verl_post_training`` package."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_POST_TRAINING_ROOT = _REPO_ROOT / "world-model-post-training"
_SRC_ROOTS = [
    _POST_TRAINING_ROOT / "shared" / "src",
    _POST_TRAINING_ROOT / "vjepa" / "src",
    _POST_TRAINING_ROOT / "wan" / "src",
    _POST_TRAINING_ROOT / "dreamdojo" / "src",
]

for _src_root in reversed(_SRC_ROOTS):
    if _src_root.is_dir() and str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))

_SRC_PACKAGE = _SRC_ROOTS[0] / "verl_post_training"

if not _SRC_PACKAGE.is_dir():
    raise ModuleNotFoundError(
        f"Could not locate source package directory: {_SRC_PACKAGE}"
    )

__path__ = [str(_SRC_PACKAGE)]
