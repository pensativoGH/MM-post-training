"""Test bootstrap: make repo-owned post-training packages importable."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_POST_TRAINING_ROOT = _REPO_ROOT / "post_training"
_PACKAGE_SRCS = [
    _POST_TRAINING_ROOT / "shared" / "src",
    _POST_TRAINING_ROOT / "vjepa" / "src",
    _POST_TRAINING_ROOT / "wan" / "src",
    _POST_TRAINING_ROOT / "dreamdojo" / "src",
]

for _package_src in reversed(_PACKAGE_SRCS):
    if _package_src.is_dir():
        src_str = str(_package_src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
