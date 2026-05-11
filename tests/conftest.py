"""Test bootstrap: make the repo-owned `verl_post_training` package importable.

The writer is expected to create `post_training/src/verl_post_training/...` as
part of M1. Adding that source path to `sys.path` here keeps tests runnable
without needing an installed distribution while the package is still under
development. Once the package ships a proper `pyproject.toml` and is installed
via `uv sync`, this shim becomes a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_SRC = _REPO_ROOT / "post_training" / "src"

if _PACKAGE_SRC.is_dir():
    src_str = str(_PACKAGE_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
