"""Repo-root import shim for the in-tree ``verl_post_training`` package."""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = (
    Path(__file__).resolve().parent.parent
    / "post_training"
    / "src"
    / "verl_post_training"
)

if not _SRC_PACKAGE.is_dir():
    raise ModuleNotFoundError(
        f"Could not locate source package directory: {_SRC_PACKAGE}"
    )

__path__ = [str(_SRC_PACKAGE)]
