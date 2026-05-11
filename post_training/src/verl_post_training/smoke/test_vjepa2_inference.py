"""Smoke helpers for repo-local V-JEPA2 inference coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..adapters.runtime import run_runtime
from ..bootstrap.third_party import discover_upstream_root
from ..launch.load_config import TaskConfig


def run_vjepa2_smoke(
    config: TaskConfig,
    *,
    pipeline_manifest: Any = None,
    output_dir: str | Path | None = None,
    split: str = "smoke",
    **kwargs: Any,
) -> dict[str, object]:
    """Execute the repo-owned smoke path without shelling into third_party/."""

    discover_upstream_root("vjepa2")
    return run_runtime(
        config,
        pipeline_manifest=pipeline_manifest,
        output_dir=Path(output_dir) if output_dir is not None else None,
        split=split,
        **kwargs,
    )


def main(
    config: TaskConfig,
    *,
    pipeline_manifest: Any = None,
    output_dir: str | Path | None = None,
    split: str = "smoke",
    **kwargs: Any,
) -> dict[str, object]:
    """Compatibility entry point for smoke callers."""

    return run_vjepa2_smoke(
        config,
        pipeline_manifest=pipeline_manifest,
        output_dir=output_dir,
        split=split,
        **kwargs,
    )


__all__ = ["main", "run_vjepa2_smoke"]
