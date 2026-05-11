"""Smoke helpers for repo-local Wan generation coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..adapters.runtime import run_runtime
from ..bootstrap.third_party import discover_upstream_root
from ..launch.load_config import TaskConfig


def run_wan_generation_smoke(
    config: TaskConfig,
    *,
    pipeline_manifest: Any = None,
    output_dir: str | Path | None = None,
    split: str = "smoke",
    **kwargs: Any,
) -> dict[str, object]:
    """Execute the repo-owned Wan smoke path without shelling into third_party/."""

    discover_upstream_root("wan22")
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

    return run_wan_generation_smoke(
        config,
        pipeline_manifest=pipeline_manifest,
        output_dir=output_dir,
        split=split,
        **kwargs,
    )


run_wan_smoke = run_wan_generation_smoke


__all__ = ["main", "run_wan_generation_smoke", "run_wan_smoke"]
