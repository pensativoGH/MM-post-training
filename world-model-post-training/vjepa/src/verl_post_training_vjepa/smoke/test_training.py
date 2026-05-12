"""Smoke helpers for repo-local V-JEPA2 training dry-run coverage."""

from __future__ import annotations

from pathlib import Path

from verl_post_training.adapters.trainer import run_trainer
from verl_post_training.bootstrap.third_party import discover_upstream_root
from verl_post_training.launch.load_config import TaskConfig


def run_vjepa2_training_smoke(
    config: TaskConfig,
    *,
    output_dir: str | Path | None = None,
    split: str = "smoke_train",
    **kwargs: object,
) -> dict[str, object]:
    """Build the V-JEPA2 training launch record from repo-level config."""

    discover_upstream_root("vjepa2")
    return run_trainer(
        config,
        output_dir=Path(output_dir) if output_dir is not None else None,
        split=split,
        **kwargs,
    )


def main(
    config: TaskConfig,
    *,
    output_dir: str | Path | None = None,
    split: str = "smoke_train",
    **kwargs: object,
) -> dict[str, object]:
    """Compatibility entry point for smoke callers."""

    return run_vjepa2_training_smoke(
        config,
        output_dir=output_dir,
        split=split,
        **kwargs,
    )


__all__ = ["main", "run_vjepa2_training_smoke"]
