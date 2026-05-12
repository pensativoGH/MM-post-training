"""Stub dataset adapter for future Wan support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from verl_post_training.bootstrap.third_party import discover_upstream_root


class WanDatasetAdapter:
    adapter_key = "wan"
    upstream_family = "wan22"

    def discover_upstream_root(self, *, manifest_path: Path | None = None) -> Path:
        """Resolve the pinned repo-owned checkout path for Wan2.2."""

        return discover_upstream_root(self.upstream_family, manifest_path=manifest_path)

    def prepare(
        self,
        pipeline_manifest: Any = None,
        output_dir: Path | None = None,
        split: str = "train",
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Path:
        raise NotImplementedError("Wan dataset preparation is not implemented yet.")
