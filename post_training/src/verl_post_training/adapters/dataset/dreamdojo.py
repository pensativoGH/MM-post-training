"""Stub dataset adapter for future DreamDojo support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from verl_post_training.bootstrap.third_party import discover_upstream_root


class DreamDojoDatasetAdapter:
    adapter_key = "dreamdojo"
    upstream_family = "dreamdojo"

    def discover_upstream_root(self, *, manifest_path: Path | None = None) -> Path:
        """Resolve the pinned repo-owned checkout path for DreamDojo."""

        return discover_upstream_root(self.upstream_family, manifest_path=manifest_path)

    def prepare(
        self,
        pipeline_manifest: Any = None,
        output_dir: Path | None = None,
        split: str = "train",
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Path:
        raise NotImplementedError("DreamDojo dataset preparation is not implemented yet.")
