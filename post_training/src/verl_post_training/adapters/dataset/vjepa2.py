"""Stub dataset adapter for future V-JEPA2 support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from verl_post_training.bootstrap.third_party import discover_upstream_root


class VJEPA2DatasetAdapter:
    adapter_key = "vjepa2"
    upstream_family = "vjepa2"

    def discover_upstream_root(self, *, manifest_path: Path | None = None) -> Path:
        """Resolve the pinned repo-owned checkout path for V-JEPA2."""

        return discover_upstream_root(self.upstream_family, manifest_path=manifest_path)

    def prepare(
        self,
        pipeline_manifest: Any = None,
        output_dir: Path | None = None,
        split: str = "train",
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Path:
        raise NotImplementedError("V-JEPA2 dataset preparation is not implemented yet.")
