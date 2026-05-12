"""Stub dataset adapter for future Wan support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class WanDatasetAdapter:
    adapter_key = "wan"

    def prepare(
        self,
        pipeline_manifest: Any = None,
        output_dir: Path | None = None,
        split: str = "train",
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Path:
        raise NotImplementedError("Wan dataset preparation is not implemented yet.")
