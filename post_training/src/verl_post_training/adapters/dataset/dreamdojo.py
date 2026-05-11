"""Stub dataset adapter for future DreamDojo support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DreamDojoDatasetAdapter:
    adapter_key = "dreamdojo"

    def prepare(
        self,
        pipeline_manifest: Any = None,
        output_dir: Path | None = None,
        split: str = "train",
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Path:
        raise NotImplementedError("DreamDojo dataset preparation is not implemented yet.")
