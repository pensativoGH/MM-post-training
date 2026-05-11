"""Stub dataset adapter for future V-JEPA2 support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class VJEPA2DatasetAdapter:
    adapter_key = "vjepa2"

    def prepare(
        self,
        pipeline_manifest: Any = None,
        output_dir: Path | None = None,
        split: str = "train",
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Path:
        raise NotImplementedError("V-JEPA2 dataset preparation is not implemented yet.")
