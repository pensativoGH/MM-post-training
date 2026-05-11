"""Stub dataset adapter for future Wan support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class WanDatasetAdapter:
    adapter_key = "wan"

    def prepare(
        self,
        input_manifest: Path,
        output_dir: Path,
        split: str,
        config: dict[str, Any],
    ) -> Path:
        raise NotImplementedError("Wan dataset preparation is not implemented yet.")
