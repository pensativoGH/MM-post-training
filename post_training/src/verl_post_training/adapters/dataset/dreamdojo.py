"""Stub dataset adapter for future DreamDojo support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DreamDojoDatasetAdapter:
    adapter_key = "dreamdojo"

    def prepare(
        self,
        input_manifest: Path,
        output_dir: Path,
        split: str,
        config: dict[str, Any],
    ) -> Path:
        raise NotImplementedError("DreamDojo dataset preparation is not implemented yet.")
