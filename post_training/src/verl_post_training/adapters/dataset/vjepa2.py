"""Stub dataset adapter for future V-JEPA2 support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class VJEPA2DatasetAdapter:
    adapter_key = "vjepa2"

    def prepare(
        self,
        input_manifest: Path,
        output_dir: Path,
        split: str,
        config: dict[str, Any],
    ) -> Path:
        raise NotImplementedError("V-JEPA2 dataset preparation is not implemented yet.")
