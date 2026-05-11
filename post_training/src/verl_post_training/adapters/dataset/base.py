"""Base contract for dataset adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class DatasetAdapter(Protocol):
    """Convert a pipeline-owned manifest into a backend-consumable asset."""

    adapter_key: str

    def prepare(
        self,
        input_manifest: Path,
        output_dir: Path,
        split: str,
        config: dict[str, Any],
    ) -> Path:
        """Prepare one output split and return the produced manifest path."""
