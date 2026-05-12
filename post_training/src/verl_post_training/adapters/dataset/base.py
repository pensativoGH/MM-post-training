"""Base contract for dataset adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class DatasetAdapter(Protocol):
    """Convert a pipeline-owned manifest into a backend-consumable asset."""

    adapter_key: str

    def prepare(
        self,
        pipeline_manifest: Any = None,
        output_dir: Path | None = None,
        split: str = "train",
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Path:
        """Prepare one output split and return the produced manifest path."""
