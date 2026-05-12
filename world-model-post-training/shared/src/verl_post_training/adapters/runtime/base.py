"""Base contract for runtime adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ...launch.dispatch import DispatchPlan
from ...launch.load_config import TaskConfig


class RuntimeAdapter(Protocol):
    """Run one repo-level inference or rollout task."""

    adapter_key: str

    def run(
        self,
        config: TaskConfig | DispatchPlan,
        *,
        pipeline_manifest: Any = None,
        output_dir: Path | None = None,
        split: str = "inference",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the task and return a normalized result payload."""
