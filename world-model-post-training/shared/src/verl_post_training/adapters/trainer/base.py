"""Base contract for trainer adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ...launch.dispatch import DispatchPlan
from ...launch.load_config import TaskConfig


class TrainerAdapter(Protocol):
    """Prepare or launch one repo-level training task."""

    adapter_key: str

    def run(
        self,
        config: TaskConfig | DispatchPlan,
        *,
        output_dir: Path | None = None,
        split: str = "train",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare the task and return a normalized launch payload."""
