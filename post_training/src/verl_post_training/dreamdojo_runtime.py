"""Repo-local DreamDojo runtime wrapper.

DreamDojo execution is intentionally deferred until the upstream integration
is approved for this repo's local-first envelope. Callers still get a
machine-readable capability result instead of an attempted best-effort launch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_dreamdojo_rollout(
    *,
    model_id: str,
    task_type: str,
    input_manifest: Path,
    output_dir: Path,
    backend_config: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "status": "unavailable",
        "available": False,
        "capability": {
            "available": False,
            "reason": "DreamDojo execution is deferred pending local-first integration approval.",
        },
        "model_id": model_id,
        "task_type": task_type,
        "input_manifest": str(input_manifest),
        "output_dir": str(output_dir),
        "backend_config": dict(backend_config or {}),
    }

