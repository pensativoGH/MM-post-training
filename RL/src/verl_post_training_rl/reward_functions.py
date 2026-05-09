from __future__ import annotations

import json
from typing import Any


def _extract_next_step_id(payload: dict[str, Any]) -> int | None:
    value = payload.get("demonstrated_next_graph_step_id")
    return value if isinstance(value, int) or value is None else None


def _coerce_ground_truth(ground_truth: Any) -> dict[str, Any] | None:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    del data_source, extra_info

    gold = _coerce_ground_truth(ground_truth)
    if gold is None:
        return 0.0

    try:
        pred = json.loads(solution_str)
    except json.JSONDecodeError:
        return 0.0

    if not isinstance(pred, dict):
        return 0.0

    return 1.0 if _extract_next_step_id(pred) == _extract_next_step_id(gold) else 0.0
