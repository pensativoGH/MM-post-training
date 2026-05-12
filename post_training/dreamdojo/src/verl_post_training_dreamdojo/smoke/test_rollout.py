"""DreamDojo rollout smoke placeholder.

DreamDojo execution is intentionally deferred in M8. This module keeps the
smoke-test touch point importable while capability reporting marks the backend
unavailable.
"""

from __future__ import annotations

import pytest

from verl_post_training.capabilities import report_capabilities


def test_dreamdojo_rollout_deferred() -> None:
    record = report_capabilities()["dreamdojo"]
    if record["available"]:
        pytest.skip("DreamDojo execution is available in this environment.")
    assert record["reason"]
