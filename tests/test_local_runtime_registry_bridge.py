from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_SRC = _REPO_ROOT / "runtime" / "src"

if str(_RUNTIME_SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_RUNTIME_SRC))

from verl_post_training_runtime.local_runtime import (  # noqa: E402
    probe_openai_compatible_runtime,
    resolve_local_runtime_spec,
)


def test_qwen_selector_keeps_existing_base_url_and_model_id():
    spec = resolve_local_runtime_spec("instruct")
    assert spec == {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "base_url": "http://127.0.0.1:8011/v1",
    }


def test_explicit_qwen_model_id_resolves_through_registry():
    spec = resolve_local_runtime_spec("Qwen/Qwen3-VL-8B-Thinking")
    assert spec == {
        "model_id": "Qwen/Qwen3-VL-8B-Thinking",
        "base_url": "http://127.0.0.1:8010/v1",
    }


def test_non_chat_model_is_rejected_before_startup():
    with pytest.raises(ValueError, match="chat runtime only supports vlm_chat"):
        resolve_local_runtime_spec("vjepa2-video-encoder-placeholder")


def test_probe_openai_compatible_runtime_accepts_healthy_qwen_server():
    payload = json.dumps(
        {"data": [{"id": "Qwen/Qwen3-VL-8B-Instruct"}]}
    ).encode("utf-8")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return payload

    with patch("urllib.request.urlopen", return_value=_Response()) as mocked:
        result = probe_openai_compatible_runtime(
            base_url="http://127.0.0.1:8011/v1",
            expected_model="Qwen/Qwen3-VL-8B-Instruct",
        )

    mocked.assert_called_once()
    assert result["ok"] is True
    assert result["base_url"] == "http://127.0.0.1:8011/v1"
    assert result["expected_model"] == "Qwen/Qwen3-VL-8B-Instruct"
