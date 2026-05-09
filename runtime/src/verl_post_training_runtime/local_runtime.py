from __future__ import annotations

import json
import urllib.error
import urllib.request


def resolve_local_runtime_spec(selector: str) -> dict[str, str]:
    normalized = selector.strip()
    if normalized == "thinking":
        return {
            "model_id": "Qwen/Qwen3-VL-8B-Thinking",
            "base_url": "http://127.0.0.1:8010/v1",
        }
    if normalized == "instruct":
        return {
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "base_url": "http://127.0.0.1:8011/v1",
        }
    if normalized == "thinking32b":
        return {
            "model_id": "Qwen/Qwen3-VL-32B-Thinking",
            "base_url": "http://127.0.0.1:8012/v1",
        }
    return {
        "model_id": normalized,
        "base_url": "http://127.0.0.1:8010/v1",
    }


def probe_openai_compatible_runtime(
    *,
    base_url: str,
    expected_model: str,
    api_key: str = "EMPTY",
    timeout_sec: float = 5.0,
) -> dict[str, object]:
    request = urllib.request.Request(
        url=base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "error": exc.read().decode("utf-8", errors="replace"),
            "base_url": base_url,
            "expected_model": expected_model,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": None,
            "error": str(exc),
            "base_url": base_url,
            "expected_model": expected_model,
        }

    models = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
    return {
        "ok": expected_model in models,
        "status": 200,
        "base_url": base_url,
        "expected_model": expected_model,
        "models": models,
    }
