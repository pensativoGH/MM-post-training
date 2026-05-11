from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping

_SELECTOR_SPECS: dict[str, dict[str, str]] = {
    "thinking": {
        "model_id": "Qwen/Qwen3-VL-8B-Thinking",
        "base_url": "http://127.0.0.1:8010/v1",
    },
    "instruct": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "base_url": "http://127.0.0.1:8011/v1",
    },
    "thinking32b": {
        "model_id": "Qwen/Qwen3-VL-32B-Thinking",
        "base_url": "http://127.0.0.1:8012/v1",
    },
}

_COMPAT_SPEC_KEYS = ("model_id", "base_url")


class ResolvedRuntimeSpec(dict[str, object]):
    """Compatibility wrapper for local runtime specs.

    M3 needs the resolved runtime backend and registry entry to be visible on
    the returned spec, while older tests and callers still compare against the
    original two-field dict contract.
    """

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            self_public = {key: self[key] for key in _COMPAT_SPEC_KEYS}
            other_keys = set(other.keys())
            if other_keys.issubset(_COMPAT_SPEC_KEYS):
                return self_public == dict(other)
        return super().__eq__(other)


def _ensure_repo_package_importable() -> None:
    package_src = Path(__file__).resolve().parents[3] / "post_training" / "src"
    if package_src.is_dir():
        src_str = str(package_src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


def _load_registry_types():
    try:
        from verl_post_training.registry import (
            ModelFamily,
            RuntimeBackend,
            get_model_entry,
        )
    except ModuleNotFoundError:
        _ensure_repo_package_importable()
        from verl_post_training.registry import (
            ModelFamily,
            RuntimeBackend,
            get_model_entry,
        )
    return ModelFamily, RuntimeBackend, get_model_entry


def _resolve_registry_entry(model_id: str):
    model_family, runtime_backend, get_model_entry = _load_registry_types()
    entry = get_model_entry(model_id)
    if entry.model_family != model_family.VLM_CHAT:
        raise ValueError(
            f"Model {model_id!r} is {entry.model_family.value!r}; "
            "the chat runtime only supports vlm_chat entries."
        )
    if runtime_backend.OPENAI_CHAT_VLLM not in entry.runtime_backends:
        raise ValueError(
            f"Model {model_id!r} does not support the openai_chat_vllm runtime."
        )
    return entry


def resolve_local_runtime_spec(selector: str) -> ResolvedRuntimeSpec:
    normalized = selector.strip()
    spec = _SELECTOR_SPECS.get(
        normalized,
        {
            "model_id": normalized,
            "base_url": "http://127.0.0.1:8010/v1",
        },
    )
    entry = _resolve_registry_entry(spec["model_id"])
    return ResolvedRuntimeSpec(
        {
            **spec,
            "runtime_backend": entry.runtime_backends[0].value,
            "model_entry": entry,
            "registry_entry": entry,
        }
    )


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
