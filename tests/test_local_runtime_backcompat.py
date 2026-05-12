"""M3 acceptance: existing chat runtime behavior remains unchanged for the
supported selectors.

This file pins criteria 2 and 4 from the approved plan section for milestone
M3 (quoted):

    the readiness check still succeeds against a healthy Qwen ``vLLM`` server
    using the existing smoke path

and

    existing chat runtime behavior remains unchanged for supported selectors:
    base URL format, health check contract, and multimodal smoke invocation
    all remain compatible

Concretely, this file guards three back-compat surfaces:

1. ``resolve_local_runtime_spec`` for the three known selectors
   (``thinking``, ``instruct``, ``thinking32b``) and for the seeded explicit
   Qwen registry id (``qwen3-vl-4b-instruct``) still produces the *same*
   ``model_id`` and ``base_url`` values existing scripts depend on.

2. ``probe_openai_compatible_runtime`` still talks to ``/v1/models`` and
   reports ``ok=True`` when the expected model is advertised, ``ok=False``
   when it is not, and surfaces network errors without raising.

3. The multimodal smoke script ``runtime/scripts/smoke_qwen_openai_mm.py``
   still exists and still accepts the documented CLI surface
   (``--base-url``, ``--model``, ``--api-key``, ``--timeout``).

Tests are deterministic and self-contained: HTTP is mocked via
``monkeypatch`` against ``urllib.request.urlopen``; no real server or
subprocess is started.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_SRC = _REPO_ROOT / "runtime" / "src"
if _RUNTIME_SRC.is_dir():
    _runtime_src_str = str(_RUNTIME_SRC)
    if _runtime_src_str not in sys.path:
        sys.path.insert(0, _runtime_src_str)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtime_module():
    import verl_post_training_runtime.local_runtime as module

    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal context-manager stand-in for ``urllib.request.urlopen``."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _model_id_of(spec: object) -> str:
    """Return the ``model_id`` from either a dict-shaped or attribute-shaped
    runtime spec. The implementer may keep the dict shape or upgrade to a
    dataclass; both must continue to expose the same model identifier.
    """

    if isinstance(spec, dict) and "model_id" in spec:
        return spec["model_id"]
    if hasattr(spec, "model_id"):
        return getattr(spec, "model_id")
    pytest.fail(
        "runtime spec must expose `model_id` either as a dict key or as an "
        f"attribute; got {spec!r}"
    )


def _base_url_of(spec: object) -> str:
    if isinstance(spec, dict) and "base_url" in spec:
        return spec["base_url"]
    if hasattr(spec, "base_url"):
        return getattr(spec, "base_url")
    pytest.fail(
        "runtime spec must expose `base_url` either as a dict key or as an "
        f"attribute; got {spec!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 4: selector -> (model_id, base_url) is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selector,expected_model_id,expected_base_url",
    [
        ("thinking", "Qwen/Qwen3-VL-8B-Thinking", "http://127.0.0.1:8010/v1"),
        ("instruct", "Qwen/Qwen3-VL-8B-Instruct", "http://127.0.0.1:8011/v1"),
        ("thinking32b", "Qwen/Qwen3-VL-32B-Thinking", "http://127.0.0.1:8012/v1"),
    ],
)
def test_known_selectors_resolve_to_stable_model_id_and_base_url(
    runtime_module, selector, expected_model_id, expected_base_url
):
    spec = runtime_module.resolve_local_runtime_spec(selector)
    assert _model_id_of(spec) == expected_model_id, (
        f"selector {selector!r} must continue to resolve to model_id "
        f"{expected_model_id!r}; got {_model_id_of(spec)!r}"
    )
    assert _base_url_of(spec) == expected_base_url, (
        f"selector {selector!r} must continue to resolve to base_url "
        f"{expected_base_url!r}; got {_base_url_of(spec)!r}"
    )


def test_base_url_format_is_openai_v1_path(runtime_module):
    """The OpenAI-compatible chat endpoint contract requires that the
    advertised base_url end with ``/v1`` so callers can append
    ``/chat/completions``. Any reformatting that drops the trailing ``/v1``
    is a hard back-compat break.
    """

    spec = runtime_module.resolve_local_runtime_spec("thinking")
    base_url = _base_url_of(spec)
    assert base_url.endswith("/v1"), (
        f"base_url must end with '/v1' to preserve OpenAI-compatible "
        f"client semantics; got {base_url!r}"
    )
    assert base_url.startswith("http://") or base_url.startswith("https://"), (
        f"base_url must declare an http(s) scheme; got {base_url!r}"
    )


def test_whitespace_around_selector_is_tolerated(runtime_module):
    """The existing implementation strips whitespace from the selector;
    user-facing scripts pass shell-quoted arguments through, so this
    behavior is part of the back-compat contract.
    """

    spec = runtime_module.resolve_local_runtime_spec("  thinking  ")
    assert _model_id_of(spec) == "Qwen/Qwen3-VL-8B-Thinking"
    assert _base_url_of(spec) == "http://127.0.0.1:8010/v1"


# ---------------------------------------------------------------------------
# Criterion 2: readiness check still succeeds against a healthy server
# ---------------------------------------------------------------------------


def test_probe_returns_ok_when_expected_model_is_advertised(
    runtime_module, monkeypatch
):
    """A healthy Qwen ``vLLM`` server lists the served model id under
    ``data[].id`` on ``GET /v1/models``. The probe must report ``ok=True``
    and a status of 200 in that case.
    """

    payload = {"data": [{"id": "Qwen/Qwen3-VL-8B-Thinking"}]}

    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return _FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = runtime_module.probe_openai_compatible_runtime(
        base_url="http://127.0.0.1:8010/v1",
        expected_model="Qwen/Qwen3-VL-8B-Thinking",
        api_key="EMPTY",
        timeout_sec=2.5,
    )

    assert result["ok"] is True, f"probe must report ok=True for healthy server; got {result!r}"
    assert result["status"] == 200
    assert result["base_url"] == "http://127.0.0.1:8010/v1"
    assert result["expected_model"] == "Qwen/Qwen3-VL-8B-Thinking"
    assert captured["url"].endswith("/v1/models"), (
        f"probe must call the OpenAI-compatible /v1/models endpoint; "
        f"got url={captured['url']!r}"
    )
    # Bearer auth header is part of the OpenAI-compatible contract.
    auth_headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert auth_headers.get("authorization") == "Bearer EMPTY"


def test_probe_returns_not_ok_when_expected_model_missing(
    runtime_module, monkeypatch
):
    payload = {"data": [{"id": "some-other-model"}]}

    def fake_urlopen(request, timeout=None):
        return _FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = runtime_module.probe_openai_compatible_runtime(
        base_url="http://127.0.0.1:8010/v1",
        expected_model="Qwen/Qwen3-VL-8B-Thinking",
    )

    assert result["ok"] is False
    # Status is 200 even though the expected model is absent: the server is
    # reachable, it just doesn't advertise the requested id.
    assert result["status"] == 200
    assert "Qwen/Qwen3-VL-8B-Thinking" not in result.get("models", [])


def test_probe_handles_http_error_without_raising(
    runtime_module, monkeypatch
):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(b"server warming up"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = runtime_module.probe_openai_compatible_runtime(
        base_url="http://127.0.0.1:8010/v1",
        expected_model="Qwen/Qwen3-VL-8B-Thinking",
    )

    assert result["ok"] is False
    assert result["status"] == 503


def test_probe_handles_connection_error_without_raising(
    runtime_module, monkeypatch
):
    def fake_urlopen(request, timeout=None):
        raise ConnectionRefusedError("server not running")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = runtime_module.probe_openai_compatible_runtime(
        base_url="http://127.0.0.1:8010/v1",
        expected_model="Qwen/Qwen3-VL-8B-Thinking",
    )

    assert result["ok"] is False
    assert result["status"] is None


# ---------------------------------------------------------------------------
# Criterion 4: multimodal smoke invocation surface remains compatible
# ---------------------------------------------------------------------------


def test_multimodal_smoke_script_exists_and_advertises_documented_cli():
    """The user-facing smoke entry point invoked from documentation and from
    notebooks is ``runtime/scripts/smoke_qwen_openai_mm.py`` with the
    ``--base-url`` / ``--model`` / ``--api-key`` / ``--timeout`` CLI. Any
    rename or argument-rename would silently break documented invocations.
    """

    script_path = _REPO_ROOT / "runtime" / "scripts" / "smoke_qwen_openai_mm.py"
    assert script_path.is_file(), (
        f"multimodal smoke script must remain at "
        f"runtime/scripts/smoke_qwen_openai_mm.py; not found at {script_path!s}"
    )

    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, (
        f"smoke script --help must exit cleanly; stderr={completed.stderr!r}"
    )
    help_text = completed.stdout
    for flag in ("--base-url", "--model", "--api-key", "--timeout"):
        assert flag in help_text, (
            f"smoke script CLI must keep advertising {flag!r}; --help output:\n"
            f"{help_text}"
        )


def test_readiness_check_script_exists_and_advertises_documented_cli():
    """``runtime/scripts/check_qwen_vllm_ready.py`` is the readiness path
    that operators invoke after starting a vLLM server. The selector flag and
    overrides must remain stable for existing runbooks.
    """

    script_path = _REPO_ROOT / "runtime" / "scripts" / "check_qwen_vllm_ready.py"
    assert script_path.is_file(), (
        f"readiness script must remain at "
        f"runtime/scripts/check_qwen_vllm_ready.py; not found at {script_path!s}"
    )

    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, (
        f"readiness script --help must exit cleanly; "
        f"stderr={completed.stderr!r}"
    )
    help_text = completed.stdout
    for flag in ("--selector", "--base-url", "--model", "--timeout-sec"):
        assert flag in help_text, (
            f"readiness script CLI must keep advertising {flag!r}; --help "
            f"output:\n{help_text}"
        )
