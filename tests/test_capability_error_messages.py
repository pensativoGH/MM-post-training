"""M8 acceptance: unsupported training combinations fail before launch with
a capability error naming the requested model family, task type, and
backend; deferred DreamDojo support is reported as unavailable rather than
attempting a best-effort launch.

This file pins the fifth and sixth M8 acceptance criteria (quoted from
the approved plan):

    unsupported training combinations fail before launch with a capability
    error that names the requested model family, task type, and backend

    if DreamDojo runnable support is deferred, the milestone still lands
    only if the capability-reporting path explicitly marks DreamDojo
    execution as unavailable instead of attempting a best-effort launch

Concretely, these tests pin:

* a Qwen + ``chat_sft`` + ``vjepa2_native`` trainer combination fails with
  a typed compatibility error whose message names all three of:
  ``vlm_chat`` (family), ``chat_sft`` (task), and ``vjepa2_native``
  (backend)
* a world_model + ``chat_rl`` combination fails with a typed compatibility
  error whose message names all three of ``world_model``, ``chat_rl``, and
  the requested backend
* the failure path does *not* import the targeted backend module before
  the compatibility gate (no ``ImportError`` from a backend module)
* the repo exposes a capability-reporting surface
  (``capabilities()`` / ``report_capabilities()`` / ``is_available()`` /
  ``unavailable_backends()`` or similar) that callers can ask whether a
  particular ``trainer_backend`` / ``runtime_backend`` can actually run
  locally
* if DreamDojo execution is deferred, the capability surface reports
  DreamDojo as **unavailable** (e.g. ``available=False`` with a non-empty
  ``reason``) instead of silently advertising it as runnable
* a runtime ``run`` call against a deferred DreamDojo plan fails with a
  capability error (not an ``ImportError`` from a missing upstream module
  and not a silent best-effort launch)

Tests use ``tmp_path`` and ``monkeypatch`` so they stay deterministic and
self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dispatch_module():
    import verl_post_training.launch.dispatch as module

    return module


@pytest.fixture(scope="module")
def load_config_module():
    import verl_post_training.launch.load_config as module

    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_loader(load_config_module):
    for attr in ("load_config", "load_task_config", "load", "from_yaml"):
        fn = getattr(load_config_module, attr, None)
        if callable(fn):
            return fn
    pytest.fail(
        "verl_post_training.launch.load_config must expose a callable named "
        "one of: load_config, load_task_config, load, from_yaml."
    )


def _resolve_dispatcher(dispatch_module):
    for attr in ("resolve_dispatch", "dispatch", "resolve", "plan", "build_plan"):
        fn = getattr(dispatch_module, attr, None)
        if callable(fn):
            return fn
    pytest.fail(
        "verl_post_training.launch.dispatch must expose a callable named "
        "one of: resolve_dispatch, dispatch, resolve, plan, build_plan."
    )


def _write_yaml(tmp_path: Path, payload: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path


def _dispatch_from_yaml(
    load_config_module, dispatch_module, tmp_path, payload
):
    loader = _resolve_loader(load_config_module)
    dispatcher = _resolve_dispatcher(dispatch_module)
    config = loader(_write_yaml(tmp_path, payload))
    return dispatcher(config)


def _assert_typed_capability_error(
    excinfo: pytest.ExceptionInfo[BaseException],
    *,
    must_mention: tuple[str, ...],
) -> None:
    exc = excinfo.value
    assert type(exc) is not Exception, (
        "capability error must be a typed exception, not a bare Exception."
    )
    assert not isinstance(exc, ImportError), (
        "capability error must land before any backend import; got "
        f"ImportError: {exc!r}"
    )
    assert not isinstance(exc, ModuleNotFoundError), (
        "capability error must land before any backend import; got "
        f"ModuleNotFoundError: {exc!r}"
    )
    msg = str(exc)
    missing = [token for token in must_mention if token not in msg]
    assert not missing, (
        "capability error message must name the requested model family, "
        f"task type, and backend; missing {missing!r} from message: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 5: unsupported combinations fail with a capability error that
# names the family, task type, and backend
# ---------------------------------------------------------------------------


def test_unsupported_chat_sft_with_vjepa2_trainer_names_all_three(
    load_config_module, dispatch_module, tmp_path
):
    """A Qwen (``vlm_chat``) + ``chat_sft`` + ``vjepa2_native`` combination
    is unsupported. The dispatcher must name model family, task type, and
    backend in the failure message.
    """

    payload = {
        "task_type": "chat_sft",
        "model_id": "qwen3-vl-4b-instruct",
        "trainer_backend": "vjepa2_native",
        "dataset_adapter": "chat_sft",
        "input_manifest": "data/pipeline/train_manifest.jsonl",
        "output_dir": "outputs/post_training/capability_chat_sft_vjepa2",
        "launcher": {
            "kind": "torchrun",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }

    with pytest.raises(Exception) as excinfo:
        _dispatch_from_yaml(
            load_config_module, dispatch_module, tmp_path, payload
        )

    _assert_typed_capability_error(
        excinfo,
        must_mention=("vlm_chat", "chat_sft", "vjepa2_native"),
    )


def test_unsupported_world_model_chat_rl_names_all_three(
    load_config_module, dispatch_module, tmp_path
):
    """A ``world_model`` model id + ``chat_rl`` task + ``verl`` trainer
    combination is unsupported. The dispatcher must name model family,
    task type, and backend in the failure message.
    """

    payload = {
        "task_type": "chat_rl",
        "model_id": "dreamdojo-world-model-placeholder",
        "trainer_backend": "verl",
        "dataset_adapter": "chat_rl",
        "input_manifest": "data/pipeline/rl_manifest.jsonl",
        "output_dir": "outputs/post_training/capability_world_model_chat_rl",
        "launcher": {
            "kind": "python_module",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }

    with pytest.raises(Exception) as excinfo:
        _dispatch_from_yaml(
            load_config_module, dispatch_module, tmp_path, payload
        )

    _assert_typed_capability_error(
        excinfo,
        must_mention=("world_model", "chat_rl", "verl"),
    )


def test_unsupported_video_encoder_chat_sft_names_all_three(
    load_config_module, dispatch_module, tmp_path
):
    """A ``video_encoder`` model id + ``chat_sft`` + ``llamafactory``
    trainer combination is unsupported. The dispatcher must name model
    family, task type, and backend in the failure message.
    """

    payload = {
        "task_type": "chat_sft",
        "model_id": "vjepa2-video-encoder-placeholder",
        "trainer_backend": "llamafactory",
        "dataset_adapter": "chat_sft",
        "input_manifest": "data/pipeline/train_manifest.jsonl",
        "output_dir": (
            "outputs/post_training/capability_video_encoder_chat_sft"
        ),
        "launcher": {
            "kind": "torchrun",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }

    with pytest.raises(Exception) as excinfo:
        _dispatch_from_yaml(
            load_config_module, dispatch_module, tmp_path, payload
        )

    _assert_typed_capability_error(
        excinfo,
        must_mention=("video_encoder", "chat_sft", "llamafactory"),
    )


def test_unsupported_combinations_do_not_import_backend(
    load_config_module, dispatch_module, tmp_path, monkeypatch
):
    """Capability checks must run before backend modules are imported.

    We block the obvious backend candidates; the dispatcher must still
    fail cleanly with a typed compatibility error, not an ``ImportError``
    from a backend.
    """

    for name in ("vllm", "llamafactory", "verl", "torch"):
        monkeypatch.setitem(sys.modules, name, None)

    payload = {
        "task_type": "chat_sft",
        "model_id": "qwen3-vl-4b-instruct",
        "trainer_backend": "vjepa2_native",
        "dataset_adapter": "chat_sft",
        "input_manifest": "data/pipeline/train_manifest.jsonl",
        "output_dir": "outputs/post_training/capability_blocked_imports",
        "launcher": {
            "kind": "torchrun",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }

    with pytest.raises(Exception) as excinfo:
        _dispatch_from_yaml(
            load_config_module, dispatch_module, tmp_path, payload
        )

    assert not isinstance(
        excinfo.value, (ImportError, ModuleNotFoundError)
    ), (
        "capability check imported a backend module before rejecting an "
        f"incompatible combination; got: {excinfo.value!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 6: deferred DreamDojo execution is reported as unavailable
# ---------------------------------------------------------------------------


def _capability_for_dreamdojo() -> dict[str, Any] | None:
    """Find the capability report for DreamDojo, tolerating a few likely
    API shapes the writer might choose.

    Returns a dict-like report (with at least an ``available`` flag) or
    ``None`` if no such surface exists yet.
    """

    candidate_modules: list[Any] = []
    for module_path in (
        "verl_post_training.capabilities",
        "verl_post_training.launch.capabilities",
        "verl_post_training.launch",
        "verl_post_training",
    ):
        try:
            module = __import__(module_path, fromlist=["*"])
        except ImportError:
            continue
        candidate_modules.append(module)

    for module in candidate_modules:
        # Direct query by family/backend name.
        for attr in (
            "report_capabilities",
            "capabilities",
            "describe_capabilities",
            "get_capabilities",
            "list_capabilities",
        ):
            fn = getattr(module, attr, None)
            if not callable(fn):
                continue
            try:
                report = fn()
            except TypeError:
                # Maybe it wants a key.
                try:
                    report = fn("dreamdojo")
                except Exception:
                    continue
            except Exception:
                continue
            extracted = _extract_dreamdojo_record(report)
            if extracted is not None:
                return extracted

        for attr in (
            "is_available",
            "is_runnable",
            "backend_available",
            "trainer_available",
            "runtime_available",
        ):
            fn = getattr(module, attr, None)
            if callable(fn):
                try:
                    available = bool(fn("dreamdojo"))
                except Exception:
                    continue
                return {"available": available, "name": "dreamdojo"}

    return None


def _extract_dreamdojo_record(report: Any) -> dict[str, Any] | None:
    if report is None:
        return None
    # Mapping shape: {"dreamdojo": {"available": False, ...}, ...}
    if isinstance(report, dict):
        record = report.get("dreamdojo")
        if isinstance(record, dict):
            return record
        if isinstance(record, bool):
            return {"available": record, "name": "dreamdojo"}
        # Maybe nested under {"trainer": {...}, "runtime": {...}}.
        for value in report.values():
            sub = _extract_dreamdojo_record(value)
            if sub is not None:
                return sub
        return None
    # Iterable of report objects.
    if isinstance(report, (list, tuple, set)):
        for item in report:
            sub = _extract_dreamdojo_record(item)
            if sub is not None:
                return sub
        return None
    # Object with attributes.
    name = getattr(report, "name", None) or getattr(report, "key", None)
    if isinstance(name, str) and "dreamdojo" in name.lower():
        return {
            "available": getattr(report, "available", None),
            "reason": getattr(report, "reason", None) or getattr(report, "message", None),
            "name": name,
        }
    return None


def test_capability_surface_reports_dreamdojo_status():
    """The capability-reporting path must exist and have a determinate
    answer for DreamDojo (available or unavailable, but not silent).
    """

    record = _capability_for_dreamdojo()
    if record is None:
        pytest.fail(
            "M8 requires a capability-reporting surface that callers can "
            "ask whether each backend is runnable. Expected one of "
            "`verl_post_training.capabilities`, "
            "`verl_post_training.launch.capabilities`, or a "
            "`capabilities()` / `report_capabilities()` function on "
            "`verl_post_training.launch` that returns a dict keyed by "
            "backend name. None was found."
        )
    assert "available" in record, (
        "DreamDojo capability record must contain an explicit `available` "
        f"flag; got {record!r}"
    )
    assert record["available"] is not None, (
        "DreamDojo capability record must declare a definite "
        f"True/False for `available`; got {record!r}"
    )


def test_deferred_dreamdojo_is_reported_as_unavailable_with_reason():
    """If DreamDojo execution is deferred (the M8 deferral clause), the
    capability surface must report it as unavailable *with a reason*
    rather than silently advertising it as runnable.

    The plan permits two outcomes:
    * DreamDojo is fully implemented and the capability reports
      ``available=True`` with no reason. In that case this test
      passes trivially.
    * DreamDojo is deferred and the capability reports
      ``available=False`` along with a non-empty reason/message that
      mentions DreamDojo. This test enforces the latter shape so a
      writer cannot silently land an unavailable backend that pretends
      to be runnable.
    """

    record = _capability_for_dreamdojo()
    if record is None:
        pytest.fail(
            "M8 capability-reporting surface missing — see "
            "test_capability_surface_reports_dreamdojo_status."
        )
    if record.get("available") is True:
        # DreamDojo is fully runnable in this build; nothing to assert.
        return

    reason_candidates = (
        record.get("reason"),
        record.get("message"),
        record.get("status_message"),
        record.get("note"),
        record.get("detail"),
    )
    reason = next(
        (r for r in reason_candidates if isinstance(r, str) and r.strip()),
        None,
    )
    assert reason, (
        "Deferred DreamDojo support must report a non-empty reason "
        "explaining why execution is unavailable; got record: "
        f"{record!r}"
    )
    assert "dreamdojo" in reason.lower(), (
        "Deferred DreamDojo capability reason should name DreamDojo so "
        f"callers can pinpoint the unavailable backend; got: {reason!r}"
    )


def test_runtime_run_against_deferred_dreamdojo_fails_with_capability_error(
    load_config_module, dispatch_module, tmp_path, monkeypatch
):
    """If DreamDojo execution is deferred, calling ``run`` on the
    world-model runtime adapter (or its registered ``run_runtime``)
    against a DreamDojo plan must fail with a typed capability error
    rather than:

    * silently launching a best-effort placeholder, or
    * raising an ``ImportError`` from a missing upstream module.

    If DreamDojo is fully runnable, this test is skipped because the
    capability gate is not engaged.
    """

    record = _capability_for_dreamdojo()
    if record is None:
        pytest.fail(
            "M8 capability-reporting surface missing — see "
            "test_capability_surface_reports_dreamdojo_status."
        )
    if record.get("available") is True:
        pytest.skip(
            "DreamDojo is reported as available; capability deferral gate "
            "is not exercised."
        )

    # Block any plausible upstream import so a best-effort launch would
    # produce an ImportError (which we will count as a regression).
    for name in (
        "dreamdojo",
        "dreamdojo_core",
        "dreamdojo.runtime",
    ):
        monkeypatch.setitem(sys.modules, name, None)

    payload = {
        "task_type": "world_model_rollout",
        "model_id": "dreamdojo-world-model-placeholder",
        "runtime_backend": "dreamdojo",
        "dataset_adapter": "dreamdojo",
        "input_manifest": "data/pipeline/dreamdojo_trajectory_manifest.jsonl",
        "output_dir": str(tmp_path / "out"),
        "launcher": {
            "kind": "python_module",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }
    loader = _resolve_loader(load_config_module)
    cfg = loader(_write_yaml(tmp_path, payload))

    runner = None
    for module_path, attr in (
        ("verl_post_training.adapters.runtime", "run_runtime"),
        ("verl_post_training.adapters.runtime", "dispatch_runtime"),
        ("verl_post_training.launch.dispatch", "run_runtime"),
    ):
        try:
            module = __import__(module_path, fromlist=[attr])
        except ImportError:
            continue
        fn = getattr(module, attr, None)
        if callable(fn):
            runner = fn
            break

    if runner is None:
        pytest.skip(
            "No runtime `run` entry point exposed yet; capability gate at "
            "execution time cannot be exercised."
        )

    with pytest.raises(Exception) as excinfo:
        runner(cfg)

    assert not isinstance(
        excinfo.value, (ImportError, ModuleNotFoundError)
    ), (
        "Deferred DreamDojo runtime must report unavailability via a "
        "capability error, not by attempting a best-effort launch that "
        f"raises ImportError. Got: {excinfo.value!r}"
    )
    assert type(excinfo.value) is not Exception, (
        "Deferred DreamDojo runtime must raise a typed capability error "
        "so callers can distinguish unavailability from unrelated bugs."
    )
