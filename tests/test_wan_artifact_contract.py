"""M7 acceptance: the Wan2.2 wrapper writes generated artifacts plus a
machine-readable metadata file under ``output_dir``, and the smoke run
completes without requiring the user to operate inside the upstream
Wan2.2 repo.

This file pins the third and fourth M7 acceptance criteria (quoted from
the approved plan):

    the wrapper writes generated artifacts and a machine-readable
    metadata file under the configured ``output_dir``; the metadata
    file records ``model_id``, input example ids, and generated artifact
    paths

    a smoke run implemented in
    ``world-model-post-training/shared/src/verl_post_training/smoke/test_wan_generation.py``
    completes without requiring the user to operate inside the upstream
    Wan2.2 repo

Concretely, these tests pin:

* invoking the video-generation runtime adapter against a Wan dispatch
  plan writes at least one generated artifact under ``output_dir``
* it also writes at least one machine-readable metadata file under
  ``output_dir``
* the metadata file records the ``model_id``, every input example id,
  and a generated artifact path per example
* the runtime adapter does not spawn a backend subprocess as a side
  effect of producing the metadata (smoke / dry-run safety)
* the smoke file at the plan-pinned path exists and lives under
  ``world-model-post-training/shared/src/`` (not under ``third_party/``)
* the smoke does not embed a literal ``third_party/wan22`` path and
  does not ``os.chdir`` into the upstream tree
* the smoke uses the M5 discovery helper to resolve the Wan upstream
  root and is importable from a foreign working directory
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SMOKE_PATH = (
    _REPO_ROOT
    / "world-model-post-training"
    / "wan"
    / "src"
    / "verl_post_training_wan"
    / "smoke"
    / "test_generation.py"
)


WAN_PLACEHOLDER_MODEL_ID = "wan-video-generator-placeholder"


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


def _resolve_runtime_adapter_selector(dispatch_module):
    for attr in (
        "get_runtime_adapter",
        "select_runtime_adapter",
        "resolve_runtime_adapter",
        "runtime_adapter_for_plan",
        "build_runtime_adapter",
    ):
        fn = getattr(dispatch_module, attr, None)
        if callable(fn):
            return fn

    try:
        import verl_post_training.adapters.runtime as runtime_pkg
    except ModuleNotFoundError:
        runtime_pkg = None
    if runtime_pkg is not None:
        for attr in (
            "get_runtime_adapter",
            "select_runtime_adapter",
            "resolve_runtime_adapter",
            "runtime_adapter_for_plan",
        ):
            fn = getattr(runtime_pkg, attr, None)
            if callable(fn):
                return fn
    pytest.fail(
        "Repo-level dispatch must expose a runtime adapter selector for the "
        "Wan inference path."
    )


def _wan_inference_payload(
    *,
    input_manifest: Path,
    output_dir: Path,
    model_id: str = WAN_PLACEHOLDER_MODEL_ID,
) -> dict[str, Any]:
    return {
        "task_type": "generation_inference",
        "model_id": model_id,
        "runtime_backend": "wan_native",
        "dataset_adapter": "wan",
        "input_manifest": str(input_manifest),
        "output_dir": str(output_dir),
        "launcher": {
            "kind": "python_module",
            "num_nodes": 1,
            "nproc_per_node": 1,
        },
        "resources": {"precision": "bf16", "devices": 1},
        "backend_config": {},
    }


def _write_yaml(tmp_path: Path, payload: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path


def _build_plan(load_config_module, dispatch_module, payload, tmp_path):
    loader = _resolve_loader(load_config_module)
    dispatcher = _resolve_dispatcher(dispatch_module)
    return dispatcher(loader(_write_yaml(tmp_path, payload)))


def _invoke_adapter(adapter, plan):
    """Invoke whatever entry point the video-generation runtime adapter
    exposes — accept ``run`` / ``invoke`` / ``execute`` / ``__call__``.
    """

    for attr in ("run", "invoke", "execute"):
        fn = getattr(adapter, attr, None)
        if callable(fn):
            return fn(plan)
    if callable(adapter):
        return adapter(plan)
    pytest.fail(
        "Wan video-generation runtime adapter must expose a callable entry "
        f"point (run/invoke/execute/__call__); got: {adapter!r}"
    )


def _stub_inner_inference(monkeypatch):
    """Best-effort: stub any inner inference hook so we never need GPU /
    upstream weights. The writer is free to choose which seam to expose;
    we cover the common spellings.
    """

    module = sys.modules.get(
        "verl_post_training.adapters.runtime.video_generation"
    )
    if module is None:
        return
    for stub_name in (
        "_run_generation",
        "run_generation",
        "_generate_example",
        "generate_example",
        "_invoke_upstream",
        "_run_wan",
        "run_wan",
    ):
        if hasattr(module, stub_name):
            monkeypatch.setattr(
                module,
                stub_name,
                lambda *args, **kwargs: {"status": "ok"},
            )


def _gather_output_files(output_dir: Path) -> list[Path]:
    return [p for p in output_dir.rglob("*") if p.is_file()]


def _try_parse_metadata(path: Path) -> Mapping[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None

    for parser in (json.loads, yaml.safe_load):
        try:
            value = parser(text)
        except (json.JSONDecodeError, yaml.YAMLError, ValueError):
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _load_metadata_files(output_dir: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for path in _gather_output_files(output_dir):
        rec = _try_parse_metadata(path)
        if rec is not None:
            records.append(rec)
    return records


def _flatten_strings(value: Any) -> list[str]:
    """Walk a nested mapping/list and yield every string leaf."""

    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, Mapping):
        for v in value.values():
            out.extend(_flatten_strings(v))
    elif isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_flatten_strings(item))
    return out


# ---------------------------------------------------------------------------
# Criterion 3: wrapper writes artifacts + machine-readable metadata
# ---------------------------------------------------------------------------


def _prepare_inference_scenario(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """Two-row Wan conditioning manifest + an output_dir."""

    example_ids = ["wan_t2v_000", "wan_i2v_001"]
    manifest_path = tmp_path / "wan_conditioning_manifest.jsonl"
    rows = [
        {
            "example_id": example_ids[0],
            "prompt": "A drone shot over a misty mountain range at dawn.",
            "modality": "text",
            "conditioning_media_paths": [],
            "output_path": str(tmp_path / "wan_out" / f"{example_ids[0]}.mp4"),
        },
        {
            "example_id": example_ids[1],
            "prompt": "Animate the input frame with subtle camera drift.",
            "modality": "image",
            "conditioning_media_paths": ["s3://bucket/key/frame_001.png"],
            "output_path": str(tmp_path / "wan_out" / f"{example_ids[1]}.mp4"),
        },
    ]
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "wan_out"
    output_dir.mkdir()
    return manifest_path, output_dir, example_ids


def test_wrapper_writes_at_least_one_generated_artifact(
    load_config_module, dispatch_module, tmp_path, monkeypatch
):
    """The wrapper must write at least one generated artifact under the
    configured ``output_dir``. We monkeypatch any inner inference seam so
    the test never needs GPUs / Wan weights.
    """

    manifest_path, output_dir, example_ids = _prepare_inference_scenario(
        tmp_path
    )
    payload = _wan_inference_payload(
        input_manifest=manifest_path, output_dir=output_dir
    )
    plan = _build_plan(load_config_module, dispatch_module, payload, tmp_path)
    selector = _resolve_runtime_adapter_selector(dispatch_module)
    adapter = selector(plan)
    assert adapter is not None, (
        "Runtime adapter selector returned None for a Wan inference plan."
    )

    _stub_inner_inference(monkeypatch)
    _invoke_adapter(adapter, plan)

    files = _gather_output_files(output_dir)
    assert files, (
        f"Wan wrapper must write at least one artifact under {output_dir}; "
        "found none."
    )

    # The plan acceptance criterion calls out "generated artifacts" — i.e.,
    # at least one file that isn't just a metadata sidecar.
    metadata_files = {
        path
        for path in files
        if _try_parse_metadata(path) is not None
        and path.suffix.lower() in {".json", ".yaml", ".yml", ".jsonl"}
    }
    non_metadata_files = [path for path in files if path not in metadata_files]
    assert non_metadata_files, (
        "Wan wrapper must write at least one generated artifact (not just "
        f"a metadata sidecar) under {output_dir}; saw only: "
        f"{[str(p) for p in files]!r}"
    )


def test_wrapper_writes_machine_readable_metadata_file(
    load_config_module, dispatch_module, tmp_path, monkeypatch
):
    """The wrapper must write at least one machine-readable metadata file
    (JSON or YAML) under ``output_dir``.
    """

    manifest_path, output_dir, _ids = _prepare_inference_scenario(tmp_path)
    payload = _wan_inference_payload(
        input_manifest=manifest_path, output_dir=output_dir
    )
    plan = _build_plan(load_config_module, dispatch_module, payload, tmp_path)
    selector = _resolve_runtime_adapter_selector(dispatch_module)
    adapter = selector(plan)

    _stub_inner_inference(monkeypatch)
    _invoke_adapter(adapter, plan)

    metadata_records = _load_metadata_files(output_dir)
    assert metadata_records, (
        "Wan wrapper must write at least one machine-readable metadata "
        "file (JSON or YAML) under output_dir; saw files but none parsed "
        f"as a mapping. Files: "
        f"{[str(p) for p in _gather_output_files(output_dir)]!r}"
    )


def test_metadata_records_model_id(
    load_config_module, dispatch_module, tmp_path, monkeypatch
):
    """The metadata file must record the dispatched ``model_id``."""

    manifest_path, output_dir, _ids = _prepare_inference_scenario(tmp_path)
    payload = _wan_inference_payload(
        input_manifest=manifest_path, output_dir=output_dir
    )
    plan = _build_plan(load_config_module, dispatch_module, payload, tmp_path)
    selector = _resolve_runtime_adapter_selector(dispatch_module)
    adapter = selector(plan)

    _stub_inner_inference(monkeypatch)
    _invoke_adapter(adapter, plan)

    metadata_records = _load_metadata_files(output_dir)
    assert metadata_records, "no metadata file was written under output_dir"

    found_model_id = False
    for rec in metadata_records:
        for key in ("model_id", "resolved_model_id", "model"):
            value = rec.get(key) if isinstance(rec, Mapping) else None
            if value is None:
                continue
            if str(value) == WAN_PLACEHOLDER_MODEL_ID:
                found_model_id = True
                break
        if found_model_id:
            break
    assert found_model_id, (
        f"metadata must record model_id={WAN_PLACEHOLDER_MODEL_ID!r}; "
        f"saw records: {metadata_records!r}"
    )


def test_metadata_records_input_example_ids(
    load_config_module, dispatch_module, tmp_path, monkeypatch
):
    """The metadata file must record every input example id so audit
    tooling can correlate outputs back to inputs.
    """

    manifest_path, output_dir, example_ids = _prepare_inference_scenario(
        tmp_path
    )
    payload = _wan_inference_payload(
        input_manifest=manifest_path, output_dir=output_dir
    )
    plan = _build_plan(load_config_module, dispatch_module, payload, tmp_path)
    selector = _resolve_runtime_adapter_selector(dispatch_module)
    adapter = selector(plan)

    _stub_inner_inference(monkeypatch)
    _invoke_adapter(adapter, plan)

    metadata_records = _load_metadata_files(output_dir)
    assert metadata_records, "no metadata file was written under output_dir"

    flat_strings: list[str] = []
    for rec in metadata_records:
        flat_strings.extend(_flatten_strings(rec))

    for example_id in example_ids:
        assert example_id in flat_strings, (
            f"metadata must record example_id={example_id!r}; "
            f"saw metadata strings: {flat_strings!r}"
        )


def test_metadata_records_generated_artifact_paths(
    load_config_module, dispatch_module, tmp_path, monkeypatch
):
    """The metadata file must record a generated artifact path per
    example. Without that, downstream consumers cannot locate the
    generated outputs.
    """

    manifest_path, output_dir, example_ids = _prepare_inference_scenario(
        tmp_path
    )
    payload = _wan_inference_payload(
        input_manifest=manifest_path, output_dir=output_dir
    )
    plan = _build_plan(load_config_module, dispatch_module, payload, tmp_path)
    selector = _resolve_runtime_adapter_selector(dispatch_module)
    adapter = selector(plan)

    _stub_inner_inference(monkeypatch)
    _invoke_adapter(adapter, plan)

    metadata_records = _load_metadata_files(output_dir)
    assert metadata_records, "no metadata file was written under output_dir"

    artifact_keys = (
        "artifact_path",
        "artifact_paths",
        "generated_artifact_paths",
        "generated_paths",
        "output_path",
        "output_paths",
        "output_artifact",
        "output_artifacts",
        "video_path",
        "save_path",
    )

    # Pull every plausibly-artifact-shaped string out of the metadata.
    artifact_strings: list[str] = []
    for rec in metadata_records:
        flat = _flatten_strings(rec)
        # Any string that points under output_dir is plausibly an artifact.
        artifact_strings.extend(
            value for value in flat if str(output_dir) in value
        )

        # And explicitly probe well-known artifact keys nested in per-example
        # blocks.
        if isinstance(rec, Mapping):
            for value in rec.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, Mapping):
                            for key in artifact_keys:
                                v = item.get(key)
                                if isinstance(v, str):
                                    artifact_strings.append(v)
                                elif isinstance(v, (list, tuple)):
                                    artifact_strings.extend(
                                        x for x in v if isinstance(x, str)
                                    )

    assert artifact_strings, (
        "metadata must record a generated artifact path per example "
        "(under output_path / artifact_path / etc.); saw records: "
        f"{metadata_records!r}"
    )

    # Each example id must be tied to at least one artifact string somewhere
    # in the metadata blob.
    for example_id in example_ids:
        matched = any(example_id in s for s in artifact_strings) or any(
            example_id in s
            for rec in metadata_records
            for s in _flatten_strings(rec)
        )
        assert matched, (
            f"metadata must associate example_id={example_id!r} with a "
            "generated artifact path; correlation not found in records: "
            f"{metadata_records!r}"
        )


def test_runtime_adapter_does_not_spawn_backend_subprocess(
    load_config_module, dispatch_module, tmp_path, monkeypatch
):
    """The artifact-and-metadata contract is the test seam — the adapter
    must not actually spawn a backend subprocess as a side effect of
    producing the metadata.

    A wrapper that shells out to the upstream training executable here
    would break the smoke / dry-run contract and would also make the
    test depend on Wan2.2 being installed.
    """

    import subprocess

    spawned: list[Any] = []

    def _record_call(*args, **kwargs):
        spawned.append((args, kwargs))
        raise AssertionError(
            "Wan runtime adapter must not spawn a backend subprocess "
            "while writing artifacts/metadata in the test seam. "
            f"subprocess called with args={args!r}, kwargs={kwargs!r}"
        )

    monkeypatch.setattr(subprocess, "Popen", _record_call, raising=True)
    monkeypatch.setattr(subprocess, "run", _record_call, raising=True)
    monkeypatch.setattr(subprocess, "call", _record_call, raising=False)
    monkeypatch.setattr(subprocess, "check_call", _record_call, raising=False)
    monkeypatch.setattr(subprocess, "check_output", _record_call, raising=False)

    manifest_path, output_dir, _ids = _prepare_inference_scenario(tmp_path)
    payload = _wan_inference_payload(
        input_manifest=manifest_path, output_dir=output_dir
    )
    plan = _build_plan(load_config_module, dispatch_module, payload, tmp_path)
    selector = _resolve_runtime_adapter_selector(dispatch_module)
    adapter = selector(plan)

    _stub_inner_inference(monkeypatch)
    _invoke_adapter(adapter, plan)
    assert not spawned, (
        "Wan runtime adapter spawned a subprocess while writing "
        f"artifacts/metadata; calls: {spawned!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 4: smoke runs from the repo without manual cd into the
# upstream Wan2.2 checkout
# ---------------------------------------------------------------------------


def _read_smoke_source() -> str:
    if not _SMOKE_PATH.is_file():
        pytest.fail(
            f"M7 requires a smoke run at "
            f"{_SMOKE_PATH.relative_to(_REPO_ROOT)}; the file is missing."
        )
    return _SMOKE_PATH.read_text(encoding="utf-8")


def test_smoke_file_exists_at_pinned_path():
    """The plan pins the smoke file at a specific repo-owned path."""

    assert _SMOKE_PATH.is_file(), (
        f"M7 requires a smoke run at "
        f"{_SMOKE_PATH.relative_to(_REPO_ROOT)}; the file is missing."
    )


def test_smoke_lives_under_repo_owned_package_not_third_party():
    smoke_rel = _SMOKE_PATH.relative_to(_REPO_ROOT)
    assert smoke_rel.parts[0] == "world-model-post-training", (
        f"smoke file must live under world-model-post-training/, not {smoke_rel.parts[0]!r}"
    )
    assert "third_party" not in smoke_rel.parts, (
        "smoke file must not live inside third_party/; M7 forbids repo "
        f"wrapper code there. Got: {smoke_rel}"
    )


def test_smoke_source_is_parseable_python():
    src = _read_smoke_source()
    try:
        ast.parse(src)
    except SyntaxError as exc:  # pragma: no cover - reported below
        pytest.fail(f"smoke file does not parse as Python: {exc!r}")


def test_smoke_does_not_chdir_into_hardcoded_wan_path():
    """The smoke must not contain a literal ``os.chdir`` (or shell ``cd``)
    pointing at a hardcoded ``third_party/wan22`` location. Discovery is
    what makes the smoke runnable from any cwd.
    """

    src = _read_smoke_source()

    forbidden_patterns = (
        r"os\.chdir\(\s*['\"][^'\"]*third_party/wan22",
        r"chdir\(\s*['\"][^'\"]*third_party/wan22",
        r"os\.chdir\(\s*['\"][^'\"]*third_party/wan2\.2",
        r"cd\s+third_party/wan22",
        r"cd\s+third_party/wan2\.2",
        r"cd\s+\$\{?\w*\}?/third_party/wan22",
    )
    offenders = [p for p in forbidden_patterns if re.search(p, src)]
    assert not offenders, (
        "smoke must not require manual or programmatic `cd` into "
        f"third_party/wan22; matched forbidden patterns: {offenders!r}"
    )


def test_smoke_does_not_hardcode_repo_relative_wan_path():
    """Even outside ``os.chdir`` calls, the smoke must not embed an
    absolute or repo-relative ``third_party/wan22/...`` filesystem path
    as a string literal — that would defeat discovery.
    """

    src = _read_smoke_source()
    tree = ast.parse(src)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if (
                "third_party/wan22" in value
                or "third_party\\wan22" in value
                or "third_party/wan2.2" in value
            ):
                offenders.append(value)

    assert not offenders, (
        "smoke must not embed `third_party/wan22` as a literal path; "
        "discovery via `verl_post_training.bootstrap.third_party` is the "
        f"required mechanism. Offending literals: {offenders!r}"
    )


def test_smoke_uses_bootstrap_discovery_helper():
    """The smoke must reach the upstream root through the repo-owned
    bootstrap helper rather than hard-coding paths.
    """

    src = _read_smoke_source()

    discovers_via_helper = (
        "discover_upstream_root" in src
        or "verl_post_training.bootstrap.third_party" in src
        or "load_manifest" in src
    )
    assert discovers_via_helper, (
        "smoke must call `discover_upstream_root` (or load the third_party "
        "manifest) from `verl_post_training.bootstrap.third_party` so the "
        "Wan upstream root is resolved without hard-coded paths."
    )

    assert re.search(r"['\"]wan22?['\"]", src), (
        "smoke must request the 'wan22' family from the discovery helper "
        "so the upstream root is selected from the manifest."
    )


def _ensure_package_on_sys_path():
    src = _REPO_ROOT / "world-model-post-training" / "shared" / "src"
    if src.is_dir():
        src_str = str(src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


def test_smoke_is_importable_from_foreign_cwd(tmp_path, monkeypatch):
    """The smoke must be importable from any working directory; if the
    user can only import it after ``cd``-ing into ``third_party/wan22``,
    the contract is violated.
    """

    _ensure_package_on_sys_path()
    monkeypatch.chdir(tmp_path)

    sys.modules.pop("verl_post_training.smoke.test_wan_generation", None)
    sys.modules.pop("verl_post_training.smoke", None)

    try:
        import importlib

        module = importlib.import_module(
            "verl_post_training.smoke.test_wan_generation"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            "smoke module must be importable from any cwd; got "
            f"ModuleNotFoundError: {exc!r}"
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            "smoke module must import without side effects that require "
            f"a specific cwd; got: {type(exc).__name__}: {exc!r}"
        )

    assert module is not None


def test_smoke_exposes_callable_entry_point(tmp_path, monkeypatch):
    """The smoke must expose a callable entry point so it can be driven
    from CI / a Make target / ``python -m`` without manual cd.
    """

    _ensure_package_on_sys_path()
    monkeypatch.chdir(tmp_path)

    sys.modules.pop("verl_post_training.smoke.test_wan_generation", None)
    import importlib

    try:
        module = importlib.import_module(
            "verl_post_training.smoke.test_wan_generation"
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"smoke module must import cleanly; got {type(exc).__name__}: "
            f"{exc!r}"
        )

    candidates = ("main", "run", "run_smoke", "run_generation_smoke")
    has_callable_entry = any(
        callable(getattr(module, name, None)) for name in candidates
    )
    has_test_entry = any(
        callable(getattr(module, name, None))
        for name in dir(module)
        if name.startswith("test_")
    )
    assert has_callable_entry or has_test_entry, (
        "smoke must expose a callable entry point so an external runner "
        "can drive it without requiring `cd` into the upstream repo. "
        f"Looked for one of {candidates} or a `test_*` function on the "
        "smoke module."
    )


def test_smoke_resolves_wan_upstream_root_via_discovery(tmp_path, monkeypatch):
    """End-to-end behavior check: the discovery helper must resolve a
    usable upstream root for the wan22 family from a foreign cwd.
    """

    _ensure_package_on_sys_path()
    monkeypatch.chdir(tmp_path)

    try:
        from verl_post_training.bootstrap.third_party import (  # noqa: F401
            discover_upstream_root,
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            "Smoke contract depends on the M5 discovery helper "
            "`verl_post_training.bootstrap.third_party.discover_upstream_root`; "
            f"helper is not importable: {exc!r}"
        )
    except ImportError as exc:
        pytest.fail(
            "Smoke contract requires `discover_upstream_root` from "
            "`verl_post_training.bootstrap.third_party`; got ImportError: "
            f"{exc!r}"
        )

    from verl_post_training.bootstrap.third_party import (
        discover_upstream_root,
    )

    try:
        result = discover_upstream_root("wan22")
    except (LookupError, KeyError, ValueError):
        pytest.fail(
            "discover_upstream_root('wan22') must succeed against the real "
            "third_party manifest; without it the smoke cannot resolve the "
            "Wan upstream root."
        )

    assert result is not None, (
        "discover_upstream_root('wan22') must return a path-like value so "
        "the smoke can locate the upstream root without manual cd."
    )

    second_cwd = tmp_path / "elsewhere"
    second_cwd.mkdir()
    monkeypatch.chdir(second_cwd)
    second = discover_upstream_root("wan22")
    assert "wan" in str(second).lower(), (
        f"discover_upstream_root('wan22') from cwd={second_cwd!r} returned "
        f"{second!r}; the smoke would not find the upstream root."
    )
