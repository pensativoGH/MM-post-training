"""M6B acceptance: the V-JEPA2 trainer adapter writes a machine-readable
launch record, and training can be launched from a repo-level config
without manual ``cd`` into ``third_party/vjepa2``.

This file pins the third and fourth M6B acceptance criteria (quoted from
the approved plan):

    the trainer adapter writes a machine-readable launch record under
    ``output_dir`` containing the resolved ``model_id``, ``task_type``,
    ``trainer_backend``, dataset manifest path, upstream root, and final
    backend config file or argument list

    a smoke or dry-run path implemented in
    ``world-model-post-training/shared/src/verl_post_training/smoke/test_vjepa2_training.py``
    proves that training can be launched from a repo-level config without
    manual ``cd`` into ``third_party/vjepa2``

Concretely, these tests pin:

* invoking the V-JEPA2 trainer adapter against a dispatch plan writes at
  least one machine-readable launch record under ``output_dir``
* the launch record contains the resolved ``model_id``, ``task_type``,
  ``trainer_backend``, dataset manifest path, upstream root, and either a
  ``backend_config_file`` path or an ``argument_list`` / ``argv``
* the smoke file at the plan-pinned path exists and lives under
  ``world-model-post-training/shared/src/`` (not under ``third_party/``)
* the smoke file does not embed a literal ``third_party/vjepa2`` path and
  does not ``os.chdir`` into the upstream tree
* the smoke uses the M5 discovery helper to resolve the upstream root and
  is importable from a foreign working directory
* the trainer adapter does not write the launch record by spawning a
  backend subprocess — the record must be produced from the wrapper
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
    / "vjepa"
    / "src"
    / "verl_post_training_vjepa"
    / "smoke"
    / "test_training.py"
)


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


def _resolve_trainer_adapter_selector(dispatch_module):
    for attr in (
        "get_trainer_adapter",
        "select_trainer_adapter",
        "resolve_trainer_adapter",
        "trainer_adapter_for_plan",
        "build_trainer_adapter",
    ):
        fn = getattr(dispatch_module, attr, None)
        if callable(fn):
            return fn

    try:
        import verl_post_training.adapters.trainer as trainer_pkg
    except ModuleNotFoundError:
        pytest.fail(
            "verl_post_training.adapters.trainer must exist with a "
            "trainer adapter selector."
        )
    for attr in (
        "get_trainer_adapter",
        "select_trainer_adapter",
        "resolve_trainer_adapter",
        "trainer_adapter_for_plan",
    ):
        fn = getattr(trainer_pkg, attr, None)
        if callable(fn):
            return fn
    pytest.fail(
        "Repo-level dispatch must expose a selector that returns a trainer "
        "adapter for a dispatch plan."
    )


def _find_vjepa2_training_model_id() -> str:
    from verl_post_training.registry import iter_entries
    from verl_post_training.registry.schemas import (
        TaskType,
        TrainerBackend,
    )

    for entry in iter_entries():
        if (
            TaskType.MASKED_VIDEO_PREDICTION in entry.supported_task_types
            and TrainerBackend.VJEPA2_NATIVE in entry.trainer_backends
        ):
            return entry.model_id
    pytest.fail(
        "M6B requires at least one V-JEPA2 registry entry that advertises "
        "both masked_video_prediction and vjepa2_native; none found."
    )


def _vjepa2_training_payload(
    *,
    model_id: str,
    input_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "task_type": "masked_video_prediction",
        "model_id": model_id,
        "trainer_backend": "vjepa2_native",
        "dataset_adapter": "vjepa2",
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


def _invoke_trainer_adapter(adapter, plan):
    """Invoke the trainer adapter's launch / prepare / dry-run hook.

    Accept ``launch``, ``run``, ``prepare``, ``invoke``, ``execute``, or
    ``__call__``. The hook must produce the launch record as a side effect
    (writing to output_dir) and may also return a metadata envelope.
    """

    for attr in (
        "launch",
        "dry_run",
        "plan",
        "prepare",
        "run",
        "invoke",
        "execute",
    ):
        fn = getattr(adapter, attr, None)
        if callable(fn):
            return fn(plan)
    if callable(adapter):
        return adapter(plan)
    pytest.fail(
        "V-JEPA2 trainer adapter must expose a callable entry point "
        "(launch/dry_run/plan/prepare/run/invoke/execute/__call__); got: "
        f"{adapter!r}"
    )


def _gather_output_files(output_dir: Path) -> list[Path]:
    return [p for p in output_dir.rglob("*") if p.is_file()]


def _try_parse_record(path: Path) -> Mapping[str, Any] | None:
    """Try to parse a single launch record file as JSON or YAML."""

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


def _load_launch_records(output_dir: Path) -> list[Mapping[str, Any]]:
    """Walk output_dir for any machine-readable launch record."""

    records: list[Mapping[str, Any]] = []
    for path in _gather_output_files(output_dir):
        rec = _try_parse_record(path)
        if rec is not None:
            records.append(rec)
    return records


def _record_field(rec: Mapping[str, Any], *candidates: str) -> Any:
    """Return the first present field name from candidates."""

    for key in candidates:
        if key in rec:
            return rec[key]
    return None


# ---------------------------------------------------------------------------
# Criterion 3: trainer adapter writes a machine-readable launch record
# ---------------------------------------------------------------------------


def _prepare_training_scenario(tmp_path: Path) -> tuple[Path, Path, str]:
    manifest_path = tmp_path / "video_train_manifest.jsonl"
    manifest_path.write_text(
        '{"example_id": "vid_train_000", "media_paths": ["/data/train/a.mp4"], "modality": "video"}\n'
        '{"example_id": "vid_train_001", "media_paths": ["/data/train/b.mp4"], "modality": "video"}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "trainer_out"
    output_dir.mkdir()
    model_id = _find_vjepa2_training_model_id()
    return manifest_path, output_dir, model_id


def test_trainer_adapter_writes_launch_record_under_output_dir(
    load_config_module, dispatch_module, tmp_path
):
    """Invoking the V-JEPA2 trainer adapter must produce at least one
    machine-readable launch record under ``output_dir``.
    """

    manifest_path, output_dir, model_id = _prepare_training_scenario(tmp_path)
    payload = _vjepa2_training_payload(
        model_id=model_id, input_manifest=manifest_path, output_dir=output_dir
    )
    plan = _build_plan(load_config_module, dispatch_module, payload, tmp_path)
    selector = _resolve_trainer_adapter_selector(dispatch_module)
    adapter = selector(plan)
    assert adapter is not None, "trainer adapter selector returned None"

    _invoke_trainer_adapter(adapter, plan)

    files = _gather_output_files(output_dir)
    assert files, (
        f"V-JEPA2 trainer adapter must write at least one artifact under "
        f"{output_dir}; found none."
    )
    records = _load_launch_records(output_dir)
    assert records, (
        "V-JEPA2 trainer adapter must write at least one machine-readable "
        "launch record (JSON or YAML) under output_dir; saw files but none "
        f"parsed as a mapping. Files: {[str(p) for p in files]!r}"
    )


def test_launch_record_carries_resolved_model_task_and_backend(
    load_config_module, dispatch_module, tmp_path
):
    """The launch record must echo the resolved ``model_id``, ``task_type``,
    and ``trainer_backend`` so smoke / CI can audit the launch.
    """

    manifest_path, output_dir, model_id = _prepare_training_scenario(tmp_path)
    payload = _vjepa2_training_payload(
        model_id=model_id, input_manifest=manifest_path, output_dir=output_dir
    )
    plan = _build_plan(load_config_module, dispatch_module, payload, tmp_path)
    selector = _resolve_trainer_adapter_selector(dispatch_module)
    adapter = selector(plan)
    _invoke_trainer_adapter(adapter, plan)

    records = _load_launch_records(output_dir)
    assert records, "no launch record was written under output_dir"

    # At least one record must carry the resolved identifiers. Allow the
    # writer to split records (e.g., one for plan, one for argv) but each
    # required value must appear in some record.
    def _has_value(field_candidates: tuple[str, ...], expected: str) -> bool:
        for rec in records:
            value = _record_field(rec, *field_candidates)
            if value is None:
                continue
            value_str = getattr(value, "value", value)
            if str(value_str) == expected:
                return True
        return False

    assert _has_value(("model_id", "resolved_model_id", "model"), model_id), (
        f"launch record must echo model_id={model_id!r}; "
        f"saw records: {records!r}"
    )
    assert _has_value(
        ("task_type", "task", "resolved_task_type"),
        "masked_video_prediction",
    ), (
        "launch record must echo task_type=masked_video_prediction; "
        f"saw records: {records!r}"
    )
    assert _has_value(
        ("trainer_backend", "backend", "resolved_trainer_backend"),
        "vjepa2_native",
    ), (
        "launch record must echo trainer_backend=vjepa2_native; "
        f"saw records: {records!r}"
    )


def test_launch_record_carries_dataset_manifest_path(
    load_config_module, dispatch_module, tmp_path
):
    """The launch record must reference the dataset manifest the adapter
    consumed — without that, downstream audit cannot tell which data the
    backend was launched against.
    """

    manifest_path, output_dir, model_id = _prepare_training_scenario(tmp_path)
    payload = _vjepa2_training_payload(
        model_id=model_id, input_manifest=manifest_path, output_dir=output_dir
    )
    plan = _build_plan(load_config_module, dispatch_module, payload, tmp_path)
    selector = _resolve_trainer_adapter_selector(dispatch_module)
    adapter = selector(plan)
    _invoke_trainer_adapter(adapter, plan)

    records = _load_launch_records(output_dir)
    assert records, "no launch record was written under output_dir"

    found = False
    for rec in records:
        for key in (
            "input_manifest",
            "dataset_manifest",
            "dataset_manifest_path",
            "manifest_path",
            "dataset_path",
        ):
            value = rec.get(key) if isinstance(rec, Mapping) else None
            if value is None:
                continue
            if str(manifest_path) in str(value):
                found = True
                break
        if found:
            break

    assert found, (
        f"launch record must reference dataset manifest path {manifest_path!r}; "
        f"saw records: {records!r}"
    )


def test_launch_record_carries_upstream_root(
    load_config_module, dispatch_module, tmp_path
):
    """The launch record must record the upstream V-JEPA2 root — without
    it, the audit trail cannot identify which checkout the launch used.
    """

    manifest_path, output_dir, model_id = _prepare_training_scenario(tmp_path)
    payload = _vjepa2_training_payload(
        model_id=model_id, input_manifest=manifest_path, output_dir=output_dir
    )
    plan = _build_plan(load_config_module, dispatch_module, payload, tmp_path)
    selector = _resolve_trainer_adapter_selector(dispatch_module)
    adapter = selector(plan)
    _invoke_trainer_adapter(adapter, plan)

    records = _load_launch_records(output_dir)
    assert records, "no launch record was written under output_dir"

    found = False
    for rec in records:
        for key in (
            "upstream_root",
            "vjepa2_root",
            "third_party_root",
            "repo_dir",
            "upstream_repo",
            "upstream_path",
        ):
            value = rec.get(key) if isinstance(rec, Mapping) else None
            if value is None:
                continue
            if "vjepa2" in str(value).lower():
                found = True
                break
        if found:
            break

    assert found, (
        "launch record must reference the upstream V-JEPA2 root "
        "(e.g. via upstream_root / vjepa2_root). Saw records: "
        f"{records!r}"
    )


def test_launch_record_carries_backend_config_or_argv(
    load_config_module, dispatch_module, tmp_path
):
    """The launch record must include either a backend config file path or
    an argument list — that is how a smoke or CI run reproduces the
    upstream launch.
    """

    manifest_path, output_dir, model_id = _prepare_training_scenario(tmp_path)
    payload = _vjepa2_training_payload(
        model_id=model_id, input_manifest=manifest_path, output_dir=output_dir
    )
    plan = _build_plan(load_config_module, dispatch_module, payload, tmp_path)
    selector = _resolve_trainer_adapter_selector(dispatch_module)
    adapter = selector(plan)
    _invoke_trainer_adapter(adapter, plan)

    records = _load_launch_records(output_dir)
    assert records, "no launch record was written under output_dir"

    backend_config_file_keys = (
        "backend_config_file",
        "config_file",
        "backend_config_path",
        "config_path",
    )
    argv_keys = (
        "argument_list",
        "argv",
        "args",
        "command",
        "command_line",
    )

    has_config_file = any(
        any(rec.get(k) for k in backend_config_file_keys if isinstance(rec, Mapping))
        for rec in records
    )
    has_argv = False
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        for k in argv_keys:
            value = rec.get(k)
            if isinstance(value, (list, tuple)) and value:
                has_argv = True
                break
            if isinstance(value, str) and value.strip():
                has_argv = True
                break
        if has_argv:
            break

    assert has_config_file or has_argv, (
        "launch record must include a backend config file path or an "
        f"argument list. Saw records: {records!r}"
    )


def test_trainer_adapter_does_not_spawn_backend_subprocess(
    load_config_module, dispatch_module, tmp_path, monkeypatch
):
    """The launch record is the dry-run contract — the adapter must not
    actually spawn a backend subprocess as a side effect of producing it.

    We monkeypatch ``subprocess.Popen`` and ``subprocess.run`` to record any
    calls; an adapter that launches the upstream training executable here
    would break the smoke / dry-run contract.
    """

    import subprocess

    spawned: list[Any] = []

    def _record_call(*args, **kwargs):
        spawned.append((args, kwargs))
        raise AssertionError(
            "V-JEPA2 trainer adapter must not spawn a backend subprocess "
            "while writing the launch record; the launch record is a "
            "dry-run artifact. "
            f"subprocess called with args={args!r}, kwargs={kwargs!r}"
        )

    monkeypatch.setattr(subprocess, "Popen", _record_call, raising=True)
    monkeypatch.setattr(subprocess, "run", _record_call, raising=True)
    monkeypatch.setattr(subprocess, "call", _record_call, raising=False)
    monkeypatch.setattr(subprocess, "check_call", _record_call, raising=False)
    monkeypatch.setattr(subprocess, "check_output", _record_call, raising=False)

    manifest_path, output_dir, model_id = _prepare_training_scenario(tmp_path)
    payload = _vjepa2_training_payload(
        model_id=model_id, input_manifest=manifest_path, output_dir=output_dir
    )
    plan = _build_plan(load_config_module, dispatch_module, payload, tmp_path)
    selector = _resolve_trainer_adapter_selector(dispatch_module)
    adapter = selector(plan)

    # The adapter should be able to run end-to-end without ever calling
    # subprocess. If it tries to, the patched callable will raise above.
    _invoke_trainer_adapter(adapter, plan)
    assert not spawned, (
        "trainer adapter spawned a subprocess while writing the launch "
        f"record; calls: {spawned!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 4: smoke exists, lives outside third_party/, uses discovery
# ---------------------------------------------------------------------------


def _read_smoke_source() -> str:
    if not _SMOKE_PATH.is_file():
        pytest.fail(
            "M6B requires a training smoke at "
            f"{_SMOKE_PATH.relative_to(_REPO_ROOT)}; the file is missing."
        )
    return _SMOKE_PATH.read_text(encoding="utf-8")


def test_training_smoke_file_exists_at_pinned_path():
    assert _SMOKE_PATH.is_file(), (
        "M6B requires a training smoke at "
        f"{_SMOKE_PATH.relative_to(_REPO_ROOT)}; the file is missing."
    )


def test_training_smoke_lives_under_repo_owned_package():
    smoke_rel = _SMOKE_PATH.relative_to(_REPO_ROOT)
    assert smoke_rel.parts[0] == "world-model-post-training", (
        f"smoke file must live under world-model-post-training/, not {smoke_rel.parts[0]!r}"
    )
    assert "third_party" not in smoke_rel.parts, (
        "smoke file must not live inside third_party/; M6B forbids repo "
        f"wrapper code there. Got: {smoke_rel}"
    )


def test_training_smoke_does_not_chdir_into_hardcoded_vjepa2_path():
    src = _read_smoke_source()
    forbidden_patterns = (
        r"os\.chdir\(\s*['\"][^'\"]*third_party/vjepa2",
        r"chdir\(\s*['\"][^'\"]*third_party/vjepa2",
        r"cd\s+third_party/vjepa2",
        r"cd\s+\$\{?\w*\}?/third_party/vjepa2",
    )
    offenders = [p for p in forbidden_patterns if re.search(p, src)]
    assert not offenders, (
        "training smoke must not require manual or programmatic `cd` into "
        f"third_party/vjepa2; matched forbidden patterns: {offenders!r}"
    )


def test_training_smoke_does_not_hardcode_repo_relative_vjepa2_path():
    src = _read_smoke_source()
    tree = ast.parse(src)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if "third_party/vjepa2" in value or "third_party\\vjepa2" in value:
                offenders.append(value)
    assert not offenders, (
        "training smoke must not embed `third_party/vjepa2` as a literal "
        "path; discovery via `verl_post_training.bootstrap.third_party` is "
        f"the required mechanism. Offending literals: {offenders!r}"
    )


def test_training_smoke_uses_bootstrap_discovery_helper():
    src = _read_smoke_source()
    discovers_via_helper = (
        "discover_upstream_root" in src
        or "verl_post_training.bootstrap.third_party" in src
        or "load_manifest" in src
    )
    assert discovers_via_helper, (
        "training smoke must call `discover_upstream_root` (or load the "
        "third_party manifest) from "
        "`verl_post_training.bootstrap.third_party` so the upstream root is "
        "resolved without hard-coded paths."
    )
    assert re.search(r"['\"]vjepa2['\"]", src), (
        "training smoke must request the 'vjepa2' family from the discovery "
        "helper so the upstream root is selected from the manifest."
    )


def test_training_smoke_is_importable_from_foreign_cwd(tmp_path, monkeypatch):
    """The smoke must be importable from any working directory; if the user
    can only import it after ``cd``-ing into ``third_party/vjepa2``, the
    contract is violated.
    """

    src_root = _REPO_ROOT / "world-model-post-training" / "shared" / "src"
    if src_root.is_dir():
        src_str = str(src_root)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
    monkeypatch.chdir(tmp_path)

    sys.modules.pop("verl_post_training.smoke.test_vjepa2_training", None)
    sys.modules.pop("verl_post_training.smoke", None)

    try:
        import importlib

        module = importlib.import_module(
            "verl_post_training.smoke.test_vjepa2_training"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            "smoke module must be importable from any cwd; got "
            f"ModuleNotFoundError: {exc!r}"
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            "smoke module must import without side effects that require a "
            f"specific cwd; got: {type(exc).__name__}: {exc!r}"
        )
    assert module is not None


def test_training_smoke_exposes_callable_entry_point(tmp_path, monkeypatch):
    """The smoke must expose a callable entry point so it can be driven
    from CI / a Make target / ``python -m`` without manual cd.
    """

    src_root = _REPO_ROOT / "world-model-post-training" / "shared" / "src"
    if src_root.is_dir():
        src_str = str(src_root)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
    monkeypatch.chdir(tmp_path)

    sys.modules.pop("verl_post_training.smoke.test_vjepa2_training", None)
    import importlib

    try:
        module = importlib.import_module(
            "verl_post_training.smoke.test_vjepa2_training"
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"smoke module must import cleanly; got {type(exc).__name__}: "
            f"{exc!r}"
        )

    candidates = ("main", "run", "run_smoke", "run_training_smoke")
    has_callable_entry = any(
        callable(getattr(module, name, None)) for name in candidates
    )
    has_test_entry = any(
        callable(getattr(module, name, None))
        for name in dir(module)
        if name.startswith("test_")
    )
    assert has_callable_entry or has_test_entry, (
        "training smoke must expose a callable entry point so an external "
        "runner can drive it without requiring `cd` into the upstream repo. "
        f"Looked for one of {candidates} or a `test_*` function."
    )


def test_training_smoke_resolves_upstream_root_via_discovery(
    tmp_path, monkeypatch
):
    """The discovery helper must be reachable from a foreign cwd and must
    return a usable upstream root for the vjepa2 family.
    """

    src_root = _REPO_ROOT / "world-model-post-training" / "shared" / "src"
    if src_root.is_dir():
        src_str = str(src_root)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
    monkeypatch.chdir(tmp_path)

    try:
        from verl_post_training.bootstrap.third_party import (  # noqa: F401
            discover_upstream_root,
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            "M6B training smoke depends on the M5 discovery helper "
            "`verl_post_training.bootstrap.third_party.discover_upstream_root`; "
            f"helper is not importable: {exc!r}"
        )

    from verl_post_training.bootstrap.third_party import (
        discover_upstream_root,
    )

    try:
        result = discover_upstream_root("vjepa2")
    except (LookupError, KeyError, ValueError):
        pytest.fail(
            "discover_upstream_root('vjepa2') must succeed against the real "
            "third_party manifest; without it the training smoke cannot "
            "resolve the upstream root."
        )

    assert result is not None, (
        "discover_upstream_root('vjepa2') must return a path-like value so "
        "the training smoke can locate the upstream root without manual cd."
    )
    assert "vjepa2" in str(result).lower(), (
        f"discover_upstream_root('vjepa2') returned {result!r}; the training "
        "smoke would not find the upstream root."
    )
