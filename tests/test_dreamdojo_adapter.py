"""M8 acceptance: the DreamDojo dataset adapter writes backend-ready
trajectory or world-model inputs without mutating source records.

This file pins the first M8 acceptance criterion (quoted from the approved
plan):

    given a pipeline manifest with supported temporal records, the
    DreamDojo adapter writes a backend-ready manifest or trajectory bundle
    without mutating source records

The adapter is expected to register against the ``dreamdojo`` key already
exposed by the M4 adapter registry. Concretely, these tests pin:

* ``get_dataset_adapter("dreamdojo")`` resolves to a real adapter (not the
  M4 ``NotImplementedError`` stub) that exposes a callable ``prepare``
  method
* given a pipeline manifest with one or more temporal (observation +
  action) trajectory rows, ``prepare`` writes at least one machine-readable
  artifact under ``output_dir``
* the written output references each source trajectory's example_id and
  its observation/action references
* the adapter does not mutate the source rows it was handed (no in-place
  list/dict edits, no key rename, no insertion)
* media references already presented as absolute paths or URIs are
  preserved verbatim so the world-model loader can consume them by
  reference (no copy/rewrite of source bytes when references are
  sufficient)
* multi-row manifests produce one logical record per row

Tests are deterministic and self-contained: they use ``tmp_path`` for the
source trajectory stubs and the adapter output, and never start a
backend process.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_OBS_SENTINEL = b"\x00\x01OBS-DREAMDOJO-TEST-SENTINEL-BYTES-AAAAAA"
_ACT_SENTINEL = b"\x02\x03ACT-DREAMDOJO-TEST-SENTINEL-BYTES-BBBBBB"


def _resolve_adapter():
    from verl_post_training.adapters.dataset import get_dataset_adapter

    adapter = get_dataset_adapter("dreamdojo")
    assert adapter is not None, (
        "get_dataset_adapter('dreamdojo') must resolve to a real adapter "
        "once M8 lands; got None."
    )
    assert hasattr(adapter, "prepare"), (
        "DreamDojo dataset adapter must expose a callable `prepare` method; "
        f"got: {adapter!r}"
    )
    return adapter


def _write_fake_blob(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _trajectory_pipeline_row(
    *, obs_path: Path, act_path: Path, idx: int = 0
) -> dict[str, Any]:
    return {
        "example_id": f"traj_{idx:03d}",
        "modality": "trajectory",
        "observation_paths": [str(obs_path)],
        "action_paths": [str(act_path)],
        "metadata": {"episode_index": idx, "horizon": 16},
    }


def _gather_output_files(output_dir: Path) -> list[Path]:
    return [p for p in output_dir.rglob("*") if p.is_file()]


def _read_text_outputs(output_dir: Path) -> str:
    chunks: list[str] = []
    for path in _gather_output_files(output_dir):
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
    return "\n".join(chunks)


def _parse_records(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _gather_output_files(output_dir):
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
        elif path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        records.append(item)
            elif isinstance(payload, dict):
                for value in payload.values():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                records.append(item)
                if not records:
                    records.append(payload)
    return records


def _call_prepare(adapter, *, manifest, output_dir):
    """Call the adapter's prepare hook, tolerating a few signature shapes.

    M4-era adapters used ``prepare(pipeline_manifest=..., output_dir=...)``;
    we accept a couple of common keyword spellings + positional as a last
    resort so writers can pick a sensible signature.
    """

    candidates = (
        {"pipeline_manifest": manifest, "output_dir": output_dir},
        {"manifest": manifest, "output_dir": output_dir},
        {"rows": manifest, "output_dir": output_dir},
        {"records": manifest, "output_dir": output_dir},
    )
    last_err: TypeError | NotImplementedError | None = None
    for kwargs in candidates:
        try:
            return adapter.prepare(**kwargs)
        except TypeError as exc:
            last_err = exc
            continue
        except NotImplementedError as exc:
            # The M4 stub raises NotImplementedError; the M8 implementation
            # must replace it. Surface this as a clean failure.
            raise AssertionError(
                "DreamDojo adapter `prepare` still raises NotImplementedError; "
                "M8 must replace the M4 stub with a real implementation. "
                f"Got: {exc!r}"
            )
    try:
        return adapter.prepare(manifest, output_dir)
    except TypeError as exc:  # pragma: no cover - reported below
        last_err = exc
    except NotImplementedError as exc:
        raise AssertionError(
            "DreamDojo adapter `prepare` still raises NotImplementedError; "
            "M8 must replace the M4 stub with a real implementation. "
            f"Got: {exc!r}"
        )
    raise AssertionError(
        "DreamDojo adapter `prepare` must accept the M4 signature "
        "(pipeline_manifest=..., output_dir=...). Last TypeError: "
        f"{last_err!r}"
    )


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------


def test_dreamdojo_adapter_is_resolvable_from_registry():
    """The M4 stub key ``dreamdojo`` must resolve to a real adapter by M8 —
    not the placeholder that raises ``NotImplementedError``.
    """

    adapter = _resolve_adapter()
    assert callable(getattr(adapter, "prepare", None)), (
        "DreamDojo adapter must implement a callable `prepare` hook so the "
        "control plane can drive dataset preparation through it."
    )


# ---------------------------------------------------------------------------
# Acceptance criterion: backend-ready manifest without mutating source rows
# ---------------------------------------------------------------------------


def test_dreamdojo_adapter_writes_backend_ready_manifest(tmp_path):
    """The adapter must emit at least one machine-readable artifact under
    ``output_dir`` that the world-model backend can consume.
    """

    adapter = _resolve_adapter()
    obs_path = _write_fake_blob(tmp_path / "src" / "obs_0.npz", _OBS_SENTINEL)
    act_path = _write_fake_blob(tmp_path / "src" / "act_0.npz", _ACT_SENTINEL)
    manifest = [_trajectory_pipeline_row(obs_path=obs_path, act_path=act_path, idx=0)]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    files = _gather_output_files(out_dir)
    assert files, (
        f"DreamDojo adapter must write at least one artifact under {out_dir}; "
        "found none."
    )

    text_blob = _read_text_outputs(out_dir)
    assert "traj_000" in text_blob, (
        "Backend-ready manifest must reference each row's example_id; "
        f"instead found:\n{text_blob[:500]}"
    )
    assert str(obs_path) in text_blob, (
        "Backend-ready manifest must reference the source observation path "
        f"{str(obs_path)!r}; instead found:\n{text_blob[:500]}"
    )
    assert str(act_path) in text_blob, (
        "Backend-ready manifest must reference the source action path "
        f"{str(act_path)!r}; instead found:\n{text_blob[:500]}"
    )


def test_dreamdojo_adapter_does_not_duplicate_source_bytes(tmp_path):
    """If the source observation/action records are reachable, the adapter
    must reference them rather than copying the raw bytes into the output
    directory.
    """

    adapter = _resolve_adapter()
    obs_path = _write_fake_blob(tmp_path / "src" / "obs_0.npz", _OBS_SENTINEL)
    act_path = _write_fake_blob(tmp_path / "src" / "act_0.npz", _ACT_SENTINEL)
    manifest = [_trajectory_pipeline_row(obs_path=obs_path, act_path=act_path, idx=0)]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    for produced in _gather_output_files(out_dir):
        try:
            contents = produced.read_bytes()
        except OSError:
            continue
        assert _OBS_SENTINEL not in contents, (
            f"DreamDojo adapter copied source observation bytes into "
            f"{produced} — the adapter must reference the source records "
            "rather than duplicating them."
        )
        assert _ACT_SENTINEL not in contents, (
            f"DreamDojo adapter copied source action bytes into {produced} — "
            "the adapter must reference the source records rather than "
            "duplicating them."
        )


def test_dreamdojo_adapter_preserves_existing_references(tmp_path):
    """If a pipeline row already carries an absolute or URI-style media
    path, the adapter must surface that reference verbatim — the upstream
    DreamDojo loader is expected to consume by reference, not from a
    relocated copy.
    """

    adapter = _resolve_adapter()
    absolute_obs = _write_fake_blob(
        tmp_path / "src" / "obs_abs.npz", _OBS_SENTINEL
    )
    absolute_act = _write_fake_blob(
        tmp_path / "src" / "act_abs.npz", _ACT_SENTINEL
    )
    manifest = [
        _trajectory_pipeline_row(
            obs_path=absolute_obs, act_path=absolute_act, idx=0
        ),
        {
            "example_id": "traj_remote_001",
            "modality": "trajectory",
            "observation_paths": ["s3://bucket/key/obs_001.npz"],
            "action_paths": ["s3://bucket/key/act_001.npz"],
            "metadata": {"episode_index": 1},
        },
    ]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    text_blob = _read_text_outputs(out_dir)
    assert str(absolute_obs) in text_blob, (
        "Absolute filesystem reference must be preserved verbatim in the "
        "backend-ready manifest."
    )
    assert "s3://bucket/key/obs_001.npz" in text_blob, (
        "URI-style remote reference must be preserved verbatim in the "
        "backend-ready manifest."
    )


def test_dreamdojo_adapter_does_not_mutate_source_manifest(tmp_path):
    """Acceptance criterion: source records must not be mutated."""

    adapter = _resolve_adapter()
    obs_path = _write_fake_blob(tmp_path / "src" / "obs_0.npz", _OBS_SENTINEL)
    act_path = _write_fake_blob(tmp_path / "src" / "act_0.npz", _ACT_SENTINEL)
    manifest = [
        _trajectory_pipeline_row(obs_path=obs_path, act_path=act_path, idx=0),
        _trajectory_pipeline_row(obs_path=obs_path, act_path=act_path, idx=1),
    ]
    snapshot = copy.deepcopy(manifest)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    assert manifest == snapshot, (
        "DreamDojo adapter must not mutate the source pipeline manifest."
    )


def test_dreamdojo_adapter_handles_multiple_rows(tmp_path):
    """Multi-row manifest must produce one logical record per row.

    The adapter is free to write a single file (e.g. one JSONL) or one
    file per row; either way, each row's reference must appear in the
    output and every example_id must be enumerable.
    """

    adapter = _resolve_adapter()
    obs_a = _write_fake_blob(tmp_path / "src" / "obs_a.npz", _OBS_SENTINEL)
    obs_b = _write_fake_blob(tmp_path / "src" / "obs_b.npz", _OBS_SENTINEL)
    act_a = _write_fake_blob(tmp_path / "src" / "act_a.npz", _ACT_SENTINEL)
    act_b = _write_fake_blob(tmp_path / "src" / "act_b.npz", _ACT_SENTINEL)
    manifest = [
        _trajectory_pipeline_row(obs_path=obs_a, act_path=act_a, idx=0),
        _trajectory_pipeline_row(obs_path=obs_b, act_path=act_b, idx=1),
    ]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    text_blob = _read_text_outputs(out_dir)
    for row in manifest:
        assert row["example_id"] in text_blob, (
            "Backend-ready manifest must reference every row's example_id. "
            f"Missing: {row['example_id']!r}; saw text blob excerpt: "
            f"{text_blob[:500]!r}"
        )

    records = _parse_records(out_dir)
    if records:
        assert len(records) >= len(manifest), (
            "DreamDojo backend-ready manifest must contain at least one "
            f"record per source row; saw {len(records)} for {len(manifest)} "
            "rows."
        )
