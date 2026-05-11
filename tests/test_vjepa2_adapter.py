"""M6 acceptance: the V-JEPA2 dataset adapter writes backend-ready video
inputs without duplicating media when references are sufficient.

This file pins the first M6 acceptance criterion (quoted from the approved
plan):

    given a pipeline manifest with video inputs, the V-JEPA2 adapter writes
    a backend-ready manifest or asset reference set without duplicating
    media when references are sufficient

The adapter is expected to register against the ``vjepa2`` key already
exposed by the M4 adapter registry. Concretely, these tests pin:

* ``get_dataset_adapter("vjepa2")`` resolves to a real adapter (not the M4
  stub) that exposes a callable ``prepare`` method
* given a pipeline manifest with one or more video rows, ``prepare`` writes
  at least one machine-readable artifact under ``output_dir``
* the written output references each source video path
* the adapter does not copy or rewrite video bytes when the source manifest
  is reachable — assertion: no file under ``output_dir`` contains the
  source video bytes
* the adapter does not mutate the source rows it was handed
* video paths that are already references (URI / absolute path) are
  preserved verbatim so downstream wrappers can consume them by reference

The tests are deterministic and self-contained — they use ``tmp_path`` for
both the source video stubs and the adapter output, and never start a
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


_VIDEO_SENTINEL = b"\x00\x00\x00\x18ftypmp42-VJEPA2-TEST-SENTINEL-BYTES-AAAA"


def _resolve_adapter():
    from verl_post_training.adapters.dataset import get_dataset_adapter

    adapter = get_dataset_adapter("vjepa2")
    assert adapter is not None, (
        "get_dataset_adapter('vjepa2') must resolve to a real adapter once "
        "M6 lands; got None."
    )
    assert hasattr(adapter, "prepare"), (
        "V-JEPA2 dataset adapter must expose a callable `prepare` method; "
        f"got: {adapter!r}"
    )
    return adapter


def _write_fake_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_VIDEO_SENTINEL)
    return path


def _video_pipeline_row(video_path: Path, idx: int = 0) -> dict[str, Any]:
    return {
        "example_id": f"vid_{idx:03d}",
        "media_paths": [str(video_path)],
        "modality": "video",
        "metadata": {"clip_index": idx},
    }


def _gather_output_files(output_dir: Path) -> list[Path]:
    return [p for p in output_dir.rglob("*") if p.is_file()]


def _read_text_outputs(output_dir: Path) -> str:
    """Concatenated text of every text-ish artifact the adapter wrote.

    The writer may choose JSON, JSONL, YAML, CSV, or another machine-readable
    format — the contract is just that source media paths appear in some
    backend-consumable artifact.
    """

    chunks: list[str] = []
    for path in _gather_output_files(output_dir):
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
    return "\n".join(chunks)


def _call_prepare(adapter, *, manifest, output_dir):
    """Call the adapter's prepare hook, tolerating a few signature shapes.

    M4-era adapters used ``prepare(pipeline_manifest=..., output_dir=...)``;
    we accept positional + a couple of common keyword spellings too.
    """

    candidates = (
        {"pipeline_manifest": manifest, "output_dir": output_dir},
        {"manifest": manifest, "output_dir": output_dir},
        {"rows": manifest, "output_dir": output_dir},
    )
    last_err: TypeError | None = None
    for kwargs in candidates:
        try:
            return adapter.prepare(**kwargs)
        except TypeError as exc:
            last_err = exc
            continue
    # Last resort: positional.
    try:
        return adapter.prepare(manifest, output_dir)
    except TypeError as exc:  # pragma: no cover - reported below
        last_err = exc
    raise AssertionError(
        "V-JEPA2 adapter `prepare` must accept the M4 signature "
        "(pipeline_manifest=..., output_dir=...). Last TypeError: "
        f"{last_err!r}"
    )


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------


def test_vjepa2_adapter_is_resolvable_from_registry():
    """The M4 stub key ``vjepa2`` must resolve to a real adapter by M6 —
    not the placeholder object the M4 milestone permits.
    """

    adapter = _resolve_adapter()
    # Guard against a bare ``object()`` stub by checking the adapter advertises
    # at least one of the standard dataset-adapter hooks.
    assert callable(getattr(adapter, "prepare", None)), (
        "V-JEPA2 adapter must implement a callable `prepare` hook so the "
        "control plane can drive dataset preparation through it."
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 1: backend-ready output without duplicating media
# ---------------------------------------------------------------------------


def test_vjepa2_adapter_writes_backend_ready_manifest(tmp_path):
    """The adapter must emit at least one machine-readable artifact under
    ``output_dir`` that downstream backends can consume.
    """

    adapter = _resolve_adapter()
    source_video = _write_fake_video(tmp_path / "src" / "clip_0.mp4")
    manifest = [_video_pipeline_row(source_video, idx=0)]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    files = _gather_output_files(out_dir)
    assert files, (
        f"V-JEPA2 adapter must write at least one artifact under {out_dir}; "
        "found none."
    )

    text_blob = _read_text_outputs(out_dir)
    assert str(source_video) in text_blob, (
        "Backend-ready manifest must reference the source video path "
        f"{str(source_video)!r}; instead found:\n{text_blob[:500]}"
    )


def test_vjepa2_adapter_does_not_duplicate_video_bytes(tmp_path):
    """Acceptance criterion: "without duplicating media when references are
    sufficient". If the source video is reachable, the adapter must not
    copy or rewrite the bytes anywhere under ``output_dir``.
    """

    adapter = _resolve_adapter()
    source_video = _write_fake_video(tmp_path / "src" / "clip_0.mp4")
    manifest = [_video_pipeline_row(source_video, idx=0)]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    for produced in _gather_output_files(out_dir):
        try:
            contents = produced.read_bytes()
        except OSError:
            continue
        assert _VIDEO_SENTINEL not in contents, (
            f"V-JEPA2 adapter copied source video bytes into {produced} — "
            "the adapter must reference the source media rather than "
            "duplicating it."
        )
        # An identical-size copy is a softer signal of duplication; flag it.
        if produced.name.endswith(source_video.suffix):
            assert produced.stat().st_size != source_video.stat().st_size or (
                produced.read_bytes() != source_video.read_bytes()
            ), (
                f"V-JEPA2 adapter wrote a same-size, same-bytes copy of the "
                f"source video at {produced}; references must be preserved."
            )


def test_vjepa2_adapter_preserves_existing_video_references(tmp_path):
    """If a pipeline row already carries an absolute or URI-style media
    path, the adapter must surface that reference verbatim — the upstream
    encoder is expected to load by reference, not from a relocated copy.
    """

    adapter = _resolve_adapter()
    absolute_path = tmp_path / "src" / "reference_clip.mp4"
    _write_fake_video(absolute_path)
    manifest = [
        _video_pipeline_row(absolute_path, idx=0),
        {
            "example_id": "vid_remote_001",
            "media_paths": ["s3://bucket/key/remote_clip.mp4"],
            "modality": "video",
            "metadata": {},
        },
    ]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    text_blob = _read_text_outputs(out_dir)
    assert str(absolute_path) in text_blob, (
        "Absolute filesystem reference must be preserved verbatim in the "
        "backend-ready manifest."
    )
    assert "s3://bucket/key/remote_clip.mp4" in text_blob, (
        "URI-style remote reference must be preserved verbatim in the "
        "backend-ready manifest."
    )


def test_vjepa2_adapter_does_not_mutate_source_manifest(tmp_path):
    adapter = _resolve_adapter()
    source_video = _write_fake_video(tmp_path / "src" / "clip_0.mp4")
    manifest = [
        _video_pipeline_row(source_video, idx=0),
        _video_pipeline_row(source_video, idx=1),
    ]
    snapshot = copy.deepcopy(manifest)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    assert manifest == snapshot, (
        "V-JEPA2 adapter must not mutate the source pipeline manifest."
    )


def test_vjepa2_adapter_handles_multiple_rows(tmp_path):
    """Multi-row manifest must produce one logical record per row.

    The adapter is free to write a single file (e.g. one JSONL) or one
    file per row; either way, each row's reference must appear in the
    output.
    """

    adapter = _resolve_adapter()
    clip_a = _write_fake_video(tmp_path / "src" / "clip_a.mp4")
    clip_b = _write_fake_video(tmp_path / "src" / "clip_b.mp4")
    manifest = [
        _video_pipeline_row(clip_a, idx=0),
        _video_pipeline_row(clip_b, idx=1),
    ]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    text_blob = _read_text_outputs(out_dir)
    assert str(clip_a) in text_blob and str(clip_b) in text_blob, (
        "Backend-ready manifest must reference every source video path. "
        f"Saw text blob excerpt: {text_blob[:500]!r}"
    )

    # Records ought to be enumerable — try to find a JSON/JSONL artifact and
    # confirm at least two records are present.
    record_count = 0
    for path in _gather_output_files(out_dir):
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                json.loads(line)  # must be valid JSON
                record_count += 1
        elif path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, list):
                record_count += len(payload)
            elif isinstance(payload, dict):
                # Some shapes nest examples under a key.
                for value in payload.values():
                    if isinstance(value, list):
                        record_count += len(value)
                        break
    if record_count:
        assert record_count >= len(manifest), (
            "V-JEPA2 backend-ready manifest must contain at least one "
            f"record per source row; saw {record_count} for {len(manifest)} rows."
        )
