"""M7 acceptance: the Wan2.2 dataset adapter writes backend-ready
conditioning manifests for supported pipeline inputs.

This file pins the first M7 acceptance criterion (quoted from the approved
plan):

    given a pipeline manifest with supported conditioning inputs, the Wan
    adapter writes a backend-ready manifest describing prompt, media
    references, and output target locations

The adapter is expected to register against the ``wan`` key already
exposed by the M4 adapter registry. Concretely, these tests pin:

* ``get_dataset_adapter("wan")`` resolves to a real adapter (not the M4
  stub) that exposes a callable ``prepare`` method
* given a pipeline manifest with prompt + (optional) media conditioning
  rows, ``prepare`` writes at least one machine-readable artifact under
  ``output_dir``
* the written conditioning manifest references each row's prompt
* the written conditioning manifest references each row's conditioning
  media path verbatim (image / video reference) so the upstream loader
  can consume it by reference
* the written conditioning manifest declares an output target location
  per row so the runtime adapter knows where to land each generated
  artifact
* the adapter does not copy or rewrite source media bytes when the source
  manifest is reachable
* the adapter does not mutate the source rows it was handed
* a text-only (prompt-only) row is also accepted as a supported
  conditioning input
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


_IMAGE_SENTINEL = b"\x89PNG\r\n\x1a\n-WAN2.2-COND-IMAGE-SENTINEL-AAAA"
_VIDEO_SENTINEL = b"\x00\x00\x00\x18ftypmp42-WAN2.2-COND-VIDEO-SENTINEL"


def _resolve_adapter():
    from verl_post_training.adapters.dataset import get_dataset_adapter

    adapter = get_dataset_adapter("wan")
    assert adapter is not None, (
        "get_dataset_adapter('wan') must resolve to a real adapter once "
        "M7 lands; got None."
    )
    assert hasattr(adapter, "prepare"), (
        "Wan dataset adapter must expose a callable `prepare` method; "
        f"got: {adapter!r}"
    )
    return adapter


def _write_fake_media(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _t2v_pipeline_row(idx: int = 0) -> dict[str, Any]:
    """Text-to-video conditioning row (prompt-only)."""

    return {
        "example_id": f"wan_t2v_{idx:03d}",
        "prompt": (
            f"A cinematic shot of a fox stalking through a snowy forest, "
            f"clip {idx}."
        ),
        "modality": "text",
        "conditioning_media_paths": [],
        "metadata": {"clip_index": idx, "task": "t2v"},
    }


def _i2v_pipeline_row(image_path: Path, idx: int = 0) -> dict[str, Any]:
    """Image-to-video conditioning row."""

    return {
        "example_id": f"wan_i2v_{idx:03d}",
        "prompt": f"Animate the input frame for clip {idx}.",
        "modality": "image",
        "conditioning_media_paths": [str(image_path)],
        "metadata": {"clip_index": idx, "task": "i2v"},
    }


def _v2v_pipeline_row(video_path: Path, idx: int = 0) -> dict[str, Any]:
    """Video-to-video conditioning row."""

    return {
        "example_id": f"wan_v2v_{idx:03d}",
        "prompt": f"Re-light the input clip {idx} with golden-hour lighting.",
        "modality": "video",
        "conditioning_media_paths": [str(video_path)],
        "metadata": {"clip_index": idx, "task": "v2v"},
    }


def _gather_output_files(output_dir: Path) -> list[Path]:
    return [p for p in output_dir.rglob("*") if p.is_file()]


def _read_text_outputs(output_dir: Path) -> str:
    """Concatenated text of every text-ish artifact the adapter wrote."""

    chunks: list[str] = []
    for path in _gather_output_files(output_dir):
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
    return "\n".join(chunks)


def _parse_records(output_dir: Path) -> list[dict[str, Any]]:
    """Best-effort: parse JSON / JSONL artifacts the adapter wrote.

    The writer is free to choose JSON, JSONL, YAML, or another machine-
    readable format — these helpers parse the JSON-shaped ones so we can
    make targeted assertions against per-row records.
    """

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
                # Common shapes: {"examples": [...]} or {"records": [...]}
                for value in payload.values():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                records.append(item)
                if not records:
                    records.append(payload)
    return records


def _call_prepare(adapter, *, manifest, output_dir):
    """Call the adapter's prepare hook, tolerating a few signature shapes."""

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
    try:
        return adapter.prepare(manifest, output_dir)
    except TypeError as exc:  # pragma: no cover - reported below
        last_err = exc
    raise AssertionError(
        "Wan adapter `prepare` must accept the M4 signature "
        "(pipeline_manifest=..., output_dir=...). Last TypeError: "
        f"{last_err!r}"
    )


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------


def test_wan_adapter_is_resolvable_from_registry():
    """The M4 stub key ``wan`` must resolve to a real adapter by M7 — not
    the placeholder object the M4 milestone permits.
    """

    adapter = _resolve_adapter()
    assert callable(getattr(adapter, "prepare", None)), (
        "Wan adapter must implement a callable `prepare` hook so the "
        "control plane can drive dataset preparation through it."
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 1: backend-ready conditioning manifest
# ---------------------------------------------------------------------------


def test_wan_adapter_writes_backend_ready_manifest_for_t2v(tmp_path):
    """For a prompt-only row, the adapter must emit at least one
    machine-readable artifact under ``output_dir`` that downstream
    backends can consume.
    """

    adapter = _resolve_adapter()
    manifest = [_t2v_pipeline_row(idx=0)]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    files = _gather_output_files(out_dir)
    assert files, (
        f"Wan adapter must write at least one artifact under {out_dir}; "
        "found none."
    )

    text_blob = _read_text_outputs(out_dir)
    assert "fox stalking through a snowy forest" in text_blob, (
        "Backend-ready conditioning manifest must reference each row's "
        f"prompt; instead found:\n{text_blob[:500]}"
    )


def test_wan_adapter_writes_backend_ready_manifest_for_i2v(tmp_path):
    """For an image-conditioned row, the conditioning manifest must
    reference both the prompt and the conditioning image path verbatim.
    """

    adapter = _resolve_adapter()
    image_path = _write_fake_media(
        tmp_path / "src" / "frame_0.png", _IMAGE_SENTINEL
    )
    manifest = [_i2v_pipeline_row(image_path, idx=0)]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    text_blob = _read_text_outputs(out_dir)
    assert "Animate the input frame for clip 0." in text_blob, (
        "Conditioning manifest must reference the row prompt; "
        f"instead found:\n{text_blob[:500]}"
    )
    assert str(image_path) in text_blob, (
        "Conditioning manifest must reference the conditioning image path "
        f"{str(image_path)!r}; instead found:\n{text_blob[:500]}"
    )


def test_wan_adapter_writes_backend_ready_manifest_for_v2v(tmp_path):
    """For a video-conditioned row, the conditioning manifest must
    reference both the prompt and the conditioning video path verbatim.
    """

    adapter = _resolve_adapter()
    video_path = _write_fake_media(
        tmp_path / "src" / "clip_0.mp4", _VIDEO_SENTINEL
    )
    manifest = [_v2v_pipeline_row(video_path, idx=0)]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    text_blob = _read_text_outputs(out_dir)
    assert "golden-hour lighting" in text_blob, (
        "Conditioning manifest must reference the row prompt; "
        f"instead found:\n{text_blob[:500]}"
    )
    assert str(video_path) in text_blob, (
        "Conditioning manifest must reference the conditioning video path "
        f"{str(video_path)!r}; instead found:\n{text_blob[:500]}"
    )


def test_wan_adapter_records_output_target_per_row(tmp_path):
    """Acceptance criterion: "output target locations". The conditioning
    manifest must declare *where* the generated artifact for each row
    will land — without that, the runtime adapter cannot place outputs
    deterministically.
    """

    adapter = _resolve_adapter()
    image_path = _write_fake_media(
        tmp_path / "src" / "frame_0.png", _IMAGE_SENTINEL
    )
    manifest = [
        _t2v_pipeline_row(idx=0),
        _i2v_pipeline_row(image_path, idx=1),
    ]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    records = _parse_records(out_dir)
    assert records, (
        "Wan conditioning manifest must be machine-readable (JSON/JSONL); "
        "no JSON-parseable records were found under output_dir. "
        f"Files: {[str(p) for p in _gather_output_files(out_dir)]!r}"
    )

    target_keys = (
        "output_path",
        "output_target",
        "output_artifact",
        "artifact_path",
        "target_path",
        "generated_path",
        "save_path",
        "output_video_path",
    )
    rows_with_target = [
        rec
        for rec in records
        if any(rec.get(key) for key in target_keys)
    ]
    assert rows_with_target, (
        "Wan conditioning manifest rows must declare an output target "
        f"location under one of {target_keys!r}; saw records: {records!r}"
    )
    assert len(rows_with_target) >= len(manifest), (
        "Each input row must have an output target location in the "
        f"conditioning manifest; saw {len(rows_with_target)} target rows "
        f"for {len(manifest)} input rows."
    )


def test_wan_adapter_does_not_duplicate_conditioning_media_bytes(tmp_path):
    """If the source conditioning media is reachable, the adapter must
    not copy or rewrite the bytes anywhere under ``output_dir`` — the
    upstream loader is expected to load by reference.
    """

    adapter = _resolve_adapter()
    image_path = _write_fake_media(
        tmp_path / "src" / "frame_0.png", _IMAGE_SENTINEL
    )
    video_path = _write_fake_media(
        tmp_path / "src" / "clip_0.mp4", _VIDEO_SENTINEL
    )
    manifest = [
        _i2v_pipeline_row(image_path, idx=0),
        _v2v_pipeline_row(video_path, idx=1),
    ]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    for produced in _gather_output_files(out_dir):
        try:
            contents = produced.read_bytes()
        except OSError:
            continue
        assert _IMAGE_SENTINEL not in contents, (
            f"Wan adapter copied source image bytes into {produced} — the "
            "adapter must reference the source media rather than "
            "duplicating it."
        )
        assert _VIDEO_SENTINEL not in contents, (
            f"Wan adapter copied source video bytes into {produced} — the "
            "adapter must reference the source media rather than "
            "duplicating it."
        )


def test_wan_adapter_preserves_existing_media_references(tmp_path):
    """If a pipeline row already carries an absolute or URI-style media
    path, the adapter must surface that reference verbatim — the upstream
    loader is expected to load by reference, not from a relocated copy.
    """

    adapter = _resolve_adapter()
    absolute_image = _write_fake_media(
        tmp_path / "src" / "ref_frame.png", _IMAGE_SENTINEL
    )
    manifest = [
        _i2v_pipeline_row(absolute_image, idx=0),
        {
            "example_id": "wan_remote_001",
            "prompt": "Animate this remote frame.",
            "modality": "image",
            "conditioning_media_paths": ["s3://bucket/key/remote_frame.png"],
            "metadata": {},
        },
    ]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    text_blob = _read_text_outputs(out_dir)
    assert str(absolute_image) in text_blob, (
        "Absolute filesystem reference must be preserved verbatim in the "
        "backend-ready conditioning manifest."
    )
    assert "s3://bucket/key/remote_frame.png" in text_blob, (
        "URI-style remote reference must be preserved verbatim in the "
        "backend-ready conditioning manifest."
    )


def test_wan_adapter_does_not_mutate_source_manifest(tmp_path):
    adapter = _resolve_adapter()
    image_path = _write_fake_media(
        tmp_path / "src" / "frame_0.png", _IMAGE_SENTINEL
    )
    manifest = [
        _t2v_pipeline_row(idx=0),
        _i2v_pipeline_row(image_path, idx=1),
    ]
    snapshot = copy.deepcopy(manifest)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    assert manifest == snapshot, (
        "Wan adapter must not mutate the source pipeline manifest."
    )


def test_wan_adapter_handles_multiple_rows(tmp_path):
    """Multi-row mixed conditioning manifest must produce one logical
    record per row — every row's prompt must reach the conditioning
    manifest, and every conditioning media path must reach it too.
    """

    adapter = _resolve_adapter()
    image_path = _write_fake_media(
        tmp_path / "src" / "frame_0.png", _IMAGE_SENTINEL
    )
    video_path = _write_fake_media(
        tmp_path / "src" / "clip_0.mp4", _VIDEO_SENTINEL
    )
    manifest = [
        _t2v_pipeline_row(idx=0),
        _i2v_pipeline_row(image_path, idx=1),
        _v2v_pipeline_row(video_path, idx=2),
    ]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _call_prepare(adapter, manifest=manifest, output_dir=out_dir)

    text_blob = _read_text_outputs(out_dir)
    for row in manifest:
        assert row["prompt"] in text_blob, (
            "Backend-ready manifest must reference every row's prompt. "
            f"Missing: {row['prompt']!r}; saw text blob excerpt: "
            f"{text_blob[:500]!r}"
        )
    assert str(image_path) in text_blob, (
        "Conditioning manifest must reference image conditioning path."
    )
    assert str(video_path) in text_blob, (
        "Conditioning manifest must reference video conditioning path."
    )

    records = _parse_records(out_dir)
    if records:
        assert len(records) >= len(manifest), (
            "Wan conditioning manifest must contain at least one record "
            f"per source row; saw {len(records)} for {len(manifest)} rows."
        )
