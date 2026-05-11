"""Acceptance tests for the chat RL dataset adapter (M4).

The chat RL adapter is introduced in M4. It must take an existing pipeline
manifest and produce a backend-consumable RL dataset (the VERL JSONL shape
used by ``RL/scripts/prepare_rl_dataset.py``) without mutating the source.

These tests pin the contract before implementation lands:

  * the registry resolution path (``get_dataset_adapter("chat_rl")``)
  * the input shape (pipeline rows with ``messages``, ``media_paths``,
    ``modality``, and a JSON-encoded gold response on the final assistant turn)
  * the output shape (JSONL records with ``prompt`` messages, ``images``,
    ``videos``, ``data_source``, ``reward_model.ground_truth``, and
    ``extra_info``)
  * non-mutation of the input rows
  * the ability to handle procedural smoke video-native rows.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest


def _resolve_adapter():
    from verl_post_training.adapters.dataset import get_dataset_adapter

    return get_dataset_adapter("chat_rl")


def _video_smoke_row(idx: int = 0) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "Predict the next step."},
            {"role": "user", "content": f"Given this clip, what comes next ({idx})?"},
            {
                "role": "assistant",
                "content": json.dumps({"next_step": f"step_{idx}"}),
            },
        ],
        "media_paths": [f"/data/procedural/video_native/clip_{idx}.mp4"],
        "modality": "video",
    }


def _image_smoke_row(idx: int = 0) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"What comes after frame {idx}?"},
            {
                "role": "assistant",
                "content": json.dumps({"next_step": f"image_step_{idx}"}),
            },
        ],
        "media_paths": [f"/data/procedural/image_native/frame_{idx}.jpg"],
        "modality": "image",
    }


def _read_rl_jsonl(output_dir: Path) -> list[dict[str, Any]]:
    jsonl_files = sorted(p for p in output_dir.rglob("*.jsonl") if p.is_file())
    assert jsonl_files, (
        f"chat_rl adapter must write at least one .jsonl file under {output_dir}; "
        "found none."
    )
    records: list[dict[str, Any]] = []
    for path in jsonl_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    assert records, "Expected at least one RL record in the chat_rl adapter output."
    return records


def test_chat_rl_adapter_is_resolvable_from_registry():
    adapter = _resolve_adapter()
    assert adapter is not None
    assert hasattr(adapter, "prepare"), (
        "chat_rl adapter must expose a callable `prepare` method."
    )


def test_chat_rl_adapter_writes_backend_consumable_jsonl(tmp_path):
    adapter = _resolve_adapter()
    rows = [_video_smoke_row(0), _video_smoke_row(1)]

    adapter.prepare(
        pipeline_manifest=rows,
        output_dir=tmp_path,
        data_source="procedural_grpo/video_next_step",
    )

    records = _read_rl_jsonl(tmp_path)
    assert len(records) == len(rows)

    sample = records[0]
    for required_field in (
        "prompt",
        "images",
        "videos",
        "data_source",
        "reward_model",
        "extra_info",
    ):
        assert required_field in sample, (
            f"RL JSONL record missing required field: {required_field!r}"
        )

    assert sample["data_source"] == "procedural_grpo/video_next_step"
    assert isinstance(sample["prompt"], list) and sample["prompt"], (
        "RL `prompt` must be a non-empty list of messages."
    )
    # Final assistant gold response must be stripped from the prompt and
    # promoted into the reward model ground truth.
    assert sample["prompt"][-1]["role"] != "assistant", (
        "The final assistant gold response must not appear in the RL prompt."
    )
    assert "ground_truth" in sample["reward_model"], (
        "RL record must expose `reward_model.ground_truth`."
    )
    assert sample["reward_model"]["ground_truth"] == {"next_step": "step_0"}


def test_chat_rl_adapter_does_not_mutate_source_manifest(tmp_path):
    adapter = _resolve_adapter()
    rows = [_video_smoke_row(0), _image_smoke_row(1)]
    snapshot = copy.deepcopy(rows)

    adapter.prepare(
        pipeline_manifest=rows,
        output_dir=tmp_path,
        data_source="procedural_grpo/mixed",
    )

    assert rows == snapshot, (
        "chat_rl adapter must not mutate the source pipeline manifest."
    )


def test_chat_rl_adapter_handles_video_native_procedural_smoke(tmp_path):
    """The procedural smoke video-native manifest must prepare successfully."""

    adapter = _resolve_adapter()
    rows = [_video_smoke_row(i) for i in range(2)]

    adapter.prepare(
        pipeline_manifest=rows,
        output_dir=tmp_path,
        data_source="procedural_grpo/video_next_step",
    )

    records = _read_rl_jsonl(tmp_path)
    assert len(records) == 2

    first = records[0]
    assert first["videos"], "Video rows must produce a non-empty `videos` list."
    # `videos` entries must include the source video path in a structured form
    # so VERL can resolve the asset.
    video_paths = [
        entry.get("video") if isinstance(entry, dict) else entry
        for entry in first["videos"]
    ]
    assert any(
        path and path.endswith("clip_0.mp4") for path in video_paths
    ), f"Expected the source video path in videos entries; got {first['videos']!r}"


def test_chat_rl_adapter_handles_robovqa_style_video_row(tmp_path):
    """A RoboVQA-style pipeline row must prepare successfully into RL JSONL."""

    adapter = _resolve_adapter()
    robovqa_row = {
        "messages": [
            {"role": "user", "content": "What is the robot doing?"},
            {
                "role": "assistant",
                "content": json.dumps({"answer": "Picking up the cup."}),
            },
        ],
        "media_paths": ["/data/robovqa/clips/episode_0001.mp4"],
        "modality": "video",
    }

    adapter.prepare(
        pipeline_manifest=[robovqa_row],
        output_dir=tmp_path,
        data_source="robovqa_cosmos_rl",
    )

    records = _read_rl_jsonl(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["data_source"] == "robovqa_cosmos_rl"
    assert record["reward_model"]["ground_truth"] == {"answer": "Picking up the cup."}
    assert record["videos"], (
        "RoboVQA video row must produce a populated `videos` list in RL output."
    )
