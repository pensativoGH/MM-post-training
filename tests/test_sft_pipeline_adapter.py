"""Acceptance tests for the chat SFT dataset adapter (M4).

The chat SFT adapter is introduced in M4. It must take an existing pipeline
manifest (the in-memory rows that the multimodal data pipeline exposes to
trainers) and produce a backend-consumable ShareGPT dataset that
LLaMA-Factory can ingest. The source manifest must remain untouched.

These tests pin the contract before implementation lands. They are
deliberately permissive about *where* and *how many* files the adapter writes,
so the writer can choose a sensible layout; but they pin:

  * the registry resolution path (``get_dataset_adapter("chat_sft")``)
  * the input shape (a list of pipeline rows with ``messages``,
    ``media_paths``, ``modality``)
  * the output shape (ShareGPT JSON records with ``conversations``,
    ``images``, ``videos``, and ``system`` fields, where the final turn is the
    assistant)
  * non-mutation of the input rows
  * the ability to handle both image-modality (RoboVQA-style) and
    video-modality (procedural smoke-style) rows.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest


def _resolve_adapter():
    from verl_post_training.adapters.dataset import get_dataset_adapter

    return get_dataset_adapter("chat_sft")


def _image_pipeline_row(idx: int = 0) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"What is in image {idx}?"},
            {"role": "assistant", "content": "A robot manipulating an object."},
        ],
        "media_paths": [f"/data/pipeline/images/sample_{idx}.jpg"],
        "modality": "image",
    }


def _video_pipeline_row(idx: int = 0) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"Describe video clip {idx}."},
            {"role": "assistant", "content": "The robot picks up the cup."},
        ],
        "media_paths": [f"/data/pipeline/videos/clip_{idx}.mp4"],
        "modality": "video",
    }


def _read_sft_output(output_dir: Path) -> list[dict[str, Any]]:
    json_files = sorted(p for p in output_dir.rglob("*.json") if p.is_file())
    assert json_files, (
        f"chat_sft adapter must write at least one JSON file under {output_dir}; "
        "found none."
    )
    examples: list[dict[str, Any]] = []
    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            examples.extend(payload)
    assert examples, (
        "Expected at least one ShareGPT record in the chat_sft adapter output."
    )
    return examples


def test_chat_sft_adapter_is_resolvable_from_registry():
    adapter = _resolve_adapter()
    assert adapter is not None
    assert hasattr(adapter, "prepare"), (
        "chat_sft adapter must expose a callable `prepare` method."
    )


def test_chat_sft_adapter_writes_backend_consumable_sharegpt(tmp_path):
    adapter = _resolve_adapter()
    rows = [_image_pipeline_row(0), _image_pipeline_row(1)]

    adapter.prepare(pipeline_manifest=rows, output_dir=tmp_path)

    examples = _read_sft_output(tmp_path)
    assert len(examples) == len(rows)

    sample = examples[0]
    for required_column in ("conversations", "images", "videos", "system"):
        assert required_column in sample, (
            f"ShareGPT record missing required column: {required_column!r}"
        )

    assert isinstance(sample["conversations"], list) and sample["conversations"], (
        "ShareGPT conversations must be a non-empty list."
    )
    assert sample["conversations"][-1]["from"] == "gpt", (
        "Final turn in a SFT ShareGPT record must be the assistant ('gpt')."
    )

    roles = {turn["from"] for turn in sample["conversations"]}
    assert "human" in roles, "ShareGPT record must include at least one human turn."

    assert sample["images"], "Image-modality row must produce a populated images list."
    assert sample["videos"] == [], "Image-modality row must not populate videos list."


def test_chat_sft_adapter_does_not_mutate_source_manifest(tmp_path):
    adapter = _resolve_adapter()
    rows = [_image_pipeline_row(0), _video_pipeline_row(1)]
    snapshot = copy.deepcopy(rows)

    adapter.prepare(pipeline_manifest=rows, output_dir=tmp_path)

    assert rows == snapshot, (
        "chat_sft adapter must not mutate the source pipeline manifest."
    )


def test_chat_sft_adapter_handles_robovqa_style_row(tmp_path):
    """A representative RoboVQA pipeline row must prepare successfully."""

    adapter = _resolve_adapter()
    robovqa_row = {
        "messages": [
            {"role": "user", "content": "What is the robot doing?"},
            {"role": "assistant", "content": "Picking up the red cup."},
        ],
        "media_paths": ["/data/robovqa/clips/episode_0001.mp4"],
        "modality": "video",
    }

    adapter.prepare(pipeline_manifest=[robovqa_row], output_dir=tmp_path)

    examples = _read_sft_output(tmp_path)
    assert len(examples) == 1
    record = examples[0]
    assert record["videos"] == ["/data/robovqa/clips/episode_0001.mp4"]
    assert record["images"] == []

    human_turns = [t for t in record["conversations"] if t["from"] == "human"]
    assert human_turns, "Expected at least one human turn in RoboVQA SFT record."
    assert any("<video>" in turn["value"] for turn in human_turns), (
        "Video-modality rows must inject a <video> placeholder into a human turn."
    )


def test_chat_sft_adapter_handles_procedural_smoke_row(tmp_path):
    """A representative procedural smoke pipeline row must prepare successfully."""

    adapter = _resolve_adapter()
    smoke_row = {
        "messages": [
            {"role": "system", "content": "You predict the next step."},
            {"role": "user", "content": "Given this video, what comes next?"},
            {"role": "assistant", "content": "{\"next_step\": \"open the drawer\"}"},
        ],
        "media_paths": ["/data/procedural/smoke/video_0.mp4"],
        "modality": "video",
    }

    adapter.prepare(pipeline_manifest=[smoke_row], output_dir=tmp_path)

    examples = _read_sft_output(tmp_path)
    assert len(examples) == 1
    record = examples[0]
    assert record["videos"] == ["/data/procedural/smoke/video_0.mp4"]
    # System prompt content must survive the conversion.
    assert "next step" in record["system"].lower()
    # The final turn must remain the assistant gold response.
    assert record["conversations"][-1]["from"] == "gpt"
    assert "next_step" in record["conversations"][-1]["value"]
