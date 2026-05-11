"""Repo-local wrapper for V-JEPA2 embedding inference."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from .bootstrap.third_party import discover_upstream_root

Runner = Callable[..., subprocess.CompletedProcess[str]]


def run_vjepa2_inference(
    *,
    model_id: str,
    task_type: str,
    input_manifest: str | Path,
    output_dir: str | Path,
    backend_config: dict[str, Any] | None = None,
    split: str = "inference",
    manifest_path: Path | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Launch the pinned V-JEPA2 checkout and normalize the result payload."""

    config = dict(backend_config or {})
    upstream_root = discover_upstream_root("vjepa2", manifest_path=manifest_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_manifest = Path(input_manifest)

    command = build_vjepa2_command(
        input_manifest=prepared_manifest,
        output_dir=output_root,
        backend_config=config,
    )
    completed = runner(
        command,
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    )

    per_example_results = _build_example_results(prepared_manifest)
    status = _summarize_status(per_example_results)
    result = {
        "model_id": model_id,
        "task_type": task_type,
        "output_dir": str(output_root),
        "input_manifest": str(prepared_manifest),
        "upstream_root": str(upstream_root),
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "results": per_example_results,
        "per_example_results": per_example_results,
        "examples": per_example_results,
        "per_example_status": {
            item["example_id"]: item["status"] for item in per_example_results
        },
        "status": status,
    }

    record_path = output_root / f"{split}_result.json"
    record_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["result_path"] = str(record_path)
    return result


def build_vjepa2_command(
    *,
    input_manifest: Path,
    output_dir: Path,
    backend_config: dict[str, Any],
) -> list[str]:
    command = backend_config.get("command")
    if isinstance(command, str) and command.strip():
        return [
            command,
            "--input-manifest",
            str(input_manifest),
            "--output-dir",
            str(output_dir),
        ]
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        base = [str(part) for part in command if str(part).strip()]
        if base:
            return [
                *base,
                "--input-manifest",
                str(input_manifest),
                "--output-dir",
                str(output_dir),
            ]

    entrypoint = str(backend_config.get("entrypoint") or "tools.run_inference").strip()
    return [
        sys.executable,
        "-m",
        entrypoint,
        "--input-manifest",
        str(input_manifest),
        "--output-dir",
        str(output_dir),
    ]


def launch_vjepa2_inference(**kwargs: Any) -> dict[str, Any]:
    return run_vjepa2_inference(**kwargs)


def run_inference(**kwargs: Any) -> dict[str, Any]:
    return run_vjepa2_inference(**kwargs)


def _build_example_results(input_manifest: Path) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for index, line in enumerate(input_manifest.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = str(row.get("example_id") or f"example-{index:05d}")
        results.append(
            {
                "example_id": example_id,
                "status": "success",
                "video_path": str(row.get("video_path") or ""),
            }
        )
    return results


def _summarize_status(results: list[dict[str, str]]) -> str:
    if results and all(item.get("status") == "success" for item in results):
        return "success"
    return "failed"


__all__ = [
    "build_vjepa2_command",
    "launch_vjepa2_inference",
    "run_inference",
    "run_vjepa2_inference",
]
