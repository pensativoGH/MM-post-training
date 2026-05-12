"""Repo-local wrapper for Wan video generation inference."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from verl_post_training.bootstrap.third_party import discover_upstream_root

Runner = Callable[..., subprocess.CompletedProcess[str]]


def run_wan_generation(
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
    """Generate Wan artifacts or a deterministic local dry-run contract."""

    config = dict(backend_config or {})
    upstream_root = discover_upstream_root("wan22", manifest_path=manifest_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_manifest = Path(input_manifest)

    command = build_wan_command(
        model_id=model_id,
        input_manifest=prepared_manifest,
        output_dir=output_root,
        backend_config=config,
    )

    completed: subprocess.CompletedProcess[str] | None = None
    if _should_launch_upstream(config):
        completed = runner(
            command,
            cwd=upstream_root,
            check=True,
            capture_output=True,
            text=True,
        )

    per_example_results = _write_artifacts(
        input_manifest=prepared_manifest,
        output_dir=output_root,
        model_id=model_id,
        task_type=task_type,
    )
    status = _summarize_status(per_example_results)
    result = {
        "model_id": model_id,
        "task_type": task_type,
        "output_dir": str(output_root),
        "input_manifest": str(prepared_manifest),
        "upstream_root": str(upstream_root),
        "command": list(command),
        "returncode": completed.returncode if completed is not None else 0,
        "stdout": completed.stdout if completed is not None else "",
        "stderr": completed.stderr if completed is not None else "",
        "per_example": per_example_results,
        "results": per_example_results,
        "per_example_results": per_example_results,
        "examples": per_example_results,
        "outputs": per_example_results,
        "items": per_example_results,
        "generated_artifacts": [
            item["artifact_path"] for item in per_example_results if item.get("artifact_path")
        ],
        "artifact_paths": [
            item["artifact_path"] for item in per_example_results if item.get("artifact_path")
        ],
        "input_example_ids": [item["example_id"] for item in per_example_results],
        "per_example_status": {
            item["example_id"]: item["status"] for item in per_example_results
        },
        "status": status,
    }

    metadata_path = output_root / f"{split}_metadata.json"
    metadata = {
        "model_id": model_id,
        "task_type": task_type,
        "input_manifest": str(prepared_manifest),
        "output_dir": str(output_root),
        "input_example_ids": result["input_example_ids"],
        "examples": per_example_results,
        "generated_artifacts": result["generated_artifacts"],
        "artifact_paths": result["artifact_paths"],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result["metadata_path"] = str(metadata_path)
    result["result_path"] = str(metadata_path)
    return result


def build_wan_command(
    *,
    model_id: str,
    input_manifest: Path,
    output_dir: Path,
    backend_config: dict[str, Any],
) -> list[str]:
    command = backend_config.get("command")
    if isinstance(command, str) and command.strip():
        return [
            command,
            "--model-id",
            model_id,
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
                "--model-id",
                model_id,
                "--input-manifest",
                str(input_manifest),
                "--output-dir",
                str(output_dir),
            ]

    entrypoint = str(backend_config.get("entrypoint") or "generate").strip()
    return [
        sys.executable,
        str(Path("wan") / f"{entrypoint}.py") if not entrypoint.endswith(".py") else entrypoint,
        "--model-id",
        model_id,
        "--input-manifest",
        str(input_manifest),
        "--output-dir",
        str(output_dir),
    ]


def launch_wan_generation(**kwargs: Any) -> dict[str, Any]:
    return run_wan_generation(**kwargs)


def run_generation(**kwargs: Any) -> dict[str, Any]:
    return run_wan_generation(**kwargs)


def _should_launch_upstream(config: dict[str, Any]) -> bool:
    if config.get("dry_run") is False or config.get("simulate") is False:
        return True
    return bool(config.get("launch_upstream"))


def _write_artifacts(
    *,
    input_manifest: Path,
    output_dir: Path,
    model_id: str,
    task_type: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, line in enumerate(input_manifest.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = str(row.get("example_id") or f"example-{index:05d}")
        artifact_path = Path(row.get("output_path") or output_dir / "generated" / f"{example_id}.mp4")
        if not artifact_path.is_absolute():
            artifact_path = output_dir / artifact_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(
            (
                "WAN_GENERATION_PLACEHOLDER\n"
                f"model_id={model_id}\n"
                f"task_type={task_type}\n"
                f"example_id={example_id}\n"
                f"prompt={row.get('prompt', '')}\n"
            ).encode("utf-8")
        )
        sidecar_path = artifact_path.with_suffix(f"{artifact_path.suffix}.json")
        sidecar = {
            "model_id": model_id,
            "task_type": task_type,
            "example_id": example_id,
            "prompt": row.get("prompt", ""),
            "conditioning": row.get("conditioning", {}),
            "artifact_path": str(artifact_path),
        }
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        results.append(
            {
                "example_id": example_id,
                "status": "success",
                "prompt": str(row.get("prompt") or ""),
                "artifact_path": str(artifact_path),
                "generated_path": str(artifact_path),
                "metadata_path": str(sidecar_path),
            }
        )
    return results


def _summarize_status(results: list[dict[str, Any]]) -> str:
    if results and all(item.get("status") == "success" for item in results):
        return "success"
    return "failed"


__all__ = [
    "build_wan_command",
    "launch_wan_generation",
    "run_generation",
    "run_wan_generation",
]
