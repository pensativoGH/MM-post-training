"""Run a minimal V-JEPA2 embedding inference sanity check.

This script is intentionally narrow: it verifies that the pinned upstream
V-JEPA2 checkout can load a local checkpoint, decode a server video, and
produce an embedding tensor on CUDA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("third_party/vjepa2"),
        help="Path to the pinned V-JEPA2 checkout.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/ubuntu/hf_models/vjepa2/direct_checkpoints/vitl.pt"),
        help="Path to a V-JEPA2 checkpoint.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("/home/ubuntu/datasets/robovqa_cosmos_sft/robovqa/clips/18153780628982899168.mp4"),
        help="Path to a local MP4 used for the sanity check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/inference_sanity/vjepa2_embedding_summary.json"),
        help="Where to write the JSON summary.",
    )
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))

    from src.hub.backbones import vjepa2_vit_large
    import src.datasets.utils.video.transforms as video_transforms
    import src.datasets.utils.video.volume_transforms as volume_transforms

    encoder, _ = vjepa2_vit_large(pretrained=False)
    state = torch.load(args.model_path, weights_only=True, map_location="cpu")
    state = state.get("target_encoder") or state.get("encoder") or state
    state = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state.items()
    }
    load_message = encoder.load_state_dict(state, strict=False)
    encoder = encoder.cuda().eval().to(dtype=torch.float16)

    frames = _read_frames(args.video_path, args.frames)
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    transform = video_transforms.Compose(
        [
            video_transforms.Resize(
                int(256.0 / 224 * args.image_size),
                interpolation="bilinear",
            ),
            video_transforms.CenterCrop(size=(args.image_size, args.image_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ]
    )
    inputs = transform(video).unsqueeze(0).cuda().to(dtype=torch.float16)
    with torch.inference_mode():
        outputs = encoder(inputs)

    summary = {
        "model_path": str(args.model_path),
        "video_path": str(args.video_path),
        "frames": len(frames),
        "input_shape": list(inputs.shape),
        "output_shape": list(outputs.shape),
        "dtype": str(outputs.dtype),
        "mean": float(outputs.float().mean().cpu()),
        "std": float(outputs.float().std().cpu()),
        "missing_keys": len(load_message.missing_keys),
        "unexpected_keys": len(load_message.unexpected_keys),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _read_frames(video_path: Path, frame_count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total:
        indices = set(np.linspace(0, max(total - 1, 0), frame_count).astype(int).tolist())
    else:
        indices = set(range(frame_count))

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in indices:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        index += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    while len(frames) < frame_count:
        frames.append(frames[-1])
    return frames[:frame_count]


if __name__ == "__main__":
    raise SystemExit(main())
