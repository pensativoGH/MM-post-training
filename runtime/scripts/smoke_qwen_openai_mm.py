#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request

from PIL import Image


def build_tiny_jpeg_data_url() -> str:
    img = Image.new("RGB", (2, 2), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a local OpenAI-compatible multimodal endpoint.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    image_url = build_tiny_jpeg_data_url()

    payload = {
        "model": args.model,
        "max_tokens": 64,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with exactly: multimodal-ok"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }

    request = urllib.request.Request(
        url=args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return exc.code or 1
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1

    content = body["choices"][0]["message"].get("content")
    print(json.dumps({"content": content, "model": body.get("model")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
