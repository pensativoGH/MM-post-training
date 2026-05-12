"""Capability reporting for repo-owned launch backends."""

from __future__ import annotations

from typing import Any


class CapabilityUnavailableError(RuntimeError):
    """Raised when a backend is known but not runnable in this environment."""


_DREAMDOJO_UNAVAILABLE_REASON = (
    "DreamDojo execution is deferred pending local-first integration approval."
)


def report_capabilities() -> dict[str, dict[str, Any]]:
    """Return deterministic local capability status by backend key."""

    return {
        "dreamdojo": {
            "name": "dreamdojo",
            "available": False,
            "reason": _DREAMDOJO_UNAVAILABLE_REASON,
            "runtime_backend": "dreamdojo",
            "trainer_backend": "dreamdojo",
        },
        "llamafactory": {
            "name": "llamafactory",
            "available": True,
            "trainer_backend": "llamafactory",
        },
        "verl": {
            "name": "verl",
            "available": True,
            "trainer_backend": "verl",
        },
        "vjepa2_native": {
            "name": "vjepa2_native",
            "available": True,
            "runtime_backend": "vjepa2_native",
            "trainer_backend": "vjepa2_native",
        },
        "wan_native": {
            "name": "wan_native",
            "available": True,
            "runtime_backend": "wan_native",
            "trainer_backend": "wan_native",
        },
    }


def capabilities() -> dict[str, dict[str, Any]]:
    return report_capabilities()


def describe_capabilities() -> dict[str, dict[str, Any]]:
    return report_capabilities()


def get_capabilities() -> dict[str, dict[str, Any]]:
    return report_capabilities()


def list_capabilities() -> dict[str, dict[str, Any]]:
    return report_capabilities()


def is_available(backend: str) -> bool:
    record = report_capabilities().get(backend)
    return bool(record and record.get("available"))


def require_available(backend: str) -> None:
    record = report_capabilities().get(backend)
    if record is None:
        raise CapabilityUnavailableError(f"Unknown backend capability: {backend}")
    if not record.get("available"):
        reason = record.get("reason") or f"{backend} is unavailable."
        raise CapabilityUnavailableError(str(reason))


__all__ = [
    "CapabilityUnavailableError",
    "capabilities",
    "describe_capabilities",
    "get_capabilities",
    "is_available",
    "list_capabilities",
    "report_capabilities",
    "require_available",
]
