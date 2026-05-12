"""Acceptance tests for the dataset adapter registry keys (M4).

The repo-owned dataset adapter registry must expose stable keys that
downstream milestones can rely on, even before the corresponding adapter
implementations land. M4 pins these five keys:

  * ``chat_sft`` (the LLaMA-Factory ShareGPT path)
  * ``chat_rl`` (the VERL JSONL path)
  * ``vjepa2`` (the V-JEPA2 video-encoder path, stubbed until M6)
  * ``wan`` (the Wan2.2 video-generator path, stubbed until M7)
  * ``dreamdojo`` (the DreamDojo world-model path, stubbed until M8)
"""

from __future__ import annotations

import pytest


REQUIRED_KEYS = ("chat_sft", "chat_rl", "vjepa2", "wan", "dreamdojo")


def _import_registry_module():
    import verl_post_training.adapters.dataset as module

    return module


def test_dataset_adapter_registry_module_is_importable():
    module = _import_registry_module()
    assert module is not None


def test_dataset_adapter_registry_exposes_required_keys():
    module = _import_registry_module()

    # Accept either a tuple/frozenset constant or a `list_dataset_adapters()`
    # helper, since both are reasonable shapes.
    keys_constant = getattr(module, "DATASET_ADAPTER_KEYS", None)
    keys_list_fn = getattr(module, "list_dataset_adapters", None)

    discovered: set[str]
    if keys_constant is not None:
        discovered = set(keys_constant)
    elif callable(keys_list_fn):
        discovered = set(keys_list_fn())
    else:
        pytest.fail(
            "verl_post_training.adapters.dataset must expose either "
            "`DATASET_ADAPTER_KEYS` or `list_dataset_adapters()` so callers can "
            "enumerate stable adapter keys."
        )

    missing = sorted(set(REQUIRED_KEYS) - discovered)
    assert not missing, (
        f"Adapter registry is missing required keys: {missing}. "
        f"Found: {sorted(discovered)}"
    )


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_each_required_key_resolves_to_an_adapter(key):
    from verl_post_training.adapters.dataset import get_dataset_adapter

    adapter = get_dataset_adapter(key)
    assert adapter is not None, (
        f"get_dataset_adapter({key!r}) must not return None; "
        "stub adapters are allowed for non-chat keys but must still resolve."
    )


def test_unknown_adapter_key_raises_a_typed_error():
    from verl_post_training.adapters.dataset import get_dataset_adapter

    with pytest.raises(LookupError):
        get_dataset_adapter("definitely-not-a-real-adapter-key")
