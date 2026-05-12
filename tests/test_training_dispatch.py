"""M8 acceptance: existing user-facing chat training entrypoints resolve
through the repo-owned trainer-adapter control plane *without changing the
required user inputs*.

This file pins the third and fourth M8 acceptance criteria (quoted from
the approved plan):

    ``SFT/scripts/run_local_sft.sh`` resolves supported Qwen chat SFT
    workflows through
    ``post_training/src/verl_post_training/adapters/trainer/llamafactory.py``
    without changing required user inputs

    ``RL/scripts/run_local_grpo.py`` resolves supported Qwen chat RL
    workflows through
    ``post_training/src/verl_post_training/adapters/trainer/verl.py``
    without changing required user inputs

Concretely, these tests pin:

* the LLaMA-Factory trainer adapter module exists at
  ``verl_post_training.adapters.trainer.llamafactory`` and exposes a
  trainer adapter
* the VERL trainer adapter module exists at
  ``verl_post_training.adapters.trainer.verl`` and exposes a trainer
  adapter
* both adapters are registered in the trainer-adapter registry so dispatch
  can find them
* ``SFT/scripts/run_local_sft.sh`` still accepts the existing user input
  (one positional path to a YAML config) and routes to the LLaMA-Factory
  trainer adapter rather than calling ``llamafactory-cli`` directly with
  no control-plane involvement
* ``RL/scripts/run_local_grpo.py`` still accepts the existing user input
  (one positional path to a YAML config + an optional ``--dry-run`` flag)
  and routes to the VERL trainer adapter rather than invoking
  ``verl.trainer.main_ppo`` directly with no control-plane involvement
* the SFT script's --help output / argument signature is backward
  compatible with the existing user contract

The tests intentionally do *not* run the trainers — they assert that the
scripts reach the trainer adapter (e.g. through dry-run modes or static
references in the script source) without changing required arguments.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SFT_SCRIPT = REPO_ROOT / "SFT" / "scripts" / "run_local_sft.sh"
RL_SCRIPT = REPO_ROOT / "RL" / "scripts" / "run_local_grpo.py"


# ---------------------------------------------------------------------------
# Adapter module presence
# ---------------------------------------------------------------------------


def test_llamafactory_trainer_adapter_module_exists():
    """The LLaMA-Factory trainer adapter module is named explicitly in the
    plan; the writer must add it under
    ``verl_post_training.adapters.trainer.llamafactory``.
    """

    try:
        import verl_post_training.adapters.trainer.llamafactory as module
    except ModuleNotFoundError as exc:
        pytest.fail(
            "M8 requires "
            "`verl_post_training.adapters.trainer.llamafactory` to exist; "
            f"got ModuleNotFoundError: {exc!r}"
        )
    assert any(
        isinstance(getattr(module, attr, None), type)
        or callable(getattr(module, attr, None))
        for attr in dir(module)
        if not attr.startswith("_")
    ), (
        "verl_post_training.adapters.trainer.llamafactory must expose a "
        "trainer adapter class or instance."
    )


def test_verl_trainer_adapter_module_exists():
    """The VERL trainer adapter module is named explicitly in the plan; the
    writer must add it under ``verl_post_training.adapters.trainer.verl``.
    """

    try:
        import verl_post_training.adapters.trainer.verl as module
    except ModuleNotFoundError as exc:
        pytest.fail(
            "M8 requires "
            "`verl_post_training.adapters.trainer.verl` to exist; "
            f"got ModuleNotFoundError: {exc!r}"
        )
    assert any(
        isinstance(getattr(module, attr, None), type)
        or callable(getattr(module, attr, None))
        for attr in dir(module)
        if not attr.startswith("_")
    ), (
        "verl_post_training.adapters.trainer.verl must expose a trainer "
        "adapter class or instance."
    )


def test_chat_trainer_adapters_registered_in_trainer_registry():
    """Both chat trainer adapters must be reachable through the trainer
    adapter registry so dispatch can resolve them by ``trainer_backend``.
    """

    from verl_post_training.adapters.trainer import (
        ADAPTER_REGISTRY,
        get_trainer_adapter,
    )

    llf_adapter = get_trainer_adapter("llamafactory")
    assert llf_adapter is not None, (
        "trainer adapter registry must expose 'llamafactory' once M8 lands."
    )
    verl_adapter = get_trainer_adapter("verl")
    assert verl_adapter is not None, (
        "trainer adapter registry must expose 'verl' once M8 lands."
    )

    keys = set(ADAPTER_REGISTRY)
    assert "llamafactory" in keys, (
        f"ADAPTER_REGISTRY missing 'llamafactory' key; got {sorted(keys)}"
    )
    assert "verl" in keys, (
        f"ADAPTER_REGISTRY missing 'verl' key; got {sorted(keys)}"
    )


# ---------------------------------------------------------------------------
# run_local_sft.sh: routes through the LLaMA-Factory trainer adapter and
# preserves the existing user-facing argument contract
# ---------------------------------------------------------------------------


def test_run_local_sft_script_exists_and_is_executable():
    assert SFT_SCRIPT.exists(), (
        f"M8 expects {SFT_SCRIPT} to remain present (the bash entrypoint "
        "must not be deleted)."
    )
    assert os.access(SFT_SCRIPT, os.X_OK), (
        f"{SFT_SCRIPT} must remain executable for current users."
    )


def test_run_local_sft_script_accepts_optional_positional_config():
    """The current user contract is: pass a YAML config path (optionally)
    as positional argument 1. The script must keep that contract.
    """

    text = SFT_SCRIPT.read_text(encoding="utf-8")
    # The current script uses ``${1:-...}`` to accept an optional positional
    # config path. The user contract must not change to a required flag or
    # to a different positional slot.
    assert re.search(r"\$\{1:-", text), (
        "run_local_sft.sh must continue to accept an optional positional "
        "config path via ${1:-...} so existing invocations keep working."
    )
    assert "CONFIG_PATH" in text, (
        "run_local_sft.sh must preserve the CONFIG_PATH variable that "
        "downstream users (and this test) rely on as the public input."
    )


def test_run_local_sft_routes_through_llamafactory_trainer_adapter():
    """The shell script must reach the LLaMA-Factory trainer adapter — not
    invoke llamafactory-cli directly with no control-plane involvement.

    The contract is satisfied if the script text references the trainer
    adapter (module or wrapper module that goes through it).
    """

    text = SFT_SCRIPT.read_text(encoding="utf-8")
    has_adapter_reference = any(
        token in text
        for token in (
            "verl_post_training.adapters.trainer.llamafactory",
            "verl_post_training/adapters/trainer/llamafactory",
            "verl_post_training.launch.dispatch",
            "verl_post_training.launch",
            "trainer.llamafactory",
        )
    )
    assert has_adapter_reference, (
        "run_local_sft.sh must route through the LLaMA-Factory trainer "
        "adapter (or the repo-owned launch dispatch path that resolves to "
        "it). The current script invokes llamafactory-cli directly without "
        "passing through the control plane; M8 must add an adapter call so "
        "the adapter is the single source of truth for SFT launch. "
        f"Script text:\n{text[:1200]}"
    )


# ---------------------------------------------------------------------------
# run_local_grpo.py: routes through the VERL trainer adapter and preserves
# the existing user-facing CLI surface
# ---------------------------------------------------------------------------


def test_run_local_grpo_script_exists():
    assert RL_SCRIPT.exists(), (
        f"M8 expects {RL_SCRIPT} to remain present (the Python entrypoint "
        "must not be deleted)."
    )


def test_run_local_grpo_help_preserves_user_contract(tmp_path):
    """``run_local_grpo.py --help`` must continue to advertise the current
    positional ``config`` argument and the ``--dry-run`` flag. If either
    is removed or renamed, the user-facing input contract has changed.
    """

    env = os.environ.copy()
    pkg_src = REPO_ROOT / "post_training" / "src"
    env["PYTHONPATH"] = (
        f"{pkg_src}{os.pathsep}{env['PYTHONPATH']}"
        if "PYTHONPATH" in env
        else str(pkg_src)
    )
    result = subprocess.run(
        [sys.executable, str(RL_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=60,
    )
    output = (result.stdout or "") + (result.stderr or "")
    # --help may exit 0 (argparse) or non-zero (gated import error); we
    # only require that the script advertises the historical inputs once
    # M8 lands.
    assert "config" in output.lower(), (
        "run_local_grpo.py --help must continue to advertise the "
        f"positional `config` argument. Output:\n{output[:1500]}"
    )
    assert "--dry-run" in output, (
        "run_local_grpo.py --help must continue to advertise the "
        f"`--dry-run` flag. Output:\n{output[:1500]}"
    )


def test_run_local_grpo_routes_through_verl_trainer_adapter():
    """The Python script must reach the VERL trainer adapter — not invoke
    ``verl.trainer.main_ppo`` directly with no control-plane involvement.

    Acceptance is satisfied if the script text imports or references the
    VERL trainer adapter (or the repo-owned launch dispatch path).
    """

    text = RL_SCRIPT.read_text(encoding="utf-8")
    has_adapter_reference = any(
        token in text
        for token in (
            "verl_post_training.adapters.trainer.verl",
            "verl_post_training.launch.dispatch",
            "verl_post_training.launch",
            "from verl_post_training",
            "import verl_post_training",
            "resolve_trainer_adapter",
            "select_trainer_adapter",
            "trainer_adapter_for_plan",
            "get_trainer_adapter",
        )
    )
    assert has_adapter_reference, (
        "run_local_grpo.py must route through the VERL trainer adapter "
        "(or the repo-owned launch dispatch path that resolves to it). "
        "The current script calls verl.trainer.main_ppo directly without "
        "passing through the control plane; M8 must add an adapter call so "
        "the adapter is the single source of truth for RL launch. "
        f"Script text:\n{text[:1200]}"
    )
