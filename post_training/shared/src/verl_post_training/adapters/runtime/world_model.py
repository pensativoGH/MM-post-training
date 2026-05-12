"""Compatibility alias for the DreamDojo world-model runtime adapter."""

from __future__ import annotations

import sys

from verl_post_training_dreamdojo import runtime as _runtime
from verl_post_training_dreamdojo.runtime import *  # noqa: F401,F403

sys.modules[__name__] = _runtime
