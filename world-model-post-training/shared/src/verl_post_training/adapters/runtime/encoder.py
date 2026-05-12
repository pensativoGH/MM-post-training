"""Compatibility alias for the V-JEPA2 encoder runtime adapter."""

from __future__ import annotations

import sys

from verl_post_training_vjepa import runtime as _runtime
from verl_post_training_vjepa.runtime import *  # noqa: F401,F403

sys.modules[__name__] = _runtime
