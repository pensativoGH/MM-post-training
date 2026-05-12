"""Compatibility alias for the Wan video-generation runtime adapter."""

from __future__ import annotations

import sys

from verl_post_training_wan import runtime as _runtime
from verl_post_training_wan.runtime import *  # noqa: F401,F403

sys.modules[__name__] = _runtime
