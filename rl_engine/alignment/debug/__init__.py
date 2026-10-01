# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Portable evidence API; runtime-specific adapters are loaded only for live replay."""

from .core import fingerprint
from .evidence import Capture
from .auto import AutoCapture, auto_capture, prepare_auto
