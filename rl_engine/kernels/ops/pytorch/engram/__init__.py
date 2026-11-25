# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Golden DSv4.1 Engram gate (profile ``dsv41-engram-v1``)."""

from .dsv41_engram import CLAMP, NORM_EPS, engram_gate, engram_gate_backward_reference

__all__ = ["CLAMP", "NORM_EPS", "engram_gate", "engram_gate_backward_reference"]
