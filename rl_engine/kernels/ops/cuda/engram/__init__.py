# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4.1 Engram gate (profile ``dsv41-engram-v1``)."""

from .dsv41_engram import PROFILE, engram_gate, engram_gate_backward_cuda, engram_gate_cuda

__all__ = ["PROFILE", "engram_gate", "engram_gate_backward_cuda", "engram_gate_cuda"]
