# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Input layouts shared by activation references and backend implementations."""

from enum import Enum


class SwiGLUPackedOrder(Enum):
    """Order of two equal halves along a packed SwiGLU input's last dimension.

    The activation is always ``SiLU(gate) * up``. This enum changes only where
    the operands are stored; it does not change the arithmetic or output shape.
    """

    GATE_UP = "gate_up"
    """First half is gate, second half is up."""

    UP_GATE = "up_gate"
    """First half is up, second half is gate, as in MiniMax-H3."""
