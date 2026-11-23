# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Arithmetic defaults shared by ordered-reduction implementations and references.

No backend or tensor library is imported here. Scheduling must not change these
values; changes require numerical checks and GPU qualification. Independent
accuracy tests must not rely solely on a reference sharing this configuration.
Operators with their own arithmetic contracts use the explicit low-level helpers.
"""

TILE_SIZE = 1024
"""Whole-vector arithmetic tile width, independent of the launch schedule."""

MAX_GROUP_SIZE = 8
"""Adjacent values folded sequentially before the group totals are halved to one."""
