# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Shared input and arithmetic configuration for softcapped selected logprob.

This module describes and validates configuration: no validation of input
tensors, reference computation, Triton imports or dispatch policy. The native
PyTorch reference shares the input contract but keeps its independent log_softmax reduction.
The fixed-order PyTorch reference and Triton use the same reduction config.
"""

from dataclasses import dataclass, field

import torch

from rl_engine.contracts.operators.ordered_reduction import MAX_GROUP_SIZE, TILE_SIZE

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
"""Logits and upstream storage dtypes; arithmetic and forward output are FP32."""

TRITON_SUPPORTED_DEVICES = ("cuda", "hip", "xpu", "musa")
"""Device types accepted by the Triton implementation; ROCm also uses 'cuda'.

This does not restrict the PyTorch references, which can also run on CPU.
Acceptance does not imply that each backend has passed numerical or performance
qualification; those checks must run on the corresponding hardware.
"""


@dataclass(frozen=True)
class SoftcappedLogprobReductionConfig:
    """Arithmetic shared by all schedules and the fixed-order reference.

    Only the tile width is configurable. Adjacent groups use the shared default,
    followed by one lower-half/upper-half tree over all group totals. These
    define arithmetic, rather than per-strategy hardware tuning knobs.
    Changing the width requires new numerical and performance qualification;
    bitwise equality is required across schedules using the same config, not
    across different widths. Configure this before importing either implementation.
    """

    tile_size: int = TILE_SIZE
    """Positive power-of-two vocabulary width; masked tail lanes contribute zero."""

    group_size: int = field(default=MAX_GROUP_SIZE, init=False)
    """Fold up to eight adjacent values left to right, starting from the first value."""

    def __post_init__(self) -> None:
        """Reject invalid tile widths as soon as the configuration is constructed.

        Rely on the int annotation; check the value needed by both reduction
        implementations. group_size is a fixed, non-init field.
        """
        if self.tile_size <= 0 or self.tile_size & (self.tile_size - 1):
            raise ValueError("tile_size must be a positive power of two")


REDUCTION_CONFIG = SoftcappedLogprobReductionConfig()
"""Single source for production tile width and the fixed-order addition structure.

Tile totals are accumulated in ascending vocabulary order, starting from FP32
positive zero. Construction checks the configured width. Kernels and the
PyTorch fixed-order helper also check the actual supplied vector width.
"""
