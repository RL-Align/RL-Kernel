# SPDX-License-Identifier: Apache-2.0
"""Route-weight gradient before FC2 activation quantization."""

from torch import Tensor

from rl_engine.backends.rocm.moe.route_weight_grad import RocmRouteWeightGradient


def route_weight_grad(dh: Tensor, silu: Tensor, up: Tensor) -> Tensor:
    """Return FP32 dp_s[M] for already-dispatched rows, preserving their order.

    dh is the upstream gradient after the FC2 base/LoRA merge and its cast.
    silu is the saved FP32 SiLU value from task 46, and up is its original up
    input. Sum (FP32(dh) * silu) * clamp(FP32(up), -10, 10) with the fixed
    pair256 profile. No multiplication or division by p_s occurs here.

    This is a first-order VJP, not a differentiable forward operator. P4 owns
    return transport; P3 owns selection/normalization/router backward.
    """
    return RocmRouteWeightGradient.reduce(dh, silu, up)
