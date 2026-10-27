# SPDX-License-Identifier: Apache-2.0
"""Explicit ROCm route-gradient candidate; no automatic backend replacement."""

import torch
from torch import Tensor

from rl_engine.contracts.operators.route_weight_grad import (
    BLOCK,
    PROFILE,
    validate_inputs,
    validate_training_inputs,
)


def _require_device(device) -> None:
    if not torch.version.hip or device.type != "cuda":
        raise RuntimeError(f"{PROFILE} requires ROCm GPU tensors; no CPU fallback")
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch not in ("gfx942", "gfx950"):
        raise RuntimeError(f"{PROFILE} has no candidate for {arch}")


class RocmRouteWeightGradient:
    """Task 48 RouteGradientBackend adapter, also usable as a standalone VJP.

    The input state must come from the corresponding task 46 forward. This
    class does not recompute SiLU or infer it from the weighted BF16 output.
    """

    profile = PROFILE

    def validate(self, config, device) -> None:
        _require_device(device)
        if type(config.ffn) is not int or config.ffn <= 0:
            raise ValueError("ffn must be a positive integer")

    @staticmethod
    @torch.no_grad()
    def reduce(dh: Tensor, silu: Tensor, up: Tensor) -> Tensor:
        validate_inputs(dh, silu, up)
        _require_device(dh.device)
        from .route_weight_grad_kernels import _finish, _partials

        m, f = dh.shape
        dp = torch.empty((m,), dtype=torch.float32, device=dh.device)
        if not m:
            return dp
        chunks = (f + BLOCK - 1) // BLOCK
        # Per-call workspace: concurrent streams and graph instances cannot
        # overwrite another call's partials. All entries are written before use.
        partials = torch.empty((m, chunks), dtype=torch.float32, device=dh.device)
        with torch.cuda.device(dh.device):
            _partials[(m, chunks)](
                dh,
                silu,
                up,
                partials,
                f,
                chunks,
                *dh.stride(),
                *silu.stride(),
                *up.stride(),
                num_warps=4,
                enable_fp_fusion=False,
            )
            _finish[(m,)](
                partials,
                dp,
                chunks,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return dp

    def backward(self, dh, gate, up, p_s, state) -> Tensor:
        validate_training_inputs(dh, gate, up, p_s, state)
        return self.reduce(dh, state.silu, up)
