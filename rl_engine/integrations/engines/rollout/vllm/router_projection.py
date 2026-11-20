# SPDX-License-Identifier: Apache-2.0
"""Bind the real unquantized gate module without changing its Parameters."""

import types

import torch

from rl_engine.backends.rocm.moe.router_projection import router_projection


def bind_rollout_router_projection(gate):
    if not gate.is_unquantized or gate.bias is not None or gate.out_dtype != torch.float32:
        raise ValueError("original bias-free FP32-output gate only")

    def forward(self, x):
        result = router_projection(x, self.weight)
        if not getattr(self, "_rlk_projection_executed", False):
            print(
                "RLK_ROLLOUT_ROUTER_PROJECTION_EXECUTED",
                __file__,
                "fixed no-split FP32 GEMM",
                flush=True,
            )
            self._rlk_projection_executed = True
        return (result, None) if self.return_bias else result

    gate.forward = types.MethodType(forward, gate)
    return gate
