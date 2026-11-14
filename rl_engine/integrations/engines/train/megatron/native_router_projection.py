# SPDX-License-Identifier: Apache-2.0
"""Bind only the native router's GEMM; preserve Parameters and routing."""

import types

from rl_engine.backends.rocm.moe.router_projection import router_projection


def bind_native_router_projection(router):
    if router.bias is not None or router.config.moe_router_dtype != "fp32":
        raise ValueError("ordinary bias-free FP32 native router only")

    def gating(self, x):
        result = router_projection(x, self.weight)
        if not getattr(self, "_rlk_projection_executed", False):
            print(
                "RLK_NATIVE_ROUTER_PROJECTION_EXECUTED",
                __file__,
                "fixed no-split FP32 GEMM",
                flush=True,
            )
            self._rlk_projection_executed = True
        return result

    router.gating = types.MethodType(gating, router)
    return router
