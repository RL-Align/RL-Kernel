# SPDX-License-Identifier: Apache-2.0
"""Opt-in instance-only binding; all other AITER kernels retain native dispatch."""

from types import MethodType

from rl_engine.backends.rocm.gemm.shared_down_fp8 import PROFILE, shared_down_fp8


def bind_shared_down(model, worker):
    from vllm.model_executor.kernels.linear.scaled_mm.aiter import AiterFp8BlockScaledMMKernel

    kernels = []
    for index in range(model.start_layer, model.end_layer):
        down = model.layers[index].ffn.shared_experts.down_proj
        kernel = down.quant_method.fp8_linear
        if (
            not isinstance(kernel, AiterFp8BlockScaledMMKernel)
            or kernel.use_triton
            or kernel.use_bpreshuffle
            or getattr(kernel, "_rlk_shared_down", False)
        ):
            raise ValueError("shared down requires an unbound gfx942 AITER blockscale kernel")
        kernels.append(kernel)
    for kernel in kernels:

        def apply(instance, A, B, As, Bs):
            return worker._record("shared_down", shared_down_fp8(A, B, As, Bs))

        kernel.apply_block_scaled_mm = MethodType(apply, kernel)
        kernel._rlk_shared_down = PROFILE
    return len(kernels)
