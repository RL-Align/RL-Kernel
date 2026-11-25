# SPDX-License-Identifier: Apache-2.0
"""Pin actual local Q-up FP8 GEMMs to the actor's no-split CK arithmetic."""

from types import MethodType

import torch

from rl_engine.backends.rocm.fp8 import fp8_dtype
from rl_engine.backends.rocm.gemm.shared_down_fp8 import KERNEL


def fixed_q_up_fp8(a, b, sa, sb):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_ck

    dtype = fp8_dtype(a.device)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[1]
        or a.shape[1] % 128
        or b.shape[0] % 128
        or a.dtype != dtype
        or b.dtype != dtype
        or sa.dtype != torch.float32
        or sb.dtype != torch.float32
        or sa.shape != (a.shape[0], a.shape[1] // 128)
        or sb.shape != (b.shape[0] // 128, b.shape[1] // 128)
        or any(t.device != a.device for t in (b, sa, sb))
    ):
        raise ValueError("Q-up requires original local FP8/block128 tensors and FP32 scales")
    out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.bfloat16)
    if a.shape[0]:
        gemm_a8w8_blockscale_ck(
            a.contiguous(),
            b.contiguous(),
            sa.contiguous(),
            sb.contiguous(),
            out,
            splitK=0,
            kernelName=KERNEL,
        )
    return out


def bind_q_up_projection(attention):
    from vllm.model_executor.kernels.linear.scaled_mm.aiter import AiterFp8BlockScaledMMKernel

    modules = [attention.wq_b]
    if attention.indexer is not None:
        modules.append(attention.indexer.wq_b)
    for module in modules:
        kernel = module.quant_method.fp8_linear
        if (
            not isinstance(kernel, AiterFp8BlockScaledMMKernel)
            or kernel.use_triton
            or kernel.use_bpreshuffle
            or kernel.config.out_dtype != torch.bfloat16
            or getattr(kernel, "_rlk_q_up_fixed", False)
        ):
            raise ValueError("expected the unbound gfx942 native local unshuffled Q-up kernel")

        def apply(instance, A, B, As, Bs):
            result = fixed_q_up_fp8(A, B, As, Bs)
            if not getattr(instance, "_rlk_q_up_executed", False):
                print(
                    "RLK_FIXED_Q_UP_EXECUTED",
                    __file__,
                    tuple(A.shape),
                    tuple(B.shape),
                    KERNEL,
                    flush=True,
                )
                instance._rlk_q_up_executed = True
            return result

        kernel.apply_block_scaled_mm = MethodType(apply, kernel)
        kernel._rlk_q_up_fixed = True
    return len(modules)
