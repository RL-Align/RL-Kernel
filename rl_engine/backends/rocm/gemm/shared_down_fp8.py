# SPDX-License-Identifier: Apache-2.0
"""Fixed no-split-K AITER profile for DSV4 shared-expert down projections."""

import torch

PROFILE = "dsv4-shared-down-aiter-ck-gfx942-nosplit-v1"
KERNEL = (
    "a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_"
    "16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1"
)


def shared_down_fp8(a, b, sa, sb):
    if not torch.version.hip or a.device.type != "cuda":
        raise RuntimeError("shared down profile requires ROCm GPU tensors")
    arch = torch.cuda.get_device_properties(a.device).gcnArchName.split(":")[0]
    if arch != "gfx942":
        raise ValueError(f"{PROFILE} requires gfx942, got {arch}")
    if a.ndim != 2 or a.shape[1] != 2048 or tuple(b.shape) != (4096, 2048):
        raise ValueError("shared down profile requires [M,2048] and [4096,2048]")
    if a.dtype != torch.float8_e4m3fnuz or b.dtype != torch.float8_e4m3fnuz:
        raise TypeError("shared down profile expects post-load FNUZ operands")
    if sa.dtype != torch.float32 or sb.dtype != torch.float32:
        raise TypeError("shared down profile requires native FP32 scales")
    if sa.shape != (a.shape[0], 16) or sb.shape != (32, 16):
        raise ValueError("shared down profile requires native block128 scales")
    if any(t.device != a.device for t in (b, sa, sb)):
        raise ValueError("shared down operands must share a GPU")
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_ck

    out = torch.empty((a.shape[0], 4096), device=a.device, dtype=torch.bfloat16)
    if a.shape[0]:
        with torch.cuda.device(a.device):
            gemm_a8w8_blockscale_ck(a, b, sa, sb, out, splitK=0, kernelName=KERNEL)
    return out
