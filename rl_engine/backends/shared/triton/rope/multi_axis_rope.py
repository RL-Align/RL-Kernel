# SPDX-License-Identifier: Apache-2.0
"""FlashAttention-derived, stride-aware rotation under a frozen FP32 profile."""

import torch
import triton

from rl_engine.ops.rope._registration import register_rotation
from rl_engine.ops.rope.multi_axis_rope import MultiAxisRoPEMetadata, validate_inputs

from ._flash_attn_rotary import rotary_kernel


def _apply(x, cos, sin, transpose):
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    batch, seq, heads, _ = x.shape
    if batch and seq:
        with torch.cuda.device(x.device):
            rotary_kernel[(triton.cdiv(heads, 2), triton.cdiv(seq, 8), batch)](
                output,
                x,
                cos,
                sin,
                None,
                0,
                seq,
                heads,
                seq,
                *output.stride(),
                *x.stride(),
                cos.stride(0) if cos.ndim == 3 else 0,
                cos.stride(-2),
                cos.stride(-1),
                sin.stride(0) if sin.ndim == 3 else 0,
                sin.stride(-2),
                sin.stride(-1),
                128,
                False,
                False,
                True,
                transpose,
                BLOCK_H=2,
                BLOCK_M=8,
                enable_fp_fusion=False,
            )
    return output


_rotation = register_rotation(
    "qwen_image_rope_triton",
    _apply,
    kernel_id="rl_engine.backends.shared.triton.rope._flash_attn_rotary.rotary_kernel",
    family="triton",
)


class MultiAxisRoPETritonOp(MultiAxisRoPEMetadata):
    backend_id = "rlkernel.triton.qwenimage_multi_axis_rope.v1"
    kernel_id = "rl_engine.backends.shared.triton.rope._flash_attn_rotary.rotary_kernel"

    def __init__(self):
        if torch.version.hip is not None or not torch.cuda.is_available():
            raise RuntimeError("this multi_axis_rope Triton profile requires NVIDIA CUDA")

    def __call__(self, x, cos, sin):
        validate_inputs(x, cos, sin)
        if x.device.type != "cuda":
            raise ValueError("Triton multi_axis_rope requires CUDA tensors")
        return _rotation(x, cos, sin, False)
