# SPDX-License-Identifier: Apache-2.0
"""Explicit FP32 reference for the Qwen-Image interleaved rotation."""

import torch

from rl_engine.ops.rope.multi_axis_rope import MultiAxisRoPEMetadata, validate_inputs


class NativeMultiAxisRoPEOp(MultiAxisRoPEMetadata):
    op_class = "elementwise"
    backend_id = "rlkernel.reference.qwenimage_multi_axis_rope.v1"
    kernel_id = "rl_engine.reference.rope.multi_axis_rope.NativeMultiAxisRoPEOp"

    def __call__(self, x, cos, sin):
        return self.forward(x, cos, sin)

    def forward(self, x, cos, sin):
        return self.forward_fp32(x, cos, sin).to(x.dtype)

    def forward_fp32(self, x, cos, sin):
        validate_inputs(x, cos, sin)
        c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)
        even, odd = x[..., 0::2].float(), x[..., 1::2].float()
        out = torch.stack((even * c - odd * s, odd * c + even * s), dim=-1)
        return out.flatten(-2)
