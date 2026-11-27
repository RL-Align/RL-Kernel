# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Qwen-Image joint attention QK GEMM at the issue #386 reference resolutions."""

from __future__ import annotations

import math

import pytest
import torch

from rl_engine.reference.attention.joint_attn_qk_gemm import NativeJointAttnQkGemmOp
from rl_engine.validation.common.bitwise import tensor_bytes_equal as bitwise_equal

HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)
# Image-token counts at the three reference resolutions of issue #386
# (1024², 1328², 1664x928) after 2x2 packing of the 16-channel latents.
REFERENCE_IMAGE_TOKENS = (4096, 6889, 6032)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _cuda_op():
    """The CUDA backend, or a skip when the native extension is not built."""
    from rl_engine.backends.cuda.attention.joint_attn_qk_gemm import JointAttnQkGemmCudaOp

    try:
        return JointAttnQkGemmCudaOp()
    except RuntimeError as exc:  # pragma: no cover - depends on the build
        pytest.skip(f"joint_attn_qk_gemm CUDA kernel unavailable: {exc}")


@requires_cuda
class TestReferenceShapes:
    """The three reference resolutions of issue #386, plus prompt text tokens.

    The full fp32 gold is not computed at these sizes; each score element reduces
    over the head dim independently, so comparing a query slice against the gold
    is a complete check of the arithmetic for those rows.
    """

    @pytest.mark.parametrize("image_tokens", REFERENCE_IMAGE_TOKENS)
    def test_query_slice_matches_the_gold(self, image_tokens):
        cuda_op = _cuda_op()
        gold = NativeJointAttnQkGemmOp()
        torch.manual_seed(23)
        text_tokens = 64
        joint = text_tokens + image_tokens
        q = torch.randn(1, 1, joint, HEAD_DIM, dtype=torch.bfloat16)
        k = torch.randn(1, 1, joint, HEAD_DIM, dtype=torch.bfloat16)

        device_rows = cuda_op.forward(q.cuda(), k.cuda(), scale=SCALE)[:, :, :8, :].cpu()
        gold_rows = gold.forward(q[:, :, :8, :], k, scale=SCALE)
        assert bitwise_equal(device_rows.contiguous(), gold_rows)

    @pytest.mark.parametrize("image_tokens", REFERENCE_IMAGE_TOKENS)
    def test_text_prefix_rows_are_padding_invariant(self, image_tokens):
        cuda_op = _cuda_op()
        torch.manual_seed(24)
        text_tokens = 64
        q = torch.randn(1, 1, 8, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 1, text_tokens, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        clean = cuda_op.forward(q, k, scale=SCALE)
        image_k = torch.randn(1, 1, image_tokens, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        joined = cuda_op.forward(q, torch.cat([k, image_k], dim=-2), scale=SCALE)
        assert bitwise_equal(joined[..., :text_tokens].contiguous(), clean)
