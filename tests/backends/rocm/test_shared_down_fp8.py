# SPDX-License-Identifier: Apache-2.0
"""Replay actual native vLLM operands without changing FP8 weights/scales."""

import os

import pytest
import torch

from rl_engine.backends.rocm.gemm.shared_down_fp8 import shared_down_fp8

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_original_vllm_operands_repeat_and_batch_position():
    path = os.getenv("RLK_DSV4_SHARED_DOWN_OPERANDS")
    if not path:
        pytest.skip("set RLK_DSV4_SHARED_DOWN_OPERANDS to a native capture")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    a, b, sa, sb = [saved[k].cuda() for k in ("A", "B", "As", "Bs")]
    expected = shared_down_fp8(a, b, sa, sb)
    aa = (a.cpu().float().reshape(1, 16, 128) * sa.cpu()[..., None]).reshape(1, 2048)
    bb = b.cpu().float() * sb.cpu().repeat_interleave(128, 0).repeat_interleave(128, 1)
    torch.testing.assert_close(expected.cpu().float(), aa @ bb.t(), atol=0.004, rtol=0.01)
    for _ in range(20):
        actual = shared_down_fp8(a, b, sa, sb)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    for m in (2, 17):
        aa = a.expand(m, -1).contiguous()
        ss = sa.expand(m, -1).contiguous()
        actual = shared_down_fp8(aa, b, ss, sb)
        for position in (0, m - 1):
            assert torch.equal(actual[position].view(torch.uint8), expected[0].view(torch.uint8))
