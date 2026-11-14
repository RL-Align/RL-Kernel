# SPDX-License-Identifier: Apache-2.0
"""gfx942 midpoint trees must support non-power-of-two token reductions."""

import pytest
import torch

from rl_engine.backends.shared.triton.embedding.lm_head import TritonLMHeadOp
from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_tree_gemm

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def tree(a, b):
    def dot(lo, hi):
        if hi - lo <= 32:
            return (a[:, lo:hi].float() @ b[lo:hi].float()).bfloat16()
        mid = lo + (hi - lo) // 2
        return dot(lo, mid) + dot(mid, hi)

    return dot(0, a.shape[1])


def bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("k", [65, 131, 1537])
def test_ragged_forward_inference_and_lm_head_wgrad(k):
    torch.manual_seed(131)
    a = torch.randn(5, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, 17, device="cuda", dtype=torch.bfloat16)
    expected = tree(a, b)
    bits(_triton_tree_gemm(a, b), expected)
    with torch.inference_mode():
        bits(_triton_tree_gemm(a, b), expected)
    hidden = torch.randn(k, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(k, 32, device="cuda", dtype=torch.bfloat16)
    TritonLMHeadOp()(hidden, weight).backward(grad)
    bits(weight.grad, tree(grad.t(), hidden.detach()))
