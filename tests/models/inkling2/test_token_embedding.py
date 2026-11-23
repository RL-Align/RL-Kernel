# SPDX-License-Identifier: Apache-2.0
"""INK-06 token embedding contract tests."""

import pytest
import torch

from rl_engine.models.inkling2.token_embedding import token_embedding


def test_embedding_accepts_int32_ids_and_padded_rows():
    weight = torch.arange(7 * 3, dtype=torch.float32).reshape(7, 3)
    ids = torch.tensor([[0, 5, 6]], dtype=torch.int32)
    output = token_embedding(ids, weight)
    assert output.dtype == weight.dtype
    assert torch.equal(output, weight.index_select(0, ids.long().reshape(-1)).reshape(1, 3, 3))


def test_embedding_output_cast_is_explicit():
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    ids = torch.tensor([1, 7], dtype=torch.int64)
    output = token_embedding(ids, weight, output_dtype=torch.float32)
    assert output.dtype == torch.float32
    assert torch.equal(output, weight.index_select(0, ids).float())


def test_embedding_backward_accumulates_duplicate_ids():
    weight = torch.randn(6, 3, requires_grad=True)
    ids = torch.tensor([2, 2, 4])
    token_embedding(ids, weight).sum().backward()
    assert torch.equal(weight.grad[2], torch.full((3,), 2.0))
    assert torch.equal(weight.grad[4], torch.ones(3))


def test_embedding_rejects_out_of_range_ids():
    with pytest.raises(ValueError, match="must be in"):
        token_embedding(torch.tensor([8]), torch.zeros(8, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_sm90_embedding_matches_reference_and_backward():
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE
    from rl_engine.models.inkling2.token_embedding import token_embedding_cuda

    if not _EXT_AVAILABLE or not hasattr(_C, "embedding_sm90_forward"):
        pytest.skip("SM90 embedding CUDA extension is unavailable")
    torch.manual_seed(17)
    weight = torch.randn(8, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ids = torch.tensor([[0, 5, 5, 7]], device="cuda", dtype=torch.int32)
    native = token_embedding_cuda(ids, weight)
    reference = token_embedding(ids, weight)
    torch.testing.assert_close(native, reference, atol=0, rtol=0)
    native.float().square().mean().backward()
    native_grad = weight.grad
    weight.grad = None
    reference.float().square().mean().backward()
    torch.testing.assert_close(native_grad, weight.grad, atol=0, rtol=0)
