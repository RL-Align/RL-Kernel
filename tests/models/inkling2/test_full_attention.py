# SPDX-License-Identifier: Apache-2.0
"""INK-15 full-attention contract tests."""

import pytest
import torch

from rl_engine.models.inkling2.full_attention import full_attention


def test_full_attention_matches_manual_gqa_reference():
    torch.manual_seed(3)
    query = torch.randn(1, 4, 4, 3)
    key = torch.randn(1, 4, 2, 3)
    value = torch.randn(1, 4, 2, 3)
    out = full_attention(query, key, value, causal=False)

    key_gqa = key.repeat_interleave(2, dim=2).float().transpose(1, 2)
    value_gqa = value.repeat_interleave(2, dim=2).float().transpose(1, 2)
    scores = torch.matmul(query.float().transpose(1, 2), key_gqa.transpose(-2, -1)) / 3**0.5
    expected = (
        torch.matmul(torch.softmax(scores, dim=-1), value_gqa).transpose(1, 2).to(query.dtype)
    )
    torch.testing.assert_close(out, expected)


def test_document_boundaries_and_sink_change_only_allowed_attention():
    query = torch.ones(1, 3, 2, 2)
    key = torch.ones(1, 3, 1, 2)
    value = torch.tensor([[[[1.0, 0.0]], [[2.0, 0.0]], [[0.0, 4.0]]]])
    documents = torch.tensor([[0, 0, 1]])
    without_sink = full_attention(
        query, key, value, query_documents=documents, key_documents=documents, causal=False
    )
    with_sink = full_attention(
        query,
        key,
        value,
        sink=torch.tensor([10.0, 10.0]),
        query_documents=documents,
        key_documents=documents,
        causal=False,
    )
    assert torch.equal(without_sink[0, 2], value[0, 2].expand(2, 2))
    assert not torch.equal(with_sink, without_sink)


def test_output_gate_and_backward_are_defined():
    query = torch.randn(2, 4, 2, 3, requires_grad=True)
    key = torch.randn(2, 4, 1, 3, requires_grad=True)
    value = torch.randn(2, 4, 1, 3, requires_grad=True)
    gate = torch.zeros(2, requires_grad=True)
    output = full_attention(query, key, value, output_gate=gate)
    output.square().mean().backward()
    for tensor in (query, key, value, gate):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_cuda_full_attention_matches_reference_and_backward():
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE
    from rl_engine.models.inkling2.full_attention import full_attention_cuda

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_full_attention_forward"):
        pytest.skip("Inkling 2 full attention CUDA extension is unavailable")
    torch.manual_seed(19)
    query = torch.randn(1, 3, 4, 5, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(1, 4, 2, 5, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(1, 4, 2, 5, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(4, 3, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    sink = torch.randn(4, device="cuda", dtype=torch.float32, requires_grad=True)
    gate = torch.randn(4, device="cuda", dtype=torch.float32, requires_grad=True)
    documents_q = torch.tensor([[0, 0, 1]], device="cuda")
    documents_k = torch.tensor([[0, 0, 1, 1]], device="cuda")

    native = full_attention_cuda(
        query,
        key,
        value,
        relative_bias=bias,
        sink=sink,
        output_gate=gate,
        query_documents=documents_q,
        key_documents=documents_k,
        causal=False,
    )
    reference = full_attention(
        query,
        key,
        value,
        relative_bias=bias,
        sink=sink,
        output_gate=gate,
        query_documents=documents_q,
        key_documents=documents_k,
        causal=False,
    )
    torch.testing.assert_close(native, reference, atol=2e-2, rtol=0)
    native.float().square().mean().backward()
    assert query.grad is not None and key.grad is not None and value.grad is not None
    assert bias.grad is not None and sink.grad is not None and gate.grad is not None
