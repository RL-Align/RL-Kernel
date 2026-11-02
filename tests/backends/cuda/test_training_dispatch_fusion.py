"""Dispatch grouping must retain sampling support, FFN outputs and gradients."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA required",
)


def test_unique_row_copy_accepts_unpacked_tp_vocab_view():
    from rl_engine.backends.cuda.sampling.unique_scatter import copy_unique_rows_

    torch.manual_seed(41)
    full = torch.randn(5, 388, device="cuda") > 0
    shard = full[:, 97:194]
    indices = torch.tensor([0, 2, 4, 7, 8], device="cuda")
    output = torch.ones(9, 97, device="cuda", dtype=torch.bool)
    expected = output.clone()
    expected.index_copy_(0, indices, shard)
    copy_unique_rows_(output, indices, shard)
    assert torch.equal(expected, output)


def test_fused_down_gradient_still_rejects_noncanonical_shape(monkeypatch):
    from rl_engine.backends.cuda.ffn.ffn import _canonical_tp_input_gradient

    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "cublaslt_nosplitk")
    monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", "8")
    grad = torch.zeros(1, 32, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(32, 17, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="does not divide canonical TP"):
        _canonical_tp_input_gradient(grad, weight, tp_world=4, column=False, disable_split_k=True)


@pytest.mark.parametrize("tokens", [1, 129, 1024])
@pytest.mark.parametrize("canonical_tp", [2, 4, 8])
def test_packed_training_matches_separate_gate_up(tokens, canonical_tp, monkeypatch):
    from rl_engine.backends.cuda.ffn.ffn import qwen3_ffn

    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "cublaslt_nosplitk")
    monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", str(canonical_tp))
    torch.manual_seed(tokens + canonical_tp)
    x = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    packed = torch.randn(2048, 512, device="cuda", dtype=torch.bfloat16)
    gate, up = [p.detach().requires_grad_() for p in packed.chunk(2)]
    down = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    inputs = (x, gate, up, down)
    separate = qwen3_ffn(*inputs, deterministic=True, disable_split_k=True)
    fused = qwen3_ffn(
        *inputs, fused_gate_up_weight=packed, deterministic=True, disable_split_k=True
    )
    dy = torch.randn_like(separate)
    separate_grads = torch.autograd.grad(separate, inputs, dy)
    fused_grads = torch.autograd.grad(fused, inputs, dy)
    for a, b in zip((separate, *separate_grads), (fused, *fused_grads)):
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("top_p,top_k", [(0.95, None), (0.7, 80), (1.0, None)])
def test_sampling_transport_chunk_preserves_exact_support(top_p, top_k):
    from rl_engine.ops.sampling.policy import vocab_parallel_sampling_keep_mask

    torch.manual_seed(29)
    logits = torch.randn(257, 151936, device="cuda", dtype=torch.bfloat16)
    active = torch.arange(257, device="cuda") % 3 != 0
    args = dict(
        real_vocab_size=151936, temperature=0.7, top_p=top_p, top_k=top_k, active_rows=active
    )
    before = vocab_parallel_sampling_keep_mask(logits, chunk_size=32, **args)
    after = vocab_parallel_sampling_keep_mask(logits, **args)
    assert torch.equal(before, after)
