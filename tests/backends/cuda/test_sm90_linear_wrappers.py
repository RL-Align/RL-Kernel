# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch


def _sm90_linear_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from rl_engine.backends.extension import _C, _EXT_AVAILABLE

        return (
            _EXT_AVAILABLE
            and hasattr(_C, "embedding_sm90_forward")
            and hasattr(_C, "lm_head_sm90_forward")
            and torch.cuda.get_device_capability()[0] == 9
        )
    except Exception:
        return False


requires_sm90_linear = pytest.mark.skipif(
    not _sm90_linear_available(),
    reason=(
        "SM90 embedding/lm_head kernels require an H200/Hopper-class GPU and "
        "KERNEL_ALIGN_FORCE_SM90=1."
    ),
)


def test_sm90_embedding_rejects_non_hopper_fallback(monkeypatch):
    from rl_engine.backends.cuda.embedding import embedding as embedding_module

    class FakeExtension:
        @staticmethod
        def embedding_sm90_forward(token_ids, weight):
            return weight[token_ids.long()]

    monkeypatch.setattr(embedding_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(embedding_module, "_C", FakeExtension)
    monkeypatch.setattr(
        embedding_module.SM90EmbeddingOp,
        "_can_use_sm90",
        staticmethod(lambda token_ids, weight: False),
    )
    op = embedding_module.SM90EmbeddingOp()
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        op.forward(torch.tensor([[1]]), torch.randn(4, 3))


def test_sm90_embedding_wrapper_calls_extension_symbol(monkeypatch):
    from rl_engine.backends.cuda.embedding import embedding as embedding_module

    calls = []

    class FakeExtension:
        @staticmethod
        def embedding_sm90_forward(token_ids, weight):
            calls.append(("forward", token_ids, weight))
            return weight[token_ids.long()]

        @staticmethod
        def embedding_sm90_forward_fp32(token_ids, weight):
            calls.append(("forward_fp32", token_ids, weight))
            return weight[token_ids.long()].float()

    monkeypatch.setattr(embedding_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(embedding_module, "_C", FakeExtension)
    monkeypatch.setattr(
        embedding_module.SM90EmbeddingOp,
        "_can_use_sm90",
        staticmethod(lambda token_ids, weight: True),
    )

    op = embedding_module.SM90EmbeddingOp()
    token_ids = torch.tensor([[1, 2], [3, 4]])
    weight = torch.randn(8, 5)

    assert op.forward(token_ids, weight).shape == (2, 2, 5)
    assert op.forward_fp32(token_ids, weight).dtype == torch.float32
    assert [name for name, *_ in calls] == ["forward", "forward_fp32"]


def test_sm90_embedding_backward_accumulates_duplicate_ids_deterministically(monkeypatch):
    from rl_engine.backends.cuda.embedding import embedding as embedding_module

    class FakeExtension:
        @staticmethod
        def embedding_sm90_forward(token_ids, weight):
            return weight[token_ids.long()]

        @staticmethod
        def embedding_sm90_forward_fp32(token_ids, weight):
            return weight[token_ids.long()].float()

    monkeypatch.setattr(embedding_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(embedding_module, "_C", FakeExtension)
    monkeypatch.setattr(
        embedding_module.SM90EmbeddingOp,
        "_can_use_sm90",
        staticmethod(lambda token_ids, weight: True),
    )

    token_ids = torch.tensor([[2, 1, 2], [3, 1, 2]])
    weight = torch.randn(5, 4, requires_grad=True)
    upstream = torch.arange(token_ids.numel() * weight.size(1), dtype=torch.float32).reshape(
        *token_ids.shape, weight.size(1)
    )

    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        (embedding_module.SM90EmbeddingOp().forward(token_ids, weight) * upstream).sum().backward()
    finally:
        torch.use_deterministic_algorithms(prev)

    expected = torch.zeros_like(weight)
    flat_ids = token_ids.reshape(-1)
    flat_upstream = upstream.reshape(-1, weight.size(1))
    for row, token_id in enumerate(flat_ids.tolist()):
        expected[token_id] += flat_upstream[row]
    assert torch.equal(weight.grad, expected)


def test_sm90_lm_head_wrapper_calls_extension_symbol(monkeypatch):
    from rl_engine.backends.cuda.embedding import lm_head as lm_head_module

    calls = []

    class FakeExtension:
        @staticmethod
        def lm_head_sm90_forward(hidden, weight, bias):
            calls.append(("forward", hidden, weight, bias))
            out = hidden.float().matmul(weight.float().t())
            if bias is not None:
                out = out + bias.float()
            return out.to(hidden.dtype)

        @staticmethod
        def lm_head_sm90_forward_fp32(hidden, weight, bias):
            calls.append(("forward_fp32", hidden, weight, bias))
            out = hidden.float().matmul(weight.float().t())
            if bias is not None:
                out = out + bias.float()
            return out.float()

    monkeypatch.setattr(lm_head_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(lm_head_module, "_C", FakeExtension)
    monkeypatch.setattr(
        lm_head_module.SM90LMHeadOp,
        "_can_use_sm90",
        staticmethod(lambda hidden, weight, bias: True),
    )

    op = lm_head_module.SM90LMHeadOp()
    hidden = torch.randn(2, 3, 7)
    weight = torch.randn(11, 7)
    bias = torch.randn(11)

    assert op.forward(hidden, weight, bias=bias).shape == (2, 3, 11)
    assert op.forward_fp32(hidden, weight, bias=bias).dtype == torch.float32
    assert [name for name, *_ in calls] == ["forward", "forward_fp32"]


def test_sm90_lm_head_bf16_backward_uses_fp32_reference_vjp(monkeypatch):
    from rl_engine.backends.cuda.embedding import lm_head as lm_head_module

    class FakeExtension:
        @staticmethod
        def lm_head_sm90_forward(hidden, weight, bias):
            out = hidden.float().matmul(weight.float().t())
            if bias is not None:
                out = out + bias.float()
            return out.to(hidden.dtype)

        @staticmethod
        def lm_head_sm90_forward_fp32(hidden, weight, bias):
            out = hidden.float().matmul(weight.float().t())
            if bias is not None:
                out = out + bias.float()
            return out.float()

        @staticmethod
        def det_gemm_fwd(a, b):
            return a.float().matmul(b.float()).to(a.dtype)

    monkeypatch.setattr(lm_head_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(lm_head_module, "_C", FakeExtension)
    monkeypatch.setattr(
        lm_head_module.SM90LMHeadOp,
        "_can_use_sm90",
        staticmethod(lambda hidden, weight, bias: True),
    )

    hidden = torch.randn(2, 3, 5, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(7, 5, dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(7, dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(2, 3, 7, dtype=torch.bfloat16)

    out = lm_head_module.SM90LMHeadOp().forward(hidden, weight, bias=bias)
    out.backward(dy)

    flat_hidden = hidden.detach().reshape(-1, hidden.size(-1))
    flat_dy = dy.reshape(-1, weight.size(0))
    expected_hidden = (
        flat_dy.float().matmul(weight.detach().float()).to(torch.bfloat16).reshape_as(hidden)
    )
    expected_weight = flat_dy.float().t().matmul(flat_hidden.float()).to(torch.bfloat16)
    expected_bias = flat_dy.float().sum(0).to(torch.bfloat16)

    assert torch.equal(hidden.grad, expected_hidden.reshape_as(hidden))
    assert torch.equal(weight.grad, expected_weight)
    assert torch.equal(bias.grad, expected_bias)


@requires_sm90_linear
def test_sm90_embedding_forward_matches_direct_gather_on_h200_hopper():
    from rl_engine.backends.cuda.embedding.embedding import SM90EmbeddingOp

    device = torch.device("cuda")
    token_ids = torch.tensor([[0, 7, 3], [5, 1, 7]], device=device)
    weight = torch.randn(11, 13, device=device, dtype=torch.bfloat16)

    out = SM90EmbeddingOp().forward(token_ids, weight)
    out_fp32 = SM90EmbeddingOp().forward_fp32(token_ids, weight)

    assert torch.equal(out, weight[token_ids])
    assert torch.equal(out_fp32, weight[token_ids].float())


@requires_sm90_linear
def test_sm90_lm_head_forward_is_slice_batch_invariant_on_h200_hopper():
    from rl_engine.backends.cuda.embedding.lm_head import SM90LMHeadOp

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(150)
    hidden = torch.randn(4, 5, 64, device=device, dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(37, 64, device=device, dtype=torch.bfloat16, generator=generator)
    bias = torch.randn(37, device=device, dtype=torch.bfloat16, generator=generator)
    op = SM90LMHeadOp()

    full = op.forward(hidden, weight, bias=bias)
    full_fp32 = op.forward_fp32(hidden, weight, bias=bias)

    assert torch.equal(op.forward(hidden[2:3], weight, bias=bias), full[2:3])
    assert torch.equal(op.forward_fp32(hidden[2:3], weight, bias=bias), full_fp32[2:3])
