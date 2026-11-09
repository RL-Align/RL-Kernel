# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 shared expert on DeepGEMM FP8.

Properties: the forward reproduces the vLLM rollout composition (golden FP8
quantization, DeepGEMM, golden clamped SwiGLU) byte for byte; forward and dX
are batch-invariant and run-to-run identical in both backward modes; results
stay within FP8 error of an FP32 reference; and the provider fails closed.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.cuda.deepgemm import deepgemm_available
from rl_engine.kernels.ops.pytorch.activation.silu_and_mul_clamp import (
    silu_and_mul_clamp,
    silu_and_mul_clamp_backward,
)
from rl_engine.kernels.ops.pytorch.quant import fp8_group_quant
from rl_engine.moe.contract import SharedBatch


def _ready() -> bool:
    if not torch.cuda.is_available() or not deepgemm_available():
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return hasattr(ext, "fp8_group_quant_forward") and hasattr(ext, "det_gemm_fwd")


pytestmark = pytest.mark.skipif(
    not _ready(), reason="SM90 + pinned DeepGEMM + CUDA 12.8 JIT + rl_engine._C required"
)

PROD = (4096, 2048)  # hidden, ffn
SMALL = (256, 128)


def _provider(backward: str = "fp8"):
    from rl_engine.moe.backends.deepgemm_shared_expert import DeepGemmSharedExpertProvider

    return DeepGemmSharedExpertProvider(backward=backward)


def _weights(hidden: int, ffn: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    w1 = (torch.randn(2 * ffn, hidden, generator=gen) / hidden**0.5).to(torch.bfloat16).cuda()
    w2 = (torch.randn(hidden, ffn, generator=gen) / ffn**0.5).to(torch.bfloat16).cuda()
    return w1, w2


def _x(tokens: int, hidden: int, seed: int = 1, scale: float = 1.0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(tokens, hidden, generator=gen) * scale).to(torch.bfloat16).cuda()


def _batch(provider, x, w1, w2) -> SharedBatch:
    return SharedBatch(x=x, w_fc1=w1, w_fc2=w2, numeric_profile=provider.numeric_profile)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


def test_forward_matches_the_rollout_composition():
    import deep_gemm

    provider = _provider()
    w1, w2 = _weights(*SMALL)
    x = _x(37, SMALL[0])
    y, saved = provider.shared_expert_mlp_fwd(_batch(provider, x, w1, w2))
    weights = saved["weights"]
    z = torch.empty(37, 2 * SMALL[1], device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt(fp8_group_quant(x), weights.w1.pair(), z, disable_ue8m0_cast=True)
    h = silu_and_mul_clamp(z)
    y_ref = torch.empty(37, SMALL[0], device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt(fp8_group_quant(h), weights.w2.pair(), y_ref, disable_ue8m0_cast=True)
    assert _same(saved["z"], z) and _same(saved["h_bf16"], h) and _same(y, y_ref)


@pytest.mark.parametrize("backward", ["fp8", "bf16_det_gemm"])
def test_forward_and_dx_are_batch_invariant(backward):
    provider = _provider(backward)
    w1, w2 = _weights(*PROD)
    x = _x(300, PROD[0])
    gen = torch.Generator().manual_seed(2)
    dy = torch.randn(300, PROD[0], generator=gen).to(torch.bfloat16).cuda()
    y, saved = provider.shared_expert_mlp_fwd(_batch(provider, x, w1, w2))
    dx = provider.shared_expert_mlp_bwd(dy, _batch(provider, x, w1, w2), saved)
    assert dx.dtype == torch.float32 and dx.shape == x.shape
    for start, stop in ((0, 1), (1, 2), (127, 130), (200, 300)):
        sub = _batch(provider, x[start:stop].contiguous(), w1, w2)
        y_part, saved_part = provider.shared_expert_mlp_fwd(sub)
        dx_part = provider.shared_expert_mlp_bwd(dy[start:stop].contiguous(), sub, saved_part)
        assert _same(y_part, y[start:stop])
        assert _same(dx_part, dx[start:stop])
    y2, saved2 = provider.shared_expert_mlp_fwd(_batch(provider, x, w1, w2))
    dx2 = provider.shared_expert_mlp_bwd(dy, _batch(provider, x, w1, w2), saved2)
    assert _same(y2, y) and _same(dx2, dx)


def test_accuracy_against_fp32_and_the_bf16_fallback():
    provider = _provider("fp8")
    reference = _provider("bf16_det_gemm")
    w1, w2 = _weights(*PROD)
    x = _x(512, PROD[0])
    gen = torch.Generator().manual_seed(3)
    dy = torch.randn(512, PROD[0], generator=gen).to(torch.bfloat16).cuda()

    y, saved = provider.shared_expert_mlp_fwd(_batch(provider, x, w1, w2))
    dx = provider.shared_expert_mlp_bwd(dy, _batch(provider, x, w1, w2), saved)
    y_r, saved_r = reference.shared_expert_mlp_fwd(_batch(reference, x, w1, w2))
    dx_r = reference.shared_expert_mlp_bwd(dy, _batch(reference, x, w1, w2), saved_r)
    assert _same(y, y_r)  # same forward in both modes

    # FP32 GEMMs on the dequantized FP8 weights (the model's weights). Both
    # backward modes differentiate at the saved forward z, so the dX reference
    # does too. Measured on an H200: y 4.7%, FP8 dX 3.8%, BF16 dX 0.74%.
    w1_d, w2_d = (w.float() for w in saved["weights"].bf16())
    y32 = silu_and_mul_clamp((x.float() @ w1_d.t()).to(torch.bfloat16)).float() @ w2_d.t()
    dh32 = (dy.float() @ w2_d).to(torch.bfloat16)
    dx32 = silu_and_mul_clamp_backward(dh32, saved["z"]).float() @ w1_d
    assert _rel(y, y32) < 0.06
    assert _rel(dx, dx32) < 0.05
    assert _rel(dx_r, dx32) < 0.012
    assert _rel(dx, dx_r) < 0.05


def test_clamp_is_active_and_differs_from_the_p5_oracle():
    from rl_engine.moe import oracle

    provider = _provider()
    w1, w2 = _weights(*SMALL)
    x = _x(64, SMALL[0], scale=60.0)  # pushes gate/up beyond the limit of 10
    y, saved = provider.shared_expert_mlp_fwd(_batch(provider, x, w1, w2))
    z = saved["z"].float()
    ffn = SMALL[1]
    assert bool((z[:, :ffn] > 10).any()) and bool((z[:, ffn:].abs() > 10).any())
    y_oracle, _ = oracle.shared_expert_mlp_fwd(SharedBatch(x=x, w_fc1=w1, w_fc2=w2))
    assert _rel(y, y_oracle) > 0.1  # the P5 oracle does not clamp


def test_prepared_weights_are_exact():
    from rl_engine.moe.backends.deepgemm_shared_expert import quantize_weight

    w1, _ = _weights(*SMALL)
    q = quantize_weight(w1)
    assert _same(quantize_weight(w1.t().contiguous()).q, q.t().q)
    assert torch.equal(quantize_weight(w1.t().contiguous()).scale, q.t().scale)
    requantized = quantize_weight(q.dequantize())
    assert _same(requantized.q, q.q) and torch.equal(requantized.scale, q.scale)


def test_checkpoint_fp8_weights_are_accepted():
    from rl_engine.moe.backends.deepgemm_shared_expert import (
        DeepGemmSharedExpertProvider,
        quantize_weight,
    )

    provider = _provider()
    w1, w2 = _weights(*SMALL)
    x = _x(9, SMALL[0])
    q1, q2 = quantize_weight(w1), quantize_weight(w2)
    prepared = DeepGemmSharedExpertProvider.prepare_fp8(q1.pair(), q2.pair())
    y, _ = provider.forward(x, prepared)
    y_ref, _ = provider.shared_expert_mlp_fwd(_batch(provider, x, w1, w2))
    assert _same(y, y_ref)


def test_fails_closed():
    from rl_engine.moe.backends.deepgemm_shared_expert import DeepGemmSharedExpertProvider

    with pytest.raises(ValueError, match="backward must be one of"):
        DeepGemmSharedExpertProvider(backward="tf32")
    provider = _provider()
    w1, w2 = _weights(*SMALL)
    x = _x(4, SMALL[0])
    with pytest.raises(NotImplementedError, match="fail-closed"):
        provider.shared_expert_mlp_fwd(SharedBatch(x=x, w_fc1=w1, w_fc2=w2))
    with pytest.raises(NotImplementedError, match="fail-closed"):
        provider.shared_expert_mlp_fwd(_batch(provider, x.cpu(), w1.cpu(), w2.cpu()))
