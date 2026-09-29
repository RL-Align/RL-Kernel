# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fused shared-expert fc1 + SwiGLU (csrc/cuda/moe/fused_shared_expert_mlp.cu).

The kernel is a performance rewrite of an existing composition, so almost every
assertion here is ``torch.equal`` rather than a tolerance: fusing fc1 with the
activation must not move a single byte.

The reference is built from ``det_gemm`` plus the SwiGLU written in torch.
That is exact, not approximate: nvcc's ``expf`` backs both ``torch.sigmoid``
and the CUDA core, so ``(g * sigmoid(g)) * u`` in FP32 reproduces the kernel's
``__fmul_rn(__fmul_rn(g, sig), u)`` bit for bit. Writing the reference this way
keeps the tests runnable against the default build, which does not compile
``shared_expert_mlp.cu``.
"""

from __future__ import annotations

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_SYMBOLS = ("fused_shared_expert_fc1_swiglu", "det_gemm_fwd_rhs_transposed")


@pytest.fixture(scope="module")
def ext():
    try:
        from rl_engine import _C
    except ImportError as exc:  # pragma: no cover - depends on the build
        pytest.skip(f"rl_engine._C is not built: {exc}")
    missing = [s for s in _SYMBOLS if not hasattr(_C, s)]
    if missing:
        pytest.skip(f"rl_engine._C lacks {missing}; rebuild the extension")
    return _C


def _operands(t: int, h: int, f: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(t, h, generator=g) * 0.7).to(torch.bfloat16).cuda()
    w1 = (torch.randn(2 * f, h, generator=g) / h**0.5).to(torch.bfloat16).cuda()
    return x, w1


def _reference(ext, x: torch.Tensor, w1: torch.Tensor) -> torch.Tensor:
    """The unfused path this kernel replaces: det_gemm fc1, then the SwiGLU."""
    f = w1.shape[0] // 2
    z = ext.det_gemm_fwd_rhs_transposed(x, w1).float()
    gate, up = z[:, :f], z[:, f:]
    return ((gate * torch.sigmoid(gate)) * up).to(torch.bfloat16)


# (T, H, F): the first block takes the SM90 tensor-core tile (F % 32 == 0),
# the second falls back to the scalar K-tree.
_SM90_SHAPES = [(128, 256, 64), (2048, 4096, 2048), (200, 512, 128), (1, 4096, 2048), (7, 128, 32)]
_SCALAR_SHAPES = [(64, 128, 40), (33, 96, 24)]


@requires_cuda
@pytest.mark.parametrize("shape", _SM90_SHAPES + _SCALAR_SHAPES)
def test_matches_the_unfused_det_path(ext, shape):
    """Fusing fc1 with the activation must be byte-neutral."""
    x, w1 = _operands(*shape)
    got = ext.fused_shared_expert_fc1_swiglu(x, w1)
    assert got.dtype is torch.bfloat16 and got.shape == (shape[0], shape[2])
    assert torch.equal(got, _reference(ext, x, w1)), f"fused path diverged at {shape}"


@requires_cuda
def test_batch_invariance(ext):
    """A row's bytes must not depend on how many rows are quantized with it."""
    x, w1 = _operands(512, 4096, 2048)
    full = ext.fused_shared_expert_fc1_swiglu(x, w1)
    # Rows on both sides of the BM = 128 tile boundary.
    for t in (0, 1, 127, 128, 129, 300, 511):
        one = ext.fused_shared_expert_fc1_swiglu(x[t : t + 1].contiguous(), w1)
        assert torch.equal(one[0], full[t]), f"row {t} diverged"
    chunk = ext.fused_shared_expert_fc1_swiglu(x[100:164].contiguous(), w1)
    assert torch.equal(chunk, full[100:164]), "sub-batch slice diverged"


@requires_cuda
def test_run_to_run_is_bitwise_stable(ext):
    x, w1 = _operands(256, 1024, 512)
    first = ext.fused_shared_expert_fc1_swiglu(x, w1)
    for _ in range(4):
        assert torch.equal(ext.fused_shared_expert_fc1_swiglu(x, w1), first)


@requires_cuda
def test_k_tree_survives_a_half_k_split(ext):
    """The property the K-tree exists for: a contiguous half-K shard is one
    child of the tree, so two TP=2 partial sums add back to the TP=1 result.

    Asserted on the GEMM half, which is where it applies. It cannot hold across
    the fused activation, because SiLU is not additive:
    ``SiLU(g1+g2)*(u1+u2) != SiLU(g1)*u1 + SiLU(g2)*u2``. Under the standard
    DSv4 layout fc1 is column-parallel and its K is never split, which is what
    makes fusing the activation legal there in the first place.
    """
    x, w1 = _operands(256, 4096, 512)
    h = x.shape[1]
    full = ext.det_gemm_fwd_rhs_transposed(x, w1)
    left = ext.det_gemm_fwd_rhs_transposed(x[:, : h // 2].contiguous(), w1[:, : h // 2].contiguous())
    right = ext.det_gemm_fwd_rhs_transposed(x[:, h // 2 :].contiguous(), w1[:, h // 2 :].contiguous())
    assert torch.equal(full, (left.float() + right.float()).to(torch.bfloat16))


@requires_cuda
def test_fails_closed_on_bad_input(ext):
    x, w1 = _operands(64, 128, 32)
    with pytest.raises(RuntimeError, match="must be BF16"):
        ext.fused_shared_expert_fc1_swiglu(x.float(), w1)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        ext.fused_shared_expert_fc1_swiglu(x.t().contiguous().t(), w1)
    with pytest.raises(RuntimeError, match="multiple of 32"):
        ext.fused_shared_expert_fc1_swiglu(x[:, :100].contiguous(), w1[:, :100].contiguous())
    with pytest.raises(RuntimeError, match="K mismatch"):
        ext.fused_shared_expert_fc1_swiglu(x, w1[:, :64].contiguous())


# --- provider ---------------------------------------------------------------


@pytest.fixture
def provider(ext):
    from rl_engine.moe.backends.shared_expert import CudaFusedSharedExpertProvider

    del ext
    try:
        return CudaFusedSharedExpertProvider()
    except NotImplementedError as exc:
        pytest.skip(f"fused shared-expert backend unavailable: {exc}")


def _batch(t: int, h: int, f: int, seed: int = 3):
    from rl_engine.moe.contract import SharedBatch

    g = torch.Generator().manual_seed(seed)
    return SharedBatch(
        x=(torch.randn(t, h, generator=g) * 0.7).to(torch.bfloat16).cuda(),
        w_fc1=(torch.randn(2 * f, h, generator=g) / h**0.5).to(torch.bfloat16).cuda(),
        w_fc2=(torch.randn(h, f, generator=g) / f**0.5).to(torch.bfloat16).cuda(),
        numeric_profile="p5-det-gemm-v1",
    )


@requires_cuda
def test_provider_forward_matches_the_unfused_composition(provider, ext):
    """End to end, including fc2: identical to running the pieces separately."""
    batch = _batch(256, 1024, 512)
    y, saved = provider.shared_expert_mlp_fwd(batch)
    h_ref = _reference(ext, batch.x, batch.w_fc1)
    y_ref = ext.det_gemm_fwd_rhs_transposed(h_ref, batch.w_fc2).float().to(torch.bfloat16)
    assert torch.equal(saved["h_bf16"], h_ref)
    assert torch.equal(y, y_ref)
    assert "z32" not in saved, "the fused kernel must not materialize z"


@requires_cuda
def test_provider_is_forward_only(provider):
    batch = _batch(64, 256, 128)
    _, saved = provider.shared_expert_mlp_fwd(batch)
    with pytest.raises(NotImplementedError, match="forward-only"):
        provider.shared_expert_mlp_bwd(torch.zeros(64, 256, device="cuda"), batch, saved)


@requires_cuda
def test_provider_rejects_a_foreign_profile(provider):
    batch = _batch(64, 256, 128)
    object.__setattr__(batch, "numeric_profile", "oracle-fp32-serial-v1")
    with pytest.raises(NotImplementedError):
        provider.shared_expert_mlp_fwd(batch)
