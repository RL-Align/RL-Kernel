# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.pytorch.sampling.gumbel_softmax import NativeGumbelSoftmaxOp

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton Gumbel-Softmax requires a CUDA device and Triton.",
)


def _inputs(seed, *, device="cpu", dtype=torch.float32, shape=(4, 17)):
    gen = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(*shape, generator=gen, device=device, dtype=dtype)
    # Keep deterministic test noise in fp32, then let each backend cast as needed.
    gumbels = (
        -torch.empty(*shape, device=device, dtype=torch.float32).exponential_(generator=gen).log()
    )
    return logits, gumbels


def _run_grad(op, logits, gumbels, grad_out, *, hard=False, tau=0.7):
    x = logits.detach().clone().requires_grad_(True)
    out = op(x, tau=tau, hard=hard, gumbels=gumbels)
    out.backward(grad_out)
    return out.detach(), x.grad


def test_native_matches_torch_reference_with_supplied_noise():
    logits, gumbels = _inputs(0)
    op = NativeGumbelSoftmaxOp()
    actual = op(logits, tau=0.9, hard=False, gumbels=gumbels)
    expected = torch.softmax((logits + gumbels.to(logits.dtype)) / 0.9, dim=-1)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_native_hard_is_one_hot_with_straight_through_gradient():
    logits, gumbels = _inputs(1)
    logits = logits.requires_grad_(True)
    out = NativeGumbelSoftmaxOp()(logits, tau=0.8, hard=True, gumbels=gumbels)
    assert torch.allclose(out.sum(dim=-1), torch.ones_like(out[..., 0]))
    assert torch.all((out == 0.0) | (out == 1.0))

    out[..., 0].sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_native_preserves_leading_shape():
    logits, gumbels = _inputs(2, shape=(2, 3, 19))
    out = NativeGumbelSoftmaxOp()(logits, tau=1.2, hard=False, gumbels=gumbels)
    assert out.shape == (2, 3, 19)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2, 3), atol=1e-6)


def test_native_temperature_changes_distribution_sharpness():
    logits, gumbels = _inputs(3, shape=(5, 23))
    op = NativeGumbelSoftmaxOp()
    cool = op(logits, tau=0.25, gumbels=gumbels)
    warm = op(logits, tau=2.0, gumbels=gumbels)
    assert cool.max(dim=-1).values.mean() > warm.max(dim=-1).values.mean()


def test_native_treats_supplied_gumbels_as_fixed_noise():
    logits, gumbels = _inputs(11)
    logits = logits.requires_grad_(True)
    gumbels = gumbels.requires_grad_(True)

    out = NativeGumbelSoftmaxOp()(logits, tau=0.8, hard=False, gumbels=gumbels)
    out[..., 0].sum().backward()

    assert logits.grad is not None
    assert gumbels.grad is None


def test_native_rejects_invalid_inputs():
    op = NativeGumbelSoftmaxOp()
    logits, gumbels = _inputs(4)
    with pytest.raises(ValueError, match="tau must be positive"):
        op(logits, tau=0.0, gumbels=gumbels)
    with pytest.raises(ValueError, match="gumbels shape"):
        op(logits, gumbels=gumbels[:, :-1])
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        op(torch.randn(8))


def test_registry_dispatch_matches_native():
    from rl_engine.kernels.registry import kernel_registry
    from rl_engine.platforms.device import device_ctx

    device = device_ctx.device if device_ctx.device_type == "cuda" or device_ctx.is_rocm else "cpu"
    logits, gumbels = _inputs(5, device=device)
    op = kernel_registry.get_op("gumbel_softmax")
    out = op(logits, tau=0.7, hard=False, gumbels=gumbels)
    ref = NativeGumbelSoftmaxOp()(logits, tau=0.7, hard=False, gumbels=gumbels)
    assert torch.allclose(out.cpu(), ref.cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton import is required for fallback wrapper test.")
def test_triton_internal_gumbel_generation_precompute_guard(monkeypatch):
    from rl_engine.kernels.ops.triton.sampling import gumbel_softmax as triton_gs

    logits, _ = _inputs(12, shape=(2, 9))
    assert not triton_gs._needs_precomputed_gumbels(logits)

    monkeypatch.setattr(triton_gs, "_MAX_RAND_OFFSET", logits.numel() - 1)
    assert triton_gs._needs_precomputed_gumbels(logits)


@requires_triton_cuda
def test_triton_forward_matches_native_soft_fp32():
    from rl_engine.kernels.ops.triton.sampling.gumbel_softmax import TritonGumbelSoftmaxOp

    logits, gumbels = _inputs(6, device="cuda", shape=(7, 257))
    out = TritonGumbelSoftmaxOp()(logits, tau=0.8, hard=False, gumbels=gumbels)
    ref = NativeGumbelSoftmaxOp()(logits, tau=0.8, hard=False, gumbels=gumbels)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@requires_triton_cuda
def test_triton_forward_matches_native_hard_fp32():
    from rl_engine.kernels.ops.triton.sampling.gumbel_softmax import TritonGumbelSoftmaxOp

    logits, gumbels = _inputs(7, device="cuda", shape=(9, 251))
    out = TritonGumbelSoftmaxOp()(logits, tau=0.6, hard=True, gumbels=gumbels)
    ref = NativeGumbelSoftmaxOp()(logits, tau=0.6, hard=True, gumbels=gumbels)
    assert torch.allclose(out, ref, atol=0.0)
    assert torch.allclose(out.sum(dim=-1), torch.ones_like(out[..., 0]))


@requires_triton_cuda
def test_triton_backward_matches_native():
    from rl_engine.kernels.ops.triton.sampling.gumbel_softmax import TritonGumbelSoftmaxOp

    logits, gumbels = _inputs(8, device="cuda", shape=(4, 3, 127))
    grad_out = torch.randn_like(logits)
    triton_out, triton_grad = _run_grad(
        TritonGumbelSoftmaxOp(), logits, gumbels, grad_out, hard=True, tau=0.9
    )
    native_out, native_grad = _run_grad(
        NativeGumbelSoftmaxOp(), logits, gumbels, grad_out, hard=True, tau=0.9
    )
    assert torch.allclose(triton_out, native_out, atol=0.0)
    assert torch.allclose(triton_grad, native_grad, atol=2e-5, rtol=2e-5)


@requires_triton_cuda
def test_triton_preserves_leading_shape_and_dtype():
    from rl_engine.kernels.ops.triton.sampling.gumbel_softmax import TritonGumbelSoftmaxOp

    logits, gumbels = _inputs(9, device="cuda", dtype=torch.float16, shape=(2, 5, 313))
    out = TritonGumbelSoftmaxOp()(logits, tau=1.0, hard=False, gumbels=gumbels)
    assert out.shape == (2, 5, 313)
    assert out.dtype == torch.float16
    assert torch.allclose(out.float().sum(dim=-1), torch.ones(2, 5, device="cuda"), atol=1e-3)


@requires_triton_cuda
def test_triton_generated_noise_smoke():
    from rl_engine.kernels.ops.triton.sampling.gumbel_softmax import TritonGumbelSoftmaxOp

    logits = torch.randn(8, 1024, device="cuda")
    out = TritonGumbelSoftmaxOp()(logits, tau=1.0, hard=False, seed=2026)
    assert out.shape == logits.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(dim=-1), torch.ones(8, device="cuda"), atol=1e-5)


@requires_triton_cuda
def test_triton_chunked_forward_matches_native_soft_fp32(monkeypatch):
    from rl_engine.kernels.ops.triton.sampling import gumbel_softmax as triton_gs

    monkeypatch.setattr(triton_gs, "_MAX_BLOCK_V", 8)
    monkeypatch.setattr(triton_gs, "_CHUNK_V", 8)
    logits, gumbels = _inputs(13, device="cuda", shape=(3, 19))
    out = triton_gs.TritonGumbelSoftmaxOp()(logits, tau=0.8, hard=False, gumbels=gumbels)
    ref = NativeGumbelSoftmaxOp()(logits, tau=0.8, hard=False, gumbels=gumbels)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@requires_triton_cuda
def test_triton_chunked_forward_matches_native_hard_fp32(monkeypatch):
    from rl_engine.kernels.ops.triton.sampling import gumbel_softmax as triton_gs

    monkeypatch.setattr(triton_gs, "_MAX_BLOCK_V", 8)
    monkeypatch.setattr(triton_gs, "_CHUNK_V", 8)
    logits, gumbels = _inputs(14, device="cuda", shape=(3, 19))
    out = triton_gs.TritonGumbelSoftmaxOp()(logits, tau=0.7, hard=True, gumbels=gumbels)
    ref = NativeGumbelSoftmaxOp()(logits, tau=0.7, hard=True, gumbels=gumbels)
    assert torch.allclose(out, ref, atol=0.0)
    assert torch.allclose(out.sum(dim=-1), torch.ones(3, device="cuda"))


@requires_triton_cuda
def test_triton_chunked_backward_matches_native(monkeypatch):
    from rl_engine.kernels.ops.triton.sampling import gumbel_softmax as triton_gs

    monkeypatch.setattr(triton_gs, "_MAX_BLOCK_V", 8)
    monkeypatch.setattr(triton_gs, "_CHUNK_V", 8)
    logits, gumbels = _inputs(15, device="cuda", shape=(2, 3, 19))
    grad_out = torch.randn_like(logits)
    triton_out, triton_grad = _run_grad(
        triton_gs.TritonGumbelSoftmaxOp(), logits, gumbels, grad_out, hard=True, tau=0.9
    )
    native_out, native_grad = _run_grad(
        NativeGumbelSoftmaxOp(), logits, gumbels, grad_out, hard=True, tau=0.9
    )
    assert torch.allclose(triton_out, native_out, atol=0.0)
    assert torch.allclose(triton_grad, native_grad, atol=2e-5, rtol=2e-5)


@requires_triton_cuda
def test_triton_chunked_hard_nograd_is_one_hot(monkeypatch):
    from rl_engine.kernels.ops.triton.sampling import gumbel_softmax as triton_gs

    monkeypatch.setattr(triton_gs, "_MAX_BLOCK_V", 8)
    monkeypatch.setattr(triton_gs, "_CHUNK_V", 8)
    logits, gumbels = _inputs(16, device="cuda", shape=(2, 3, 19))
    with torch.no_grad():
        out = triton_gs.TritonGumbelSoftmaxOp()(logits, tau=0.7, hard=True, gumbels=gumbels)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2, 3, device="cuda"))
    assert torch.all((out == 0.0) | (out == 1.0))


@requires_triton_cuda
def test_triton_qwen3_vocab_smoke_with_supplied_gumbels():
    from rl_engine.kernels.ops.triton.sampling.gumbel_softmax import TritonGumbelSoftmaxOp

    logits, gumbels = _inputs(17, device="cuda", shape=(1, 151936))
    out = TritonGumbelSoftmaxOp()(logits, tau=1.0, hard=False, gumbels=gumbels)
    ref = NativeGumbelSoftmaxOp()(logits, tau=1.0, hard=False, gumbels=gumbels)
    assert out.shape == logits.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(dim=-1), torch.ones(1, device="cuda"), atol=1e-5)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@requires_triton_cuda
def test_triton_hard_nograd_generated_noise_fast_path():
    from rl_engine.kernels.ops.triton.sampling.gumbel_softmax import TritonGumbelSoftmaxOp

    logits = torch.randn(3, 5, 1024, device="cuda")
    with torch.no_grad():
        out = TritonGumbelSoftmaxOp()(logits, tau=0.7, hard=True, seed=2027)
    assert out.shape == logits.shape
    assert out.dtype == logits.dtype
    assert torch.allclose(out.sum(dim=-1), torch.ones(3, 5, device="cuda"))
    assert torch.all((out == 0.0) | (out == 1.0))
