# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Bit-equality harness and tests for the Qwen-Image attn-out bias GEMM.

The harness implements the frozen comparison procedure from the contract
(docs/operators/attn-out-bias-gemm.md):

    1. assert shapes match (explicit, for a locatable failure message)
    2. assert dtypes match
    3. materialise contiguous copies (a copy never changes logical element bits;
       column slices / transposes need it for the dtype reinterpretation)
    4. compare the *bit patterns of the logical elements* via a dtype bitcast
       (int16 for bf16, int32 for fp32) -- not the underlying allocation bytes

The bitcast assertion distinguishes +0.0/-0.0 and NaN payloads, which
``torch.equal`` (value semantics) does not; both assertions are always issued
together -- the value assertion gives readable failures, the bitcast assertion
carries the contract.

The acceptance structure is star-shaped: every backend (pytorch reference,
Triton, CUDA) is compared byte-for-byte against the FP32 CPU same-tree
reference; cross-backend equality is a transitive corollary; batch-invariance
checks compare a backend against itself across batch composition, shape,
padding, and repeat runs, without the reference involved.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from rl_engine.kernels.ops.pytorch.linear.attn_out_bias_gemm import (
    NativeAttnOutBiasGemmOp,
    tree_gemm_fp32,
)

_DTYPE_BITVIEW = {torch.bfloat16: torch.int16, torch.float32: torch.int32}

_DIM = 64  # small stand-in for 3072 keeps CPU reference iterations short
_SIZES = [1, 2, 3, 7, 20]


def _bitview(dtype: torch.dtype) -> torch.dtype:
    if dtype not in _DTYPE_BITVIEW:
        raise ValueError(f"no bitview mapping for {dtype}")
    return _DTYPE_BITVIEW[dtype]


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Contract comparison: logical-element bit patterns, not allocation bytes."""
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    a_c = a.contiguous().cpu()
    b_c = b.contiguous().cpu()
    return bool(torch.equal(a_c.view(_bitview(a.dtype)), b_c.view(_bitview(b.dtype))))


def assert_bitwise_equal(a: torch.Tensor, b: torch.Tensor, context: str = "") -> None:
    prefix = f"[{context}] " if context else ""
    assert a.shape == b.shape, f"{prefix}shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
    assert a.dtype == b.dtype, f"{prefix}dtype mismatch: {a.dtype} vs {b.dtype}"
    assert torch.equal(a, b), f"{prefix}value mismatch (torch.equal)"
    if a.numel() > 0:
        assert bitwise_equal(a, b), f"{prefix}bit-pattern mismatch (bitcast compare)"


def _make_inputs(rows: int, dim: int, dtype: torch.dtype, seed: int = 1234):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, dim, generator=gen).to(dtype)
    weight = torch.randn(dim, dim, generator=gen).to(dtype)
    bias = torch.randn(dim, generator=gen).to(dtype)
    return x, weight, bias


class TestReferenceSelfChecks:
    """The same-tree reference must be self-consistent before it judges anyone."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("rows", _SIZES)
    def test_deterministic_repeat(self, dtype, rows):
        op = NativeAttnOutBiasGemmOp()
        x, w, b = _make_inputs(rows, _DIM, dtype)
        assert_bitwise_equal(op(x, w, bias=b), op(x, w, bias=b), f"repeat rows={rows}")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_batch_layout_invariance(self, dtype):
        op = NativeAttnOutBiasGemmOp()
        x, w, b = _make_inputs(7, _DIM, dtype)
        full = op(x, w, bias=b)
        # full-batch row vs the same row computed alone
        assert_bitwise_equal(op(x[:1], w, bias=b), full[:1], "row0 alone")
        assert_bitwise_equal(op(x[3:5], w, bias=b), full[3:5], "rows 3:5 alone")
        # [1, S] leading-dim variants
        assert_bitwise_equal(op(x.unsqueeze(0), w, bias=b).squeeze(0), full, "leading [1,S]")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_padding_invariance(self, dtype):
        op = NativeAttnOutBiasGemmOp()
        x, w, b = _make_inputs(4, _DIM, dtype)
        pad = torch.randn(6, _DIM).to(dtype)
        padded = torch.cat([x, pad], dim=0)
        assert_bitwise_equal(op(padded, w, bias=b)[:4], op(x, w, bias=b), "padded prefix")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_fp32_upcast_gold(self, dtype):
        op = NativeAttnOutBiasGemmOp()
        x, w, b = _make_inputs(5, _DIM, dtype)
        out = op(x, w, bias=b)
        gold = op.forward_fp32(x, w, bias=b)
        # the single output cast must be exactly the RNE cast of the fp32 gold
        assert_bitwise_equal(out, gold.to(dtype), "single-cast vs gold")

    def test_zero_reduction_dim(self):
        a = torch.zeros(3, 0)
        bvec = torch.zeros(4, 0)
        out = tree_gemm_fp32(a, bvec)
        assert out.shape == (3, 4)
        assert bool((out == 0).all())

    def test_short_tail_leaf_tree_matches_manual(self):
        # R=33 -> one 32-wide leaf + one 1-wide tail leaf; mid-split tree
        # T(0,2) = T(0,1) + T(1,2). Verify against a hand-built tree with the
        # same correctly-rounded FMA chains (torch.addcmul).
        gen = torch.Generator().manual_seed(7)
        a = torch.randn(2, 33, generator=gen)
        b = torch.randn(3, 33, generator=gen)

        def fma_leaf(a_cols, b_cols):
            acc = torch.zeros(2, 3)
            for k in range(a_cols.size(1)):
                acc = torch.addcmul(acc, a_cols[:, k].unsqueeze(1), b_cols[:, k].unsqueeze(0))
            return acc

        leaf0 = fma_leaf(a[:, :32], b[:, :32])
        leaf1 = fma_leaf(a[:, 32:], b[:, 32:])
        manual = leaf0 + leaf1
        assert_bitwise_equal(tree_gemm_fp32(a, b), manual, "R=33 manual tree")


class TestReferenceBackward:
    """Gradient discipline: autograd through the op vs the explicit tree VJP."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_autograd_gradients_flow(self, dtype):
        op = NativeAttnOutBiasGemmOp()
        x, w, b = _make_inputs(3, _DIM, dtype)
        for tensor in (x, w, b):
            tensor.requires_grad_(True)
        out = op(x, w, bias=b)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        for tensor, name in ((x, "x"), (w, "weight"), (b, "bias")):
            assert tensor.grad is not None, f"missing grad for {name}"
            assert torch.isfinite(tensor.grad.float()).all(), f"non-finite grad for {name}"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_gradient_batch_invariance(self, dtype):
        op = NativeAttnOutBiasGemmOp()
        x, w, b = _make_inputs(6, _DIM, dtype)
        x_full = x.clone().requires_grad_(True)
        out_full = op(x_full, w, bias=b)
        grad_full = torch.randn_like(out_full)
        out_full.backward(grad_full)

        x_slice = x[:2].clone().requires_grad_(True)
        out_slice = op(x_slice, w, bias=b)
        out_slice.backward(grad_full[:2])
        assert_bitwise_equal(x_slice.grad, x_full.grad[:2], "dx row-slice invariance")


try:
    from rl_engine.kernels.ops.triton.linear.attn_out_bias_gemm import TritonAttnOutBiasGemmOp

    _HAS_TRITON_OP = True
except Exception:  # pragma: no cover - triton or CUDA missing
    _HAS_TRITON_OP = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON_OP and torch.cuda.is_available()),
    reason="Triton backend requires triton and a CUDA device",
)


@requires_triton_cuda
class TestTritonBackendBitwise:
    """Star-shaped acceptance: triton vs the FP32 CPU same-tree reference.

    Both dtypes byte-for-byte: one correctly-rounded FMA discipline on both
    sides (kernels use fma.rn; the reference uses torch.addcmul, an
    oracle-verified FMA). No tolerance path.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("rows", [1, 2, 7, 33, 64])
    def test_forward_vs_reference(self, dtype, rows):
        reference = NativeAttnOutBiasGemmOp()
        backend = TritonAttnOutBiasGemmOp()
        x, w, b = _make_inputs(rows, _DIM, dtype)
        out_cpu = reference(x, w, bias=b)
        out_gpu = backend(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(out_gpu, out_cpu, f"triton vs reference rows={rows}")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward_vs_reference(self, dtype):
        reference = NativeAttnOutBiasGemmOp()
        backend = TritonAttnOutBiasGemmOp()
        x, w, b = _make_inputs(5, _DIM, dtype)
        # one fixed upstream grad, reused by both runs (regenerating per run
        # would compare gradients of two different losses)
        g = torch.randn(5, _DIM, generator=torch.Generator().manual_seed(99)).to(dtype)

        def grads_on(op, device):
            xd = x.detach().clone().to(device).requires_grad_(True)
            wd = w.detach().clone().to(device).requires_grad_(True)
            bd = b.detach().clone().to(device).requires_grad_(True)
            out = op(xd, wd, bias=bd)
            out.backward(g.to(device))
            return xd.grad, wd.grad, bd.grad

        grads_ref = grads_on(reference, "cpu")
        grads_tri = grads_on(backend, "cuda")
        for ref, tri, name in zip(grads_ref, grads_tri, ("dx", "dW", "db")):
            assert_bitwise_equal(tri.cpu(), ref, f"triton {name} vs reference")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_invariance_on_gpu(self, dtype):
        backend = TritonAttnOutBiasGemmOp()
        x, w, b = _make_inputs(9, _DIM, dtype)
        full = backend(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(
            backend(x[:3].cuda(), w.cuda(), bias=b.cuda()).cpu(), full[:3], "triton slice"
        )
        pad = torch.randn(5, _DIM).to(dtype)
        padded = torch.cat([x, pad], dim=0)
        assert_bitwise_equal(
            backend(padded.cuda(), w.cuda(), bias=b.cuda()).cpu()[:9], full, "triton padded"
        )


try:
    from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

    _HAS_CUDA_OP = True
except Exception:  # pragma: no cover - extension not built
    _HAS_CUDA_OP = False

requires_cuda_ext = pytest.mark.skipif(
    not (_HAS_CUDA_OP and torch.cuda.is_available()),
    reason="CUDA backend requires the compiled extension and a CUDA device",
)


@requires_cuda_ext
class TestCudaBackendBitwise:
    """CUDA backend vs the FP32 CPU reference (and vs the Triton backend).

    Both dtypes byte-for-byte; see the note on TestTritonBackendBitwise.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("rows", [1, 2, 7, 33, 64])
    def test_forward_vs_reference(self, dtype, rows):
        reference = NativeAttnOutBiasGemmOp()
        backend = CudaAttnOutBiasGemmOp()
        x, w, b = _make_inputs(rows, _DIM, dtype)
        out_cpu = reference(x, w, bias=b)
        out_gpu = backend(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(out_gpu, out_cpu, f"cuda vs reference rows={rows}")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward_vs_reference(self, dtype):
        reference = NativeAttnOutBiasGemmOp()
        backend = CudaAttnOutBiasGemmOp()
        x, w, b = _make_inputs(5, _DIM, dtype)
        g = torch.randn(5, _DIM, generator=torch.Generator().manual_seed(99)).to(dtype)

        def grads_on(op, device):
            xd = x.detach().clone().to(device).requires_grad_(True)
            wd = w.detach().clone().to(device).requires_grad_(True)
            bd = b.detach().clone().to(device).requires_grad_(True)
            out = op(xd, wd, bias=bd)
            out.backward(g.to(device))
            return xd.grad, wd.grad, bd.grad

        grads_ref = grads_on(reference, "cpu")
        grads_cuda = grads_on(backend, "cuda")
        for ref, cur, name in zip(grads_ref, grads_cuda, ("dx", "dW", "db")):
            assert_bitwise_equal(cur.cpu(), ref, f"cuda {name} vs reference")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.skipif(
        not (_HAS_TRITON_OP and torch.cuda.is_available()),
        reason="cross-backend needs triton",
    )
    def test_cuda_vs_triton_bitwise(self, dtype):
        # Same FMA discipline on both device backends: byte-for-byte in BOTH
        # dtypes, including fp32 inputs (this is the fp32 strict bar).
        cuda_op = CudaAttnOutBiasGemmOp()
        triton_op = TritonAttnOutBiasGemmOp()
        x, w, b = _make_inputs(9, _DIM, dtype)
        out_cuda = cuda_op(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        out_triton = triton_op(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(out_cuda, out_triton, f"cuda vs triton {dtype}")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_invariance_on_gpu(self, dtype):
        backend = CudaAttnOutBiasGemmOp()
        x, w, b = _make_inputs(9, _DIM, dtype)
        full = backend(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(
            backend(x[:3].cuda(), w.cuda(), bias=b.cuda()).cpu(), full[:3], "cuda slice"
        )
        pad = torch.randn(5, _DIM).to(dtype)
        padded = torch.cat([x, pad], dim=0)
        assert_bitwise_equal(
            backend(padded.cuda(), w.cuda(), bias=b.cuda()).cpu()[:9], full, "cuda padded"
        )


class TestValidation:
    def test_weight_dim_mismatch(self):
        op = NativeAttnOutBiasGemmOp()
        x = torch.randn(2, 8)
        w = torch.randn(4, 7)
        with pytest.raises(ValueError, match="must match weight input dim"):
            op(x, w)

    def test_bias_size_mismatch(self):
        op = NativeAttnOutBiasGemmOp()
        x = torch.randn(2, 8)
        w = torch.randn(4, 8)
        b = torch.randn(5)
        with pytest.raises(ValueError, match="bias size"):
            op(x, w, bias=b)

    def test_dtype_mismatch(self):
        op = NativeAttnOutBiasGemmOp()
        x = torch.randn(2, 8, dtype=torch.float32)
        w = torch.randn(4, 8, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="dtypes must match"):
            op(x, w)


def test_registry_dispatches_attn_out_bias_gemm():
    from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp
    from rl_engine.kernels.ops.pytorch.linear.attn_out_bias_gemm import NativeAttnOutBiasGemmOp
    from rl_engine.kernels.ops.triton.linear.attn_out_bias_gemm import TritonAttnOutBiasGemmOp
    from rl_engine.kernels.registry import kernel_registry

    if torch.cuda.is_available():
        op = kernel_registry.get_op("attn_out_bias_gemm", device="cuda")
        assert isinstance(op, TritonAttnOutBiasGemmOp), type(op).__name__
        assert hasattr(op, "forward")
    op = kernel_registry.get_op("attn_out_bias_gemm", device="cpu")
    assert isinstance(op, NativeAttnOutBiasGemmOp), type(op).__name__
    assert hasattr(op, "forward") and hasattr(op, "forward_fp32")
    if torch.cuda.is_available():
        assert CudaAttnOutBiasGemmOp is not None  # symbol import sanity


@pytest.mark.skipif(
    not torch.cuda.is_available() or _HAS_CUDA_OP is False or _HAS_TRITON_OP is False,
    reason="shape sweep needs CUDA with triton and cuda backends",
)
class TestRealDimShapeSweep:
    """Acceptance tiers at the real 3072 dim (issue #386 resolution shapes).

    Large tiers verify the two device backends byte-for-byte against each
    other in BOTH dtypes (same FMA discipline) plus row-slice invariance;
    small tiers additionally check the CPU same-tree reference bitwise.
    """

    _DIM = 3072
    _TIERS = [1, 2, 7, 20, 300, 4096, 6032, 6889]

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("rows", _TIERS)
    def test_tier_cross_backend_bitwise(self, dtype, rows):
        from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

        cuda_op = CudaAttnOutBiasGemmOp()
        triton_op = TritonAttnOutBiasGemmOp()
        gen = torch.Generator().manual_seed(rows * 7 + (1 if dtype == torch.float32 else 0))
        x = torch.randn(rows, self._DIM, generator=gen).to(dtype).cuda()
        w = torch.randn(self._DIM, self._DIM, generator=gen).to(dtype).cuda()
        b = torch.randn(self._DIM, generator=gen).to(dtype).cuda()
        out_cuda = cuda_op(x, w, bias=b).cpu()
        out_triton = triton_op(x, w, bias=b).cpu()
        assert_bitwise_equal(out_cuda, out_triton, f"tier S={rows} {dtype}")
        # row-slice invariance at the real dim, on the product (triton) path
        if rows >= 2:
            head = triton_op(x[:1], w, bias=b).cpu()
            assert_bitwise_equal(head, out_triton[:1], f"tier slice S={rows}")

    @pytest.mark.parametrize("rows", [1, 7, 300])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_small_tier_vs_reference(self, dtype, rows):
        from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

        reference = NativeAttnOutBiasGemmOp()
        cuda_op = CudaAttnOutBiasGemmOp()
        gen = torch.Generator().manual_seed(rows * 13 + 5)
        x = torch.randn(rows, self._DIM, generator=gen).to(dtype)
        w = torch.randn(self._DIM, self._DIM, generator=gen).to(dtype)
        b = torch.randn(self._DIM, generator=gen).to(dtype)
        out_cpu = reference(x, w, bias=b)
        out_gpu = cuda_op(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(out_gpu, out_cpu, f"real-dim vs reference S={rows}")


@pytest.mark.skipif(
    not torch.cuda.is_available() or _HAS_CUDA_OP is False or _HAS_TRITON_OP is False,
    reason="stream coverage needs CUDA with both device backends",
)
def test_both_stream_weight_sets():
    """Issue acceptance: both stream weight sets (to_out / to_add_out).

    The MMDiT projects the image stream and the text stream through separate
    weight sets with the same operator. Exercise both with independently
    drawn weights and stream-typical row counts (image ~1024^2 tier, text
    ~padded prompt), byte-for-byte across the two device backends.
    """
    from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

    cuda_op = CudaAttnOutBiasGemmOp()
    triton_op = TritonAttnOutBiasGemmOp()
    gen = torch.Generator().manual_seed(77)
    stream_specs = {"image(to_out)": 4096, "text(to_add_out)": 300}
    for stream, rows in stream_specs.items():
        x = torch.randn(rows, 3072, generator=gen).bfloat16().cuda()
        w = torch.randn(3072, 3072, generator=gen).bfloat16().cuda()
        b = torch.randn(3072, generator=gen).bfloat16().cuda()
        out_cuda = cuda_op(x, w, bias=b).cpu()
        out_triton = triton_op(x, w, bias=b).cpu()
        assert_bitwise_equal(out_cuda, out_triton, f"both-streams {stream}")


class TestBiasNoneEdge:
    """Op-specific edge (the analogue of #204's ignore-index class): bias=None."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_reference_bias_none_forward(self, dtype):
        op = NativeAttnOutBiasGemmOp()
        x, w, _ = _make_inputs(5, _DIM, dtype)
        out = op(x, w)  # bias omitted
        gold = op.forward_fp32(x, w).to(dtype)
        assert_bitwise_equal(out, gold, "reference bias=None")

    @pytest.mark.skipif(
        not torch.cuda.is_available() or _HAS_TRITON_OP is False or _HAS_CUDA_OP is False,
        reason="bias=None backend check needs CUDA with both device backends",
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backends_bias_none_bitwise(self, dtype):
        from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

        reference = NativeAttnOutBiasGemmOp()
        x, w, _ = _make_inputs(6, _DIM, dtype)
        out_cpu = reference(x, w)
        out_tri = TritonAttnOutBiasGemmOp()(x.cuda(), w.cuda()).cpu()
        out_cuda = CudaAttnOutBiasGemmOp()(x.cuda(), w.cuda()).cpu()
        assert_bitwise_equal(out_tri, out_cpu, "triton bias=None vs reference")
        assert_bitwise_equal(out_cuda, out_cpu, "cuda bias=None vs reference")
        # the two device backends share the FMA discipline: bitwise in both dtypes
        assert_bitwise_equal(out_cuda, out_tri, f"bias=None cross-backend {dtype}")

    @pytest.mark.skipif(
        not torch.cuda.is_available() or _HAS_TRITON_OP is False or _HAS_CUDA_OP is False,
        reason="bias=None backward needs CUDA with both device backends",
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward_bias_none_grads(self, dtype):
        from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

        g = torch.randn(4, _DIM, generator=torch.Generator().manual_seed(21)).to(dtype)
        for op, device in ((TritonAttnOutBiasGemmOp(), "cuda"), (CudaAttnOutBiasGemmOp(), "cuda")):
            x, w, _ = _make_inputs(4, _DIM, dtype)
            xd = x.to(device).requires_grad_(True)
            wd = w.to(device).requires_grad_(True)
            out = op(xd, wd)
            out.backward(g.to(device))
            assert xd.grad is not None and wd.grad is not None
            assert torch.isfinite(xd.grad.float()).all() and torch.isfinite(wd.grad.float()).all()
            assert wd.grad is not None
            del xd, wd


@requires_triton_cuda
class TestBackendEdges:
    """#204-template categories: fail-closed handling + deterministic repeat."""

    @pytest.mark.parametrize("backend", ["triton", "cuda"])
    def test_rejects_fp16_fail_closed(self, backend):
        # Unsupported strict configurations fail closed (issue #386); the
        # analogue of #204's fallback class is a loud, typed rejection.
        if backend == "triton":
            op = TritonAttnOutBiasGemmOp()
        else:
            from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

            op = CudaAttnOutBiasGemmOp()
        x = torch.randn(2, _DIM, dtype=torch.float16).cuda()
        w = torch.randn(_DIM, _DIM, dtype=torch.float16).cuda()
        with pytest.raises((ValueError, TypeError)):
            op(x, w)

    @pytest.mark.parametrize("backend", ["triton", "cuda"])
    def test_rejects_cpu_tensors(self, backend):
        if backend == "triton":
            op = TritonAttnOutBiasGemmOp()
        else:
            from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

            op = CudaAttnOutBiasGemmOp()
        x, w, b = _make_inputs(2, _DIM, torch.bfloat16)
        with pytest.raises(ValueError):
            op(x, w, bias=b)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", ["triton", "cuda"])
    def test_deterministic_repeat_gpu(self, dtype, backend):
        if backend == "triton":
            op = TritonAttnOutBiasGemmOp()
        else:
            from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

            op = CudaAttnOutBiasGemmOp()
        x, w, b = _make_inputs(7, _DIM, dtype)
        first = op(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        second = op(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(first, second, f"{backend} repeat {dtype}")


@pytest.mark.skipif(
    not torch.cuda.is_available() or _HAS_TRITON_OP is False or _HAS_CUDA_OP is False,
    reason="non-contiguous gradient regression needs CUDA with both device backends",
)
class TestReviewRegressions:
    """Regressions requested by the PR #459 review."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_expanded_upstream_gradient_backward(self, dtype):
        # out.sum().backward() hands the op an EXPANDED (non-contiguous) grad:
        # `torch.ones` broadcast to the output shape. The host side must
        # materialise it before the C++ contiguity checks (review comment on
        # the CUDA backward path).
        reference = NativeAttnOutBiasGemmOp()
        backends = (TritonAttnOutBiasGemmOp(), CudaAttnOutBiasGemmOp())
        x, w, b = _make_inputs(5, _DIM, dtype)
        g = torch.randn(5, _DIM, generator=torch.Generator().manual_seed(31)).to(dtype)

        def grads_sum(op, device):
            xd = x.detach().clone().to(device).requires_grad_(True)
            wd = w.detach().clone().to(device).requires_grad_(True)
            bd = b.detach().clone().to(device).requires_grad_(True)
            op(xd, wd, bias=bd).sum().backward()  # expanded ones gradient
            return [t.grad.cpu() for t in (xd, wd, bd)]

        def grads_with_g(op, device):
            xd = x.detach().clone().to(device).requires_grad_(True)
            wd = w.detach().clone().to(device).requires_grad_(True)
            bd = b.detach().clone().to(device).requires_grad_(True)
            op(xd, wd, bias=bd).backward(g.to(device))
            return [t.grad.cpu() for t in (xd, wd, bd)]

        ref_sum = grads_sum(reference, "cpu")
        for backend in backends:
            outs = grads_sum(backend, "cuda")
            for got, ref, name in zip(outs, ref_sum, ("dx", "dW", "db")):
                assert_bitwise_equal(got, ref, f"sum-grad {name} {type(backend).__name__}")
            # sanity: a random dY must of course differ from the all-ones dY
            outs_g = grads_with_g(backend, "cuda")
            assert not torch.equal(outs_g[0], outs[0])

    def test_reference_is_fma_not_separate(self):
        # Two-element reduction where separate mul+add and fused FMA differ:
        # step 0 sets acc = fl(a0*b0) = -fl(a1*b1); step 1 then gives
        # exactly +0.0 under separate rounding, but a nonzero residual under
        # a correctly-rounded FMA. Guards against the reference (or a kernel)
        # silently reverting to separate discipline.
        import struct

        def f32(x):
            return struct.unpack("<f", struct.pack("<f", x))[0]

        a0, b0 = f32(0.1), f32(-0.2)
        a1, b1 = f32(0.1), f32(0.2)
        acc = f32(a0 * b0)  # first step from +0.0 rounds identically either way
        separate = f32(f32(a1 * b1) + acc)  # == 0.0 exactly
        fused = f32(a1 * b1 + acc)  # python double is exact for these products
        assert separate == 0.0 and fused != 0.0, "counterexample construction failed"

        a = torch.tensor([[a0, a1]])
        b = torch.tensor([[b0, b1]])
        out = tree_gemm_fp32(a, b)
        assert out.item() != 0.0, "reference computed the SEPARATE result"
        assert out.item() == fused, "reference FMA emulation is not correctly rounded"

    @pytest.mark.parametrize("backend", ["triton", "cuda"])
    def test_rejects_bias_on_wrong_device(self, backend):
        if backend == "triton":
            op = TritonAttnOutBiasGemmOp()
        else:
            from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

            op = CudaAttnOutBiasGemmOp()
        x, w, b = _make_inputs(2, _DIM, torch.bfloat16)
        with pytest.raises(ValueError):
            op(x.cuda(), w.cuda(), bias=b)  # bias stays on CPU


@pytest.mark.skipif(
    not torch.cuda.is_available() or _HAS_TRITON_OP is False or _HAS_CUDA_OP is False,
    reason="edge shapes need CUDA with both device backends",
)
class TestEdgeShapes:
    """Robustness edges: zero rows and odd (short-tail-leaf) dims on device."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", ["triton", "cuda"])
    def test_zero_rows(self, dtype, backend):
        if backend == "triton":
            op = TritonAttnOutBiasGemmOp()
        else:
            from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

            op = CudaAttnOutBiasGemmOp()
        x = torch.zeros(0, _DIM).to(dtype).cuda()
        w = torch.randn(_DIM, _DIM).to(dtype).cuda()
        b = torch.randn(_DIM).to(dtype).cuda()
        out = op(x, w, bias=b)
        assert out.shape == (0, _DIM)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", ["triton", "cuda"])
    def test_odd_dim_bitwise_vs_reference(self, dtype, backend):
        # R=33 exercises a short tail leaf (one 32-wide + one 1-wide) on the
        # device kernels; must stay bitwise vs the same-tree reference.
        reference = NativeAttnOutBiasGemmOp()
        if backend == "triton":
            op = TritonAttnOutBiasGemmOp()
        else:
            from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

            op = CudaAttnOutBiasGemmOp()
        gen = torch.Generator().manual_seed(41)
        x = torch.randn(3, 33, generator=gen).to(dtype)
        w = torch.randn(5, 33, generator=gen).to(dtype)
        b = torch.randn(5, generator=gen).to(dtype)
        out_cpu = reference(x, w, bias=b)
        out_gpu = op(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(out_gpu, out_cpu, f"odd-dim {backend} {dtype}")


# ---------------------------------------------------------------------------
# Independent spec implementation (mandatory-acceptance rule: an independent
# reference must independently implement the SAME frozen FP semantics; whole-
# formula fp64 results are NOT a bitwise standard). Scalar ``libm fmaf`` is a
# correctly-rounded FP32 FMA by IEEE 754 specification.
# ---------------------------------------------------------------------------
try:
    import ctypes

    _LIBM = ctypes.CDLL("libm.so.6")
    _LIBM.fmaf.restype = ctypes.c_float
    _LIBM.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    _HAS_FMAF = True
except Exception:  # pragma: no cover
    _HAS_FMAF = False

_requires_fmaf = pytest.mark.skipif(not _HAS_FMAF, reason="libm fmaf unavailable")


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _spec_leaf(a_vec, b_vec, k0, k1):
    acc = 0.0
    for k in range(k0, k1):
        acc = _LIBM.fmaf(a_vec[k], b_vec[k], acc)
    return acc


def _spec_tree(a_vec, b_vec, red, lo, hi):
    if hi - lo == 1:
        k0 = lo * 32
        return _spec_leaf(a_vec, b_vec, k0, min(k0 + 32, red))
    mid = lo + (hi - lo) // 2
    return _f32(_spec_tree(a_vec, b_vec, red, lo, mid) + _spec_tree(a_vec, b_vec, red, mid, hi))


def _spec_forward_row(x_row, w_rows, bias_row, dtype):
    """Independent frozen-semantics forward for one row (returns fp32 list)."""
    red = len(x_row)
    leaves = (red + 31) // 32
    out = []
    for n, w_row in enumerate(w_rows):
        tree = _spec_tree(x_row, w_row, red, 0, leaves)
        v = _f32(tree + bias_row[n]) if bias_row is not None else tree
        out.append(_f32(v))
    return out


@_requires_fmaf
class TestFmaPrimitive:
    """Rule gate 1: the reference primitive must BE a correctly-rounded FMA."""

    def test_addcmul_passes_reported_midband_counterexample(self):
        # Regression for the externally reported counterexample that disproved
        # the old fp64-emulation claim: a = b = 1+2^-12, c = 2^-60. The exact
        # a*b + c sits strictly above the fp32 midpoint 1+2^-11+2^-24, so the
        # correctly-rounded FMA rounds UP; the fp64-relay path rounds the
        # intermediate onto the midpoint and ties-to-even rounds DOWN.
        a = _f32(1.0 + 2**-12)
        b = _f32(1.0 + 2**-12)
        c = _f32(2.0**-60)
        acc = torch.tensor([c], dtype=torch.float32)
        got = torch.addcmul(acc, torch.tensor([a]), torch.tensor([b])).item()
        oracle = _LIBM.fmaf(a, b, c)
        assert struct.unpack("<I", struct.pack("<f", got))[0] == 0x3F801001
        assert struct.unpack("<I", struct.pack("<f", oracle))[0] == 0x3F801001
        # and the refuted path, kept as a negative witness
        refuted = _f32(_f32(a * b) + c)
        assert struct.unpack("<I", struct.pack("<f", refuted))[0] == 0x3F801000

    def test_addcmul_full_chain_matches_fmaf(self):
        # Full leaf-chain regression: a 2^-60 seed from the first FMA must be
        # carried exactly through the second FMA (no intermediate rounding).
        a = _f32(1.0 + 2**-12)
        seed = _LIBM.fmaf(_f32(2.0**-30), _f32(2.0**-30), 0.0)
        assert struct.unpack("<I", struct.pack("<f", seed))[0] == 0x21800000  # 2^-60
        acc = torch.tensor([seed], dtype=torch.float32)
        got = torch.addcmul(acc, torch.tensor([a]), torch.tensor([a])).item()
        expect = _LIBM.fmaf(a, a, seed)
        assert _f32(got) == _f32(expect)
        assert struct.unpack("<I", struct.pack("<f", expect))[0] == 0x3F801001

    def test_addcmul_passes_double_rounding_counterexample(self):
        # separate mul+add gives exactly 0.0; a true FMA leaves a residual.
        a, b = _f32(0.1), _f32(0.2)
        c = _f32(-_f32(a * b))
        acc = torch.full((1,), c, dtype=torch.float32)
        got = torch.addcmul(acc, torch.tensor([a]), torch.tensor([b])).item()
        oracle = _LIBM.fmaf(a, b, c)
        assert _f32(got) == _f32(oracle) and _f32(oracle) != 0.0

    @pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    def test_addcmul_matches_fmaf_oracle_random(self, device):
        gen = torch.Generator().manual_seed(42)
        n = 4096
        A = torch.randn(n, generator=gen).abs() * 1e4
        B = torch.randn(n, generator=gen)
        C = torch.randn(n, generator=gen) * 1e3
        r = torch.addcmul(C.to(device), A.to(device), B.to(device)).cpu()
        for i in range(n):
            assert _f32(_LIBM.fmaf(A[i].item(), B[i].item(), C[i].item())) == r[i].item(), i

    @pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    def test_addcmul_matches_fmaf_oracle_adversarial(self, device):
        # construct exact a*b + c landing on fp32 midpoints (the double-
        # rounding danger band)
        gen = torch.Generator().manual_seed(43)
        n = 2048
        A = torch.randn(n, generator=gen)
        B = torch.randn(n, generator=gen)
        rows = []
        for i in range(n):
            a, b = A[i].item(), B[i].item()
            p = a * b
            e = int(np.floor(np.log2(abs(p)))) if p != 0 else 0
            mid = (2 * int(torch.randint(1, 1 << 20, (1,), generator=gen)) + 1) * 2.0 ** (e - 24)
            c = _f32(mid - p)
            if c == 0.0 and mid - p != 0:
                continue
            rows.append((a, b, c))
        acc = torch.tensor([r[2] for r in rows], dtype=torch.float32)
        r = torch.addcmul(
            acc.to(device),
            torch.tensor([r[0] for r in rows]).to(device),
            torch.tensor([r[1] for r in rows]).to(device),
        ).cpu()
        for i, (a, b, c) in enumerate(rows):
            assert _f32(_LIBM.fmaf(a, b, c)) == r[i].item(), (a, b, c)


@_requires_fmaf
class TestIndependentSpecBitwise:
    """Rule 7: independent implementation of the frozen FP semantics, bitwise
    against every backend. No fp64 whole-formula comparisons, no tolerances."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_vs_independent_spec(self, dtype):
        torch.manual_seed(5)
        rows, dim = 4, 40  # includes a short tail leaf (40 = 32 + 8)
        x = torch.randn(rows, dim).to(dtype)
        w = torch.randn(5, dim).to(dtype)
        b = torch.randn(5).to(dtype)
        xf, wf, bf = x.float(), w.float(), b.float()
        spec = torch.tensor(
            [
                _spec_forward_row(xf[s].tolist(), wf.tolist(), bf.tolist(), dtype)
                for s in range(rows)
            ],
            dtype=torch.float32,
        ).to(dtype)
        assert_bitwise_equal(NativeAttnOutBiasGemmOp()(x, w, bias=b), spec, "reference vs spec")
        if torch.cuda.is_available() and _HAS_TRITON_OP:
            out = TritonAttnOutBiasGemmOp()(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
            assert_bitwise_equal(out, spec, "triton vs spec")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward_vs_independent_spec(self, dtype):
        from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

        torch.manual_seed(6)
        rows, dim = 3, 40
        x = torch.randn(rows, dim).to(dtype)
        w = torch.randn(6, dim).to(dtype)
        b = torch.randn(6).to(dtype)
        dY = torch.randn(rows, 6).to(dtype)
        xf, wf, dYf = x.float(), w.float(), dY.float()
        leaves = (6 + 31) // 32  # dx reduction over N=6

        # dx: per contract, tree over n in [0, N) with operand pairs (dY[s,n], W[n,k])
        dx_spec = torch.zeros(rows, dim, dtype=torch.float32)
        for s in range(rows):
            for k in range(dim):
                dx_spec[s, k] = _spec_tree(dYf[s].tolist(), wf[:, k].tolist(), 6, 0, leaves)
        # dW: ascending-row left fold, one fmaf per row
        dw_spec = torch.zeros(6, dim, dtype=torch.float32)
        for n in range(6):
            for k in range(dim):
                acc = 0.0
                for s_ in range(rows):
                    acc = _LIBM.fmaf(dYf[s_, n].item(), xf[s_, k].item(), acc)
                dw_spec[n, k] = acc
        # db: ascending add fold
        db_spec = torch.zeros(6, dtype=torch.float32)
        for n in range(6):
            acc = 0.0
            for s_ in range(rows):
                acc = _f32(acc + dYf[s_, n].item())
            db_spec[n] = acc

        for op, dev, name in (
            (NativeAttnOutBiasGemmOp(), "cpu", "reference"),
            (TritonAttnOutBiasGemmOp(), "cuda", "triton"),
            (CudaAttnOutBiasGemmOp(), "cuda", "cuda"),
        ):
            if dev == "cuda" and not torch.cuda.is_available():
                continue
            xd = x.detach().clone().to(dev).requires_grad_(True)
            wd = w.detach().clone().to(dev).requires_grad_(True)
            bd = b.detach().clone().to(dev).requires_grad_(True)
            op(xd, wd, bias=bd).backward(dY.to(dev))
            # gradients are cast once to the input/param dtype at return
            assert_bitwise_equal(xd.grad.cpu(), dx_spec.to(dtype), f"{name} dx vs spec")
            assert_bitwise_equal(wd.grad.cpu(), dw_spec.to(dtype), f"{name} dW vs spec")
            assert_bitwise_equal(bd.grad.cpu(), db_spec.to(dtype), f"{name} db vs spec")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.skipif(
        not torch.cuda.is_available() or _HAS_TRITON_OP is False,
        reason="device backends needed",
    )
    def test_parameter_gradients_bitwise_deterministic(self, dtype):
        # Mandatory rule: parameter gradients are bitwise-identical when the
        # full input, the row order and dY are fixed (determinism scope).
        from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

        x, w, b = _make_inputs(6, _DIM, dtype)
        dY = torch.randn(6, _DIM).to(dtype)
        for op in (TritonAttnOutBiasGemmOp(), CudaAttnOutBiasGemmOp()):
            grads = []
            for _ in range(2):
                xd = x.detach().clone().cuda().requires_grad_(True)
                wd = w.detach().clone().cuda().requires_grad_(True)
                bd = b.detach().clone().cuda().requires_grad_(True)
                op(xd, wd, bias=bd).backward(dY.cuda())
                grads.append((xd.grad, wd.grad, bd.grad))
            for first, second, name in zip(grads[0], grads[1], ("dx", "dW", "db")):
                assert_bitwise_equal(
                    first.cpu(), second.cpu(), f"{type(op).__name__} {name} repeat"
                )
