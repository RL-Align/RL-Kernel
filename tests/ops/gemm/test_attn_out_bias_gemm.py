# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Bit-equality harness and tests for the Qwen-Image attn-out bias GEMM.

The harness implements the frozen comparison procedure from the contract
(docs/operators/attn-out-bias-gemm.md):

    1. assert shapes match (explicit, for a locatable failure message)
    2. assert dtypes match
    3. require all elements of both results to be finite
    4. materialise contiguous copies (a copy never changes logical element bits;
       column slices / transposes need it for the dtype reinterpretation)
    5. compare the *bit patterns of the logical elements* via a dtype bitcast
       (int16 for bf16, int32 for fp32) -- not the underlying allocation bytes

The raw integer comparator distinguishes signed zeros and NaN payloads.
The acceptance assertion rejects any non-finite result before comparing bits.

The small reference cases compare device results with CPU results. The
3072-wide sweep compares CUDA with Triton at every tier and CUDA with the
CPU reference at 1, 7 and 300 rows. Invariance checks compare rows from
different executions of the same backend.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from rl_engine.reference.gemm.attn_out_bias_gemm import (
    NativeAttnOutBiasGemmOp,
    _fma_fp32,
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
    """Compare raw logical-element bits, including signed zeros and NaN payloads."""
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
    assert torch.isfinite(a).all(), f"{prefix}non-finite first tensor"
    assert torch.isfinite(b).all(), f"{prefix}non-finite second tensor"
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
        # same correctly-rounded FMA chains.
        gen = torch.Generator().manual_seed(7)
        a = torch.randn(2, 33, generator=gen)
        b = torch.randn(3, 33, generator=gen)

        def fma_leaf(a_cols, b_cols):
            acc = torch.zeros(2, 3)
            for k in range(a_cols.size(1)):
                acc = _fma_fp32(acc, a_cols[:, k].unsqueeze(1), b_cols[:, k].unsqueeze(0))
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
    from rl_engine.backends.shared.triton.gemm.attn_out_bias_gemm import (
        _TRITON_AVAILABLE,
        TritonAttnOutBiasGemmOp,
    )

    _HAS_TRITON_OP = _TRITON_AVAILABLE
except Exception:  # pragma: no cover - triton or CUDA missing
    _HAS_TRITON_OP = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON_OP and torch.cuda.is_available()),
    reason="Triton backend requires triton and a CUDA device",
)


@requires_triton_cuda
class TestTritonBackendBitwise:
    """Compare Triton results with the FP32 CPU reference in both dtypes."""

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
    from rl_engine.backends.cuda.gemm.attn_out_bias_gemm import (
        CudaAttnOutBiasGemmOp,
        _cuda_backend_available,
    )

    _HAS_CUDA_OP = _cuda_backend_available()
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
    from rl_engine.backends.cuda.gemm.attn_out_bias_gemm import CudaAttnOutBiasGemmOp
    from rl_engine.backends.shared.triton.gemm.attn_out_bias_gemm import TritonAttnOutBiasGemmOp
    from rl_engine.reference.gemm.attn_out_bias_gemm import NativeAttnOutBiasGemmOp
    from rl_engine.runtime.registry import kernel_registry

    if torch.version.cuda is not None and torch.cuda.is_available():
        op = kernel_registry.get_op("attn_out_bias_gemm", device="cuda")
        if _HAS_TRITON_OP:
            expected = TritonAttnOutBiasGemmOp
        elif _HAS_CUDA_OP:
            expected = CudaAttnOutBiasGemmOp
        else:
            expected = NativeAttnOutBiasGemmOp
        assert isinstance(op, expected), type(op).__name__
        assert op.trace["fallback"] == (not _HAS_TRITON_OP)
        assert hasattr(op, "forward")
    op = kernel_registry.get_op("attn_out_bias_gemm", device="cpu")
    assert isinstance(op, NativeAttnOutBiasGemmOp), type(op).__name__
    assert hasattr(op, "forward") and hasattr(op, "forward_fp32")


class TestRealDimShapeSweep:
    """Acceptance tiers at the real 3072 dim (issue #386 resolution shapes).

    Large tiers verify the two device backends byte-for-byte against each
    other in BOTH dtypes (same FMA discipline) plus row-slice invariance;
    small tiers additionally check the CPU same-tree reference bitwise.
    """

    _DIM = 3072
    _TIERS = [1, 2, 7, 20, 300, 4096, 6032, 6889]

    def _make_tier_inputs(self, dtype, rows):
        gen = torch.Generator().manual_seed(rows * 7 + (1 if dtype == torch.float32 else 0))
        x = torch.randn(rows, self._DIM, generator=gen).to(dtype).cuda()
        w = torch.randn(self._DIM, self._DIM, generator=gen).to(dtype).cuda()
        b = torch.randn(self._DIM, generator=gen).to(dtype).cuda()
        return x, w, b

    @requires_cuda_ext
    @requires_triton_cuda
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("rows", _TIERS)
    def test_tier_cross_backend_bitwise(self, dtype, rows):
        from rl_engine.backends.cuda.gemm.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

        cuda_op = CudaAttnOutBiasGemmOp()
        triton_op = TritonAttnOutBiasGemmOp()
        x, w, b = self._make_tier_inputs(dtype, rows)
        out_cuda = cuda_op(x, w, bias=b).cpu()
        out_triton = triton_op(x, w, bias=b).cpu()
        assert_bitwise_equal(out_cuda, out_triton, f"tier S={rows} {dtype}")

    @requires_triton_cuda
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("rows", _TIERS[1:])
    def test_triton_row_slice_invariance(self, dtype, rows):
        triton_op = TritonAttnOutBiasGemmOp()
        x, w, b = self._make_tier_inputs(dtype, rows)
        full = triton_op(x, w, bias=b).cpu()
        head = triton_op(x[:1], w, bias=b).cpu()
        assert_bitwise_equal(head, full[:1], f"tier slice S={rows}")

    @requires_cuda_ext
    @pytest.mark.parametrize("rows", [1, 7, 300])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_small_tier_vs_reference(self, dtype, rows):
        from rl_engine.backends.cuda.gemm.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

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
    from rl_engine.backends.cuda.gemm.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

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

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backend_bias_none_vs_reference(self, dtype, gpu_op):
        x, w, _ = _make_inputs(6, _DIM, dtype)
        expected = NativeAttnOutBiasGemmOp()(x, w)
        actual = gpu_op(x.cuda(), w.cuda()).cpu()
        assert_bitwise_equal(actual, expected, f"{gpu_op.backend_id} bias=None vs reference")

    @pytest.mark.skipif(
        not torch.cuda.is_available() or _HAS_TRITON_OP is False or _HAS_CUDA_OP is False,
        reason="bias=None backend check needs CUDA with both device backends",
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backends_bias_none_bitwise(self, dtype):
        from rl_engine.backends.cuda.gemm.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

        x, w, _ = _make_inputs(6, _DIM, dtype)
        out_tri = TritonAttnOutBiasGemmOp()(x.cuda(), w.cuda()).cpu()
        out_cuda = CudaAttnOutBiasGemmOp()(x.cuda(), w.cuda()).cpu()
        assert_bitwise_equal(out_cuda, out_tri, f"bias=None cross-backend {dtype}")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward_bias_none_grads(self, dtype, gpu_op):
        g = torch.randn(4, _DIM, generator=torch.Generator().manual_seed(21)).to(dtype)
        x, w, _ = _make_inputs(4, _DIM, dtype)
        xd = x.cuda().requires_grad_(True)
        wd = w.cuda().requires_grad_(True)
        out = gpu_op(xd, wd)
        out.backward(g.cuda())
        assert xd.grad is not None and wd.grad is not None
        assert torch.isfinite(xd.grad.float()).all() and torch.isfinite(wd.grad.float()).all()


@pytest.fixture(
    params=[
        pytest.param("triton", marks=requires_triton_cuda),
        pytest.param("cuda", marks=requires_cuda_ext),
    ]
)
def gpu_op(request):
    if request.param == "triton":
        return TritonAttnOutBiasGemmOp()
    return CudaAttnOutBiasGemmOp()


class TestBackendEdges:
    """#204-template categories: fail-closed handling + deterministic repeat."""

    def test_rejects_fp16_fail_closed(self, gpu_op):
        # Unsupported strict configurations fail closed (issue #386); the
        # analogue of #204's fallback class is a loud, typed rejection.
        x = torch.randn(2, _DIM, dtype=torch.float16).cuda()
        w = torch.randn(_DIM, _DIM, dtype=torch.float16).cuda()
        with pytest.raises((ValueError, TypeError)):
            gpu_op(x, w)

    def test_rejects_cpu_tensors(self, gpu_op):
        x, w, b = _make_inputs(2, _DIM, torch.bfloat16)
        with pytest.raises(ValueError):
            gpu_op(x, w, bias=b)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_deterministic_repeat_gpu(self, dtype, gpu_op):
        x, w, b = _make_inputs(7, _DIM, dtype)
        first = gpu_op(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        second = gpu_op(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(first, second, f"{gpu_op.backend_id} repeat {dtype}")


class TestReviewRegressions:
    """Regressions requested by the PR #459 review."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_expanded_upstream_gradient_backward(self, dtype, gpu_op):
        # out.sum().backward() hands the op an EXPANDED (non-contiguous) grad:
        # `torch.ones` broadcast to the output shape. The host side must
        # materialise it before the C++ contiguity checks (review comment on
        # the CUDA backward path).
        reference = NativeAttnOutBiasGemmOp()
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
        outs = grads_sum(gpu_op, "cuda")
        for got, ref, name in zip(outs, ref_sum, ("dx", "dW", "db")):
            assert_bitwise_equal(got, ref, f"sum-grad {name} {gpu_op.backend_id}")
        # A random upstream gradient must differ from the expanded all-ones gradient.
        outs_g = grads_with_g(gpu_op, "cuda")
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

    def test_rejects_bias_on_wrong_device(self, gpu_op):
        x, w, b = _make_inputs(2, _DIM, torch.bfloat16)
        with pytest.raises(ValueError):
            gpu_op(x.cuda(), w.cuda(), bias=b)  # bias stays on CPU


class TestEdgeShapes:
    """Robustness edges: zero rows and odd (short-tail-leaf) dims on device."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_zero_rows(self, dtype, gpu_op):
        x = torch.zeros(0, _DIM).to(dtype).cuda()
        w = torch.randn(_DIM, _DIM).to(dtype).cuda()
        b = torch.randn(_DIM).to(dtype).cuda()
        out = gpu_op(x, w, bias=b)
        assert out.shape == (0, _DIM)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_zero_rows_backward(self, dtype, gpu_op):
        # Empty batch: dx is empty and the parameter folds over zero terms
        # are exact zeros -- defined behaviour, identical on every backend.
        x = torch.zeros(0, _DIM, dtype=dtype, device="cuda").requires_grad_(True)
        w = torch.randn(_DIM, _DIM, dtype=dtype, device="cuda").requires_grad_(True)
        b = torch.randn(_DIM, dtype=dtype, device="cuda").requires_grad_(True)
        out = gpu_op(x, w, bias=b)
        out.backward(torch.zeros_like(out))
        assert x.grad is not None and x.grad.numel() == 0
        assert torch.count_nonzero(w.grad).item() == 0
        assert torch.count_nonzero(b.grad).item() == 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_odd_dim_bitwise_vs_reference(self, dtype, gpu_op):
        # R=33 exercises a short tail leaf (one 32-wide + one 1-wide) on the
        # device kernels; must stay bitwise vs the same-tree reference.
        reference = NativeAttnOutBiasGemmOp()
        gen = torch.Generator().manual_seed(41)
        x = torch.randn(3, 33, generator=gen).to(dtype)
        w = torch.randn(5, 33, generator=gen).to(dtype)
        b = torch.randn(5, generator=gen).to(dtype)
        out_cpu = reference(x, w, bias=b)
        out_gpu = gpu_op(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(out_gpu, out_cpu, f"odd-dim {gpu_op.backend_id} {dtype}")


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

    @pytest.mark.parametrize(
        "device", ["cpu"] + (["cuda"] if torch.version.cuda and torch.cuda.is_available() else [])
    )
    def test_fma_boundary_bits(self, device):
        near_one = 1 + 2**-12
        largest = torch.finfo(torch.float32).max
        triples = [
            (sign * near_one, near_one, tail * 2.0**-60) for sign in (-1, 1) for tail in (-1, 1)
        ] + [
            (1 + 2**-23, 1 - 2**-23, -1.0),
            (-0.0, 1.0, -0.0),
            (-0.0, 1.0, +0.0),
            (-(2.0**-126), 2.0**-24, 0.0),
            (2.0**-126, 2.0**-24, 2.0**-149),
            (largest, 2.0, -largest),
            (largest, 1.0, 2.0**103),
        ]
        a, b, c = torch.tensor(triples, dtype=torch.float32).T
        expected = torch.tensor([_LIBM.fmaf(x, y, z) for x, y, z in triples], dtype=torch.float32)
        got = _fma_fp32(c.to(device), a.to(device), b.to(device))
        assert torch.equal(got.cpu().view(torch.int32), expected.view(torch.int32))

    def test_fma_passes_reported_midband_counterexample(self):
        # For a = b = 1+2^-12 the exact product is the FP32 midpoint
        # 1+2^-11+2^-24. Adding 2^-60 before rounding moves the result
        # above the midpoint, so the correctly rounded FMA rounds up.
        a = _f32(1.0 + 2**-12)
        b = _f32(1.0 + 2**-12)
        c = _f32(2.0**-60)
        acc = torch.tensor([c], dtype=torch.float32)
        got = _fma_fp32(acc, torch.tensor([a]), torch.tensor([b])).item()
        oracle = _LIBM.fmaf(a, b, c)
        assert struct.unpack("<I", struct.pack("<f", got))[0] == 0x3F801001
        assert struct.unpack("<I", struct.pack("<f", oracle))[0] == 0x3F801001
        # Rounding the product to FP32 first rounds the tie down.
        refuted = _f32(_f32(a * b) + c)
        assert struct.unpack("<I", struct.pack("<f", refuted))[0] == 0x3F801000

    def test_fma_full_chain_matches_fmaf(self):
        # The first product moves the second FMA just above its FP32 midpoint.
        first = _f32(2.0**-30)
        a = _f32(1.0 + 2**-12)
        row = torch.tensor([[first, a]], dtype=torch.float32)
        got = tree_gemm_fp32(row, row)

        seed = _LIBM.fmaf(first, first, 0.0)
        assert struct.unpack("<I", struct.pack("<f", seed))[0] == 0x21800000  # 2^-60
        expect = _LIBM.fmaf(a, a, seed)
        expected_bits = struct.unpack("<I", struct.pack("<f", expect))[0]
        assert expected_bits == 0x3F801001
        assert got.view(torch.int32).item() == expected_bits

    def test_fma_passes_double_rounding_counterexample(self):
        # separate mul+add gives exactly 0.0; a true FMA leaves a residual.
        a, b = _f32(0.1), _f32(0.2)
        c = _f32(-_f32(a * b))
        acc = torch.full((1,), c, dtype=torch.float32)
        got = _fma_fp32(acc, torch.tensor([a]), torch.tensor([b])).item()
        oracle = _LIBM.fmaf(a, b, c)
        assert _f32(got) == _f32(oracle) and _f32(oracle) != 0.0

    @pytest.mark.parametrize(
        "device", ["cpu"] + (["cuda"] if torch.version.cuda and torch.cuda.is_available() else [])
    )
    def test_fma_matches_fmaf_oracle_random(self, device):
        gen = torch.Generator().manual_seed(42)
        n = 4096
        A = torch.randn(n, generator=gen).abs() * 1e4
        B = torch.randn(n, generator=gen)
        C = torch.randn(n, generator=gen) * 1e3
        r = _fma_fp32(C.to(device), A.to(device), B.to(device)).cpu()
        for i in range(n):
            assert _f32(_LIBM.fmaf(A[i].item(), B[i].item(), C[i].item())) == r[i].item(), i

    @pytest.mark.parametrize(
        "device", ["cpu"] + (["cuda"] if torch.version.cuda and torch.cuda.is_available() else [])
    )
    def test_fma_matches_fmaf_oracle_adversarial(self, device):
        # For odd u and v up to 511, (1 + u/4096) * (1 + v/4096) lies
        # in [1, 2) and is an odd multiple of 2**-24: an exact FP32 midpoint.
        rng = np.random.default_rng(43)
        rows = []
        for left, right in rng.integers(0, 256, size=(256, 2)).tolist():
            exponent = int(rng.integers(-60, 61))
            a = (1 + (2 * left + 1) / 4096) * 2.0**exponent
            b = 1 + (2 * right + 1) / 4096
            tail = 2.0 ** (exponent - 60)
            for sign in (-1, 1):
                rows.extend(((sign * a, b, -tail), (sign * a, b, tail)))

        a, b, c = torch.tensor(rows, dtype=torch.float32).unbind(dim=1)
        expected_bits = torch.tensor([_LIBM.fmaf(*row) for row in rows], dtype=torch.float32).view(
            torch.int32
        )
        # FP64 addition loses either tail. Exactly one of each pair must
        # disagree with the single-rounding oracle after narrowing to FP32.
        via_fp64 = (a.double() * b.double() + c.double()).float()
        assert (via_fp64.view(torch.int32) != expected_bits).sum().item() == len(rows) // 2

        actual = _fma_fp32(c.to(device), a.to(device), b.to(device)).cpu()
        assert torch.equal(actual.view(torch.int32), expected_bits)


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
        if torch.cuda.is_available() and _HAS_CUDA_OP:
            out = CudaAttnOutBiasGemmOp()(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
            assert_bitwise_equal(out, spec, "cuda vs spec")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward_vs_independent_spec(self, dtype):
        from rl_engine.backends.cuda.gemm.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

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

        for op_class, dev, name, available in (
            (NativeAttnOutBiasGemmOp, "cpu", "reference", True),
            (TritonAttnOutBiasGemmOp, "cuda", "triton", _HAS_TRITON_OP),
            (CudaAttnOutBiasGemmOp, "cuda", "cuda", _HAS_CUDA_OP),
        ):
            if dev == "cuda" and (not torch.cuda.is_available() or not available):
                continue
            op = op_class()
            xd = x.detach().clone().to(dev).requires_grad_(True)
            wd = w.detach().clone().to(dev).requires_grad_(True)
            bd = b.detach().clone().to(dev).requires_grad_(True)
            op(xd, wd, bias=bd).backward(dY.to(dev))
            # gradients are cast once to the input/param dtype at return
            assert_bitwise_equal(xd.grad.cpu(), dx_spec.to(dtype), f"{name} dx vs spec")
            assert_bitwise_equal(wd.grad.cpu(), dw_spec.to(dtype), f"{name} dW vs spec")
            assert_bitwise_equal(bd.grad.cpu(), db_spec.to(dtype), f"{name} db vs spec")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_parameter_gradients_bitwise_deterministic(self, dtype, gpu_op):
        # Mandatory rule: parameter gradients are bitwise-identical when the
        # full input, the row order and dY are fixed (determinism scope).
        x, w, b = _make_inputs(6, _DIM, dtype)
        dY = torch.randn(6, _DIM).to(dtype)
        grads = []
        for _ in range(2):
            xd = x.detach().clone().cuda().requires_grad_(True)
            wd = w.detach().clone().cuda().requires_grad_(True)
            bd = b.detach().clone().cuda().requires_grad_(True)
            gpu_op(xd, wd, bias=bd).backward(dY.cuda())
            grads.append((xd.grad, wd.grad, bd.grad))
        for first, second, name in zip(grads[0], grads[1], ("dx", "dW", "db")):
            assert_bitwise_equal(first.cpu(), second.cpu(), f"{gpu_op.backend_id} {name} repeat")


@pytest.mark.parametrize(
    ("dtype", "bits", "payload"),
    [(torch.float32, torch.int32, 0x7FC00001), (torch.bfloat16, torch.int16, 0x7FC1)],
)
def test_strict_acceptance_rejects_even_bitwise_identical_nans(dtype, bits, payload):
    nan = torch.tensor([payload], dtype=bits).view(dtype)
    assert bitwise_equal(nan, nan.clone())
    other = torch.tensor([payload + 1], dtype=bits).view(dtype)
    assert not bitwise_equal(nan, other)
    with pytest.raises(AssertionError, match="non-finite"):
        assert_bitwise_equal(nan, nan.clone())


@pytest.mark.skipif(
    not (_HAS_TRITON_OP and torch.cuda.is_available()), reason="requires Triton GPU"
)
def test_triton_grid_spans_wide_output():
    from rl_engine.backends.shared.triton.gemm.attn_out_bias_gemm import (
        triton_dw_left_fold,
        triton_tree_gemm,
    )

    width = 65535 * 64 + 1
    values = torch.arange(width, dtype=torch.float32, device="cuda")
    one = values.new_ones((1, 1))
    expected = values.view(1, width)
    assert torch.equal(triton_tree_gemm(one, values.view(width, 1)), expected)
    assert torch.equal(triton_dw_left_fold(one, expected), expected)


@requires_triton_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("block_m", "block_n"), [(32, 64), (16, 32)])
def test_triton_tile_geometry_preserves_bits(dtype, block_m, block_n, monkeypatch):
    from rl_engine.backends.shared.triton.gemm import attn_out_bias_gemm as triton_gemm

    # K=201 gives seven leaves with a short tail. S and N leave partial tiles.
    generator = torch.Generator().manual_seed(386)
    x, weight, bias, dy = (
        torch.randn(shape, generator=generator).to(dtype)
        for shape in ((35, 201), (97, 201), (97,), (35, 97))
    )

    def forward_and_grads(op, device):
        inputs = tuple(t.detach().to(device).requires_grad_(True) for t in (x, weight, bias))
        y = op(inputs[0], inputs[1], bias=inputs[2])
        return y, *torch.autograd.grad(y, inputs, dy.to(device))

    expected = forward_and_grads(NativeAttnOutBiasGemmOp(), "cpu")
    monkeypatch.setattr(triton_gemm, "_BLOCK_M", block_m)
    monkeypatch.setattr(triton_gemm, "_BLOCK_N", block_n)
    actual = forward_and_grads(triton_gemm.TritonAttnOutBiasGemmOp(), "cuda")
    for name, got, gold in zip(("y", "dx", "dW", "db"), actual, expected, strict=True):
        assert_bitwise_equal(got, gold, f"{name} with Triton tile {block_m}x{block_n}")


@requires_triton_cuda
def test_triton_tree_workspace():
    from rl_engine.backends.shared.triton.gemm.attn_out_bias_gemm import triton_tree_gemm

    # Seven leaves with a short tail exercise uneven splits at multiple depths.
    a = torch.randn(256, 201, device="cuda")
    b = torch.randn(512, 201, device="cuda")
    reference = tree_gemm_fp32(a.cpu(), b.cpu())
    triton_tree_gemm(a, b)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = triton_tree_gemm(a, b)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    assert peak <= result.numel() * result.element_size()
    assert_bitwise_equal(result, reference, "uneven Triton tree")
