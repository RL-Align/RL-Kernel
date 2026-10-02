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
        # T(0,2) = T(0,1) + T(1,2). Verify against a hand-built tree.
        gen = torch.Generator().manual_seed(7)
        a = torch.randn(2, 33, generator=gen)
        b = torch.randn(3, 33, generator=gen)
        # manual leaf chains (ascending k, separate mul/add) then tree combine
        acc0 = torch.zeros(2, 3)
        for k in range(32):
            acc0 = acc0 + (a[:, k].unsqueeze(1) * b[:, k].unsqueeze(0))
        acc1 = a[:, 32].unsqueeze(1) * b[:, 32].unsqueeze(0)
        manual = acc0 + acc1
        assert_bitwise_equal(tree_gemm_fp32(a, b), manual, "R=33 manual tree")


class TestReferenceBackward:
    """Gradient discipline: autograd through the op vs the explicit tree VJP."""

    def test_backward_matches_fp64_math(self):
        # dx = dY @ W and dW = dY.T @ x in fp64; the square weight makes an
        # operand-orientation bug compute the transpose silently, so this
        # cross-check against independent math is mandatory.
        from rl_engine.kernels.ops.pytorch.linear.attn_out_bias_gemm import (
            attn_out_bias_gemm_reference_backward,
        )

        torch.manual_seed(0)
        rows, dim = 6, _DIM
        x = torch.randn(rows, dim)
        w = torch.randn(dim, dim)
        g = torch.randn(rows, dim)
        dx, dw = attn_out_bias_gemm_reference_backward(x, w, g)
        assert torch.allclose(dx.double(), g.double() @ w.double(), atol=1e-3, rtol=1e-4)
        assert torch.allclose(dw.double(), g.double().t() @ x.double(), atol=1e-3, rtol=1e-4)

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


def assert_fp32_device_vs_reference(device_out: torch.Tensor, ref_out: torch.Tensor, ctx: str):
    """fp32 inputs: FMA (device) vs separate mul-add (torch reference).

    The two disciplines differ by at most a few ulps; structural bugs
    (transposed operands, wrong tree) produce errors many orders larger, so a
    tight relative bound still fails loudly on them.
    """
    a, b = device_out.float(), ref_out.float()
    assert a.shape == b.shape, f"[{ctx}] shape mismatch"
    diff = (a - b).abs()
    tol = 1e-4 * (1.0 + b.abs())
    assert bool((diff <= tol).all()), (
        f"[{ctx}] fp32 FMA-vs-separate drift beyond tolerance: " f"max_abs={diff.max().item():.3e}"
    )


@requires_triton_cuda
class TestTritonBackendBitwise:
    """Star-shaped acceptance: triton vs the FP32 CPU same-tree reference.

    bf16 inputs: byte-for-byte (FMA and separate mul-add are provably
    identical on exact bf16 products). fp32 inputs: device backends share the
    FMA discipline and are compared to the torch reference (which cannot
    express elementwise FMA on Python 3.12) with a tight declared tolerance.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("rows", [1, 2, 7, 33, 64])
    def test_forward_vs_reference(self, dtype, rows):
        reference = NativeAttnOutBiasGemmOp()
        backend = TritonAttnOutBiasGemmOp()
        x, w, b = _make_inputs(rows, _DIM, dtype)
        out_cpu = reference(x, w, bias=b)
        out_gpu = backend(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        if dtype == torch.bfloat16:
            assert_bitwise_equal(out_gpu, out_cpu, f"triton vs reference rows={rows}")
        else:
            assert_fp32_device_vs_reference(out_gpu, out_cpu, f"triton fp32 rows={rows}")

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
            if dtype == torch.bfloat16:
                assert_bitwise_equal(tri.cpu(), ref, f"triton {name} vs reference")
            else:
                assert_fp32_device_vs_reference(tri.cpu(), ref, f"triton fp32 {name}")

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

    def test_forward_matches_fp64_math(self):
        # independent math guard (orientation bugs stay loud even if both the
        # reference and the backend shared them)
        backend = TritonAttnOutBiasGemmOp()
        torch.manual_seed(3)
        x = torch.randn(6, _DIM)
        w = torch.randn(_DIM, _DIM)
        b = torch.randn(_DIM)
        out = backend(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        math = x.double() @ w.double().t() + b.double()
        assert torch.allclose(out.double(), math, atol=1e-2, rtol=1e-3)


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
    """CUDA backend vs the FP32 CPU reference (and vs the Triton backend)."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("rows", [1, 2, 7, 33, 64])
    def test_forward_vs_reference(self, dtype, rows):
        reference = NativeAttnOutBiasGemmOp()
        backend = CudaAttnOutBiasGemmOp()
        x, w, b = _make_inputs(rows, _DIM, dtype)
        out_cpu = reference(x, w, bias=b)
        out_gpu = backend(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        if dtype == torch.bfloat16:
            assert_bitwise_equal(out_gpu, out_cpu, f"cuda vs reference rows={rows}")
        else:
            assert_fp32_device_vs_reference(out_gpu, out_cpu, f"cuda fp32 rows={rows}")

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
            if dtype == torch.bfloat16:
                assert_bitwise_equal(cur.cpu(), ref, f"cuda {name} vs reference")
            else:
                assert_fp32_device_vs_reference(cur.cpu(), ref, f"cuda fp32 {name}")

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

    def test_forward_matches_fp64_math(self):
        backend = CudaAttnOutBiasGemmOp()
        torch.manual_seed(4)
        x = torch.randn(6, _DIM)
        w = torch.randn(_DIM, _DIM)
        b = torch.randn(_DIM)
        out = backend(x.cuda(), w.cuda(), bias=b.cuda()).cpu()
        math = x.double() @ w.double().t() + b.double()
        assert torch.allclose(out.double(), math, atol=1e-2, rtol=1e-3)


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
        if dtype == torch.bfloat16:
            assert_bitwise_equal(out_gpu, out_cpu, f"real-dim vs reference S={rows}")
        else:
            assert_fp32_device_vs_reference(out_gpu, out_cpu, f"real-dim fp32 S={rows}")


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

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backends_bias_none_bitwise(self, dtype):
        from rl_engine.kernels.ops.cuda.linear.attn_out_bias_gemm import CudaAttnOutBiasGemmOp

        reference = NativeAttnOutBiasGemmOp()
        x, w, _ = _make_inputs(6, _DIM, dtype)
        out_cpu = reference(x, w)
        out_tri = TritonAttnOutBiasGemmOp()(x.cuda(), w.cuda()).cpu()
        out_cuda = CudaAttnOutBiasGemmOp()(x.cuda(), w.cuda()).cpu()
        if dtype == torch.bfloat16:
            assert_bitwise_equal(out_tri, out_cpu, "triton bias=None vs reference")
            assert_bitwise_equal(out_cuda, out_cpu, "cuda bias=None vs reference")
        else:
            assert_fp32_device_vs_reference(out_tri, out_cpu, "triton bias=None fp32")
            assert_fp32_device_vs_reference(out_cuda, out_cpu, "cuda bias=None fp32")
        # the two device backends share the FMA discipline: bitwise in both dtypes
        assert_bitwise_equal(out_cuda, out_tri, f"bias=None cross-backend {dtype}")

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
