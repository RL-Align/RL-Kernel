# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Bit-equality harness and tests for the Qwen-Image txt_in_rmsnorm_linear.

The harness implements the frozen comparison procedure (contract v3.6): assert
shape, then dtype, then materialise contiguous copies, then compare the bit
patterns of the logical elements via a dtype bitcast (int16 for bf16, int32
for fp32). Zero tolerance paths anywhere in this file.
"""

from __future__ import annotations

import struct

import pytest
import torch

from rl_engine.kernels.ops.pytorch.norm.txt_in_rmsnorm_linear import (
    TXT_IN_HIDDEN,
    TXT_IN_OUT,
    NativeTxtInRMSNormLinearOp,
    row_sumsq_tree,
    txt_in_row_dot_tree,
)

try:
    import ctypes

    _LIBM = ctypes.CDLL("libm.so.6")
    _LIBM.fmaf.restype = ctypes.c_float
    _LIBM.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    _LIBM.sqrtf.restype = ctypes.c_float
    _LIBM.sqrtf.argtypes = [ctypes.c_float]
    _HAS_FMAF = True
except Exception:  # pragma: no cover
    _HAS_FMAF = False

_requires_fmaf = pytest.mark.skipif(not _HAS_FMAF, reason="libm fmaf unavailable")

_DTYPE_BITVIEW = {torch.bfloat16: torch.int16, torch.float32: torch.int32}
_EPS32 = struct.unpack("<f", struct.pack("<I", 0x358637BD))[0]


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _bitview(dtype: torch.dtype) -> torch.dtype:
    return _DTYPE_BITVIEW[dtype]


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Contract comparison: logical-element bit patterns."""
    if a.shape != b.shape or a.dtype != b.dtype:
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


def _make_inputs(rows: int, seed: int = 1234, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, TXT_IN_HIDDEN, generator=gen).to(dtype)
    gamma = torch.randn(TXT_IN_HIDDEN, generator=gen).to(dtype)
    W = torch.randn(TXT_IN_OUT, TXT_IN_HIDDEN, generator=gen).to(dtype)
    b = torch.randn(TXT_IN_OUT, generator=gen).to(dtype)
    return x, gamma, W, b


# ---------------------------------------------------------------------------
# Independent spec (mandatory rule: separate code path, same frozen semantics).
# Scalar fmaf/sqrtf chains, own indexing -- no shared logic with the op.
# ---------------------------------------------------------------------------
def _spec_leaf_fma(a_vec, b_vec, k0, k1):
    acc = 0.0
    for k in range(k0, k1):
        acc = _LIBM.fmaf(a_vec[k], b_vec[k], acc)
    return acc


def _spec_tree(a_vec, b_vec, red, lo, hi):
    if hi - lo == 1:
        k0 = lo * 32
        return _spec_leaf_fma(a_vec, b_vec, k0, min(k0 + 32, red))
    mid = lo + (hi - lo) // 2
    return _f32(_spec_tree(a_vec, b_vec, red, lo, mid) + _spec_tree(a_vec, b_vec, red, mid, hi))


def _spec_norm_stats(x_row):
    hidden = len(x_row)
    # sumsq: FMA(x, x, acc) chains == FMA chain over the row values themselves
    sumsq = _spec_tree(x_row, x_row, hidden, 0, (hidden + 31) // 32)
    var = _f32(sumsq / 3584.0)
    t = _f32(var + _EPS32)
    sq32 = _f32(_LIBM.sqrtf(t))
    rstd = _f32(1.0 / sq32)
    xhat = [_f32(v * rstd) for v in x_row]
    return xhat, rstd


def _spec_forward(x2d, gamma, W, b, dtype):
    rows = len(x2d)
    # x2d: list of rows, each a flat list of H floats
    out = []
    for s in range(rows):
        xhat, _ = _spec_norm_stats(x2d[s])
        z = [_f32(xhat[h] * gamma[h]) for h in range(TXT_IN_HIDDEN)]
        for n in range(len(W)):
            tree = _spec_tree(z, W[n], TXT_IN_HIDDEN, 0, (TXT_IN_HIDDEN + 31) // 32)
            v = _f32(tree + b[n])
            out.append(_f32(v))
    t = torch.tensor(out, dtype=torch.float32).view(rows, len(W))
    return t.to(dtype)


class TestReferenceSelfChecks:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_deterministic_repeat(self, dtype):
        op = NativeTxtInRMSNormLinearOp()
        x, gamma, W, b = _make_inputs(3, dtype=dtype)
        assert_bitwise_equal(op(x, gamma, W, bias=b), op(x, gamma, W, bias=b), "repeat")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_row_batch_invariance(self, dtype):
        op = NativeTxtInRMSNormLinearOp()
        x, gamma, W, b = _make_inputs(7, dtype=dtype)
        full = op(x, gamma, W, bias=b)
        assert_bitwise_equal(op(x[:1], gamma, W, bias=b), full[:1], "row0 alone")
        assert_bitwise_equal(op(x[3:5], gamma, W, bias=b), full[3:5], "rows 3:5")
        assert_bitwise_equal(op(x.unsqueeze(0), gamma, W, bias=b).squeeze(0), full, "leading [1,S]")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_padding_invariance(self, dtype):
        op = NativeTxtInRMSNormLinearOp()
        x, gamma, W, b = _make_inputs(4, dtype=dtype)
        pad = torch.randn(6, TXT_IN_HIDDEN).to(dtype)
        padded = torch.cat([x, pad], dim=0)
        assert_bitwise_equal(op(padded, gamma, W, bias=b)[:4], op(x, gamma, W, bias=b), "padded")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_single_cast_vs_gold(self, dtype):
        op = NativeTxtInRMSNormLinearOp()
        x, gamma, W, b = _make_inputs(5, dtype=dtype)
        out = op(x, gamma, W, bias=b)
        gold = op.forward_fp32(x, gamma, W, bias=b)
        assert_bitwise_equal(out, gold.to(dtype), "single cast at output")


class TestReferenceBackward:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_gradients_flow(self, dtype):
        op = NativeTxtInRMSNormLinearOp()
        x, gamma, W, b = _make_inputs(3, dtype=dtype)
        for t in (x, gamma, W, b):
            t.requires_grad_(True)
        out = op(x, gamma, W, bias=b)
        out.backward(torch.randn_like(out))
        for t, n in ((x, "dx"), (gamma, "dgamma"), (W, "dW"), (b, "db")):
            assert t.grad is not None and torch.isfinite(t.grad.float()).all(), n

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_dx_row_batch_invariance(self, dtype):
        op = NativeTxtInRMSNormLinearOp()
        x, gamma, W, b = _make_inputs(6, dtype=dtype)
        dY = torch.randn(6, TXT_IN_OUT, generator=torch.Generator().manual_seed(77)).to(dtype)
        xd = x.clone().requires_grad_(True)
        out = op(xd, gamma, W, bias=b)
        out.backward(dY)
        xs = x[:2].clone().requires_grad_(True)
        op(xs, gamma, W, bias=b).backward(dY[:2])
        assert_bitwise_equal(xs.grad, xd.grad[:2], "dx slice invariance")


class TestValidation:
    def test_mixed_dtypes_rejected(self):
        op = NativeTxtInRMSNormLinearOp()
        x = torch.randn(2, TXT_IN_HIDDEN, dtype=torch.bfloat16)
        gamma = torch.randn(TXT_IN_HIDDEN, dtype=torch.float32)
        W = torch.randn(TXT_IN_OUT, TXT_IN_HIDDEN, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="share one dtype"):
            op(x, gamma, W)

    def test_wrong_shapes_rejected(self):
        op = NativeTxtInRMSNormLinearOp()
        x = torch.randn(2, 4096)
        gamma = torch.randn(TXT_IN_HIDDEN)
        W = torch.randn(TXT_IN_OUT, TXT_IN_HIDDEN)
        with pytest.raises(ValueError):
            op(x, gamma, W)


@_requires_fmaf
class TestIndependentSpecBitwise:
    """Independent implementation of the frozen semantics, bitwise vs the
    reference (small dims keep the scalar spec tractable)."""

    def test_forward_vs_independent_spec(self):
        # small hidden/out dims: build op-alternative via direct construction
        # using the op's own shapes but the spec's own code path.  For speed
        # we exercise full 3584->3072 on one row.
        torch.manual_seed(5)
        x = torch.randn(1, TXT_IN_HIDDEN).float()
        gamma = torch.randn(TXT_IN_HIDDEN).float()
        W = torch.randn(TXT_IN_OUT, TXT_IN_HIDDEN).float()
        b = torch.randn(TXT_IN_OUT).float()
        op_out = NativeTxtInRMSNormLinearOp()(x, gamma, W, bias=b)
        spec = _spec_forward([x[0].tolist()], gamma.tolist(), W.tolist(), b.tolist(), torch.float32)
        assert_bitwise_equal(op_out, spec, "reference vs spec")

    def test_backward_vs_independent_spec(self):
        # Full frozen dims (the op is fail-closed on [3072, 3584]); one row
        # keeps the scalar spec tractable: dgamma/dW/db single-row folds are
        # exact, and dz costs one 96-leaf tree per h.
        torch.manual_seed(6)
        rows = 1
        x = torch.randn(rows, TXT_IN_HIDDEN).float()
        gamma = torch.randn(TXT_IN_HIDDEN).float()
        W = torch.randn(TXT_IN_OUT, TXT_IN_HIDDEN).float()
        b = torch.randn(TXT_IN_OUT).float()
        dY = torch.randn(rows, TXT_IN_OUT).float()

        xd = x.clone().requires_grad_(True)
        gd = gamma.clone().requires_grad_(True)
        Wd = W.clone().requires_grad_(True)
        bd = b.clone().requires_grad_(True)
        NativeTxtInRMSNormLinearOp()(xd, gd, Wd, bias=bd).backward(dY)

        g_l = dY[0].tolist()
        W_l = W.tolist()
        Wt = list(map(list, zip(*W_l)))  # Wt[h][n] = W[n][h], pure transpose
        del W_l

        xhat_l, rstd = _spec_norm_stats(x[0].tolist())
        gamma_l = gamma.tolist()
        z = [_f32(xhat_l[h] * gamma_l[h]) for h in range(TXT_IN_HIDDEN)]

        # dz[h] = N-dim tree (96 leaves) of dY[n]*W[n,h]; du := dz (identity)
        dz = [
            _spec_tree(g_l, Wt[h], TXT_IN_OUT, 0, (TXT_IN_OUT + 31) // 32)
            for h in range(TXT_IN_HIDDEN)
        ]
        dxh = [_f32(dz[h] * gamma_l[h]) for h in range(TXT_IN_HIDDEN)]
        dot = _spec_tree(dxh, xhat_l, TXT_IN_HIDDEN, 0, (TXT_IN_HIDDEN + 31) // 32)
        t1 = _f32(dot / 3584.0)
        # t2/t3 isolated ops -- contraction into FMS is forbidden by contract
        dx_row = [_f32(rstd * _f32(dxh[h] - _f32(xhat_l[h] * t1))) for h in range(TXT_IN_HIDDEN)]
        assert_bitwise_equal(xd.grad[0], torch.tensor(dx_row, dtype=torch.float32), "dx row")

        # dgamma = ascending-row FMA fold from +0.0; one row -> single fmaf
        dgamma_spec = torch.tensor(
            [_LIBM.fmaf(dz[h], xhat_l[h], 0.0) for h in range(TXT_IN_HIDDEN)]
        )
        assert_bitwise_equal(gd.grad, dgamma_spec, "dgamma fold")

        # dW[n,h] = FMA(g[n], z[h]) from +0.0; spot columns + spot rows
        for h in (0, 1, 1791, 3583):
            col = torch.tensor([_LIBM.fmaf(g_l[n], z[h], 0.0) for n in range(TXT_IN_OUT)])
            assert_bitwise_equal(Wd.grad[:, h], col, f"dW col {h}")
        for n in (0, 1536, 3071):
            row = torch.tensor([_LIBM.fmaf(g_l[n], z[h], 0.0) for h in range(TXT_IN_HIDDEN)])
            assert_bitwise_equal(Wd.grad[n], row, f"dW row {n}")

        # db = ascending pure-add fold from +0.0; one row -> identity
        assert_bitwise_equal(bd.grad, dY[0].contiguous(), "db fold")


@_requires_fmaf
class TestNormPrimitiveDetails:
    """Contract-pinned primitives: three-step rstd, sumsq tree, FMA dot."""

    def test_three_step_rstd_vs_sqrtf_chain(self):
        torch.manual_seed(7)
        var = (torch.rand(1 << 16) * 99.0).float()
        t32 = var + _EPS32
        r_ref = torch.tensor([_f32(1.0 / _LIBM.sqrtf(v)) for v in t32.tolist()])
        from rl_engine.kernels.ops.pytorch.norm.txt_in_rmsnorm_linear import three_step_rstd

        assert_bitwise_equal(three_step_rstd(var), r_ref, "three-step rstd")

    def test_whole_fp64_rsqrt_is_forbidden_form(self):
        # the forbidden form actually differs -- guard against reintroduction
        torch.manual_seed(8)
        var = (torch.rand(4096) * 99.0).float()
        t32 = (var + _EPS32).double()
        whole = (1.0 / torch.sqrt(t32)).float()
        three_step = 1.0 / torch.sqrt(t32).float()
        assert not torch.equal(
            whole.view(torch.int32), three_step.view(torch.int32)
        ), "expected the two forms to differ (1471/5000); if equal, tighten"

    def test_row_sumsq_tree_matches_manual_fma(self):
        torch.manual_seed(9)
        x = torch.randn(2, 33).float()  # short tail leaf
        got = row_sumsq_tree(x)
        for s in range(2):
            acc = 0.0
            for h in range(32):
                acc = _LIBM.fmaf(x[s, h].item(), x[s, h].item(), acc)
            acc = _f32(acc + _LIBM.fmaf(x[s, 32].item(), x[s, 32].item(), 0.0))
            assert _f32(got[s].item()) == acc, s

    def test_row_dot_tree_is_fma_not_mul_then_sum(self):
        # adversarial: mul-then-sum rounds the product; FMA does not
        torch.manual_seed(10)
        a = torch.full((1, 2), (1.0 + 2**-12)).float()
        c_seed = _LIBM.fmaf(2**-30, 2**-30, 0.0)  # 2^-60
        b = torch.tensor([[c_seed, 1.0 + 2**-12]]).float()
        got = txt_in_row_dot_tree(a, b).item()
        expect = _LIBM.fmaf(
            (1.0 + 2**-12),
            (1.0 + 2**-12),
            _LIBM.fmaf(2**-30, 2**-30, 0.0),
        )
        assert _f32(got) == _f32(expect)


# ---------------------------------------------------------------------------
# Device backends: Triton and CUDA vs the FP32 CPU gold (star acceptance).
# ---------------------------------------------------------------------------
try:
    from rl_engine.kernels.ops.triton.norm.txt_in_rmsnorm_linear import TritonTxtInRMSNormLinearOp

    _HAS_TRITON_OP = True
except Exception:  # pragma: no cover - triton or CUDA missing
    _HAS_TRITON_OP = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON_OP and torch.cuda.is_available()),
    reason="Triton backend requires triton and a CUDA device",
)

try:
    from rl_engine.kernels.ops.cuda.norm.txt_in_rmsnorm_linear import CudaTxtInRMSNormLinearOp

    _HAS_CUDA_OP = True
except Exception:  # pragma: no cover - extension not built
    _HAS_CUDA_OP = False

requires_cuda_ext = pytest.mark.skipif(
    not (_HAS_CUDA_OP and torch.cuda.is_available()),
    reason="CUDA backend requires the compiled extension and a CUDA device",
)

requires_any_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="division probe needs a CUDA device"
)


def _grads_on(op, x, gamma, W, b, dY, device):
    xa = x.detach().clone().to(device).requires_grad_(True)
    ga = gamma.detach().clone().to(device).requires_grad_(True)
    Wa = W.detach().clone().to(device).requires_grad_(True)
    ba = b.detach().clone().to(device).requires_grad_(True)
    op(xa, ga, Wa, bias=ba).backward(dY.to(device))
    return xa.grad, ga.grad, Wa.grad, ba.grad


class TestTrueDivisionDiscipline:
    """The var/t1 quotients must be correctly rounded on EVERY device.

    torch CUDA ``tensor / python_float`` silently multiplies by the
    single-rounded reciprocal (CPU divides); the contract's division runs
    through ``true_div_rn``. The quotient of two fp32 values is exact in
    fp64 (24-bit significands, <= 48-bit quotient), so the fp64-relay is an
    EXACT oracle for correctly-rounded fp32 division.
    """

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_true_div_rn_matches_exact_quotient(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        from rl_engine.kernels.ops.pytorch.norm.txt_in_rmsnorm_linear import true_div_rn

        gen = torch.Generator().manual_seed(31)
        ss = (torch.rand(1 << 16, generator=gen) * 4000.0).to(device)
        oracle = (ss.double() / 3584.0).float()
        assert_bitwise_equal(true_div_rn(ss, 3584.0), oracle, f"exact quotient {device}")

    @requires_any_cuda
    def test_raw_scalar_div_is_the_trap(self):
        # documents why the helper exists: the raw CUDA scalar division is a
        # reciprocal-multiply and differs from the exact quotient on this
        # distribution (~55% of samples historically)
        gen = torch.Generator().manual_seed(31)
        ss = torch.rand(1 << 16, generator=gen) * 4000.0
        oracle = (ss.double() / 3584.0).float().cuda()
        raw = ss.cuda() / 3584.0
        assert not torch.equal(raw.view(torch.int32), oracle.view(torch.int32))


@requires_triton_cuda
class TestTritonBackendBitwise:
    """Triton vs the FP32 CPU reference: both dtypes byte-for-byte.

    One FMA discipline on both sides (kernels use libdevice.fma_rn; the
    reference uses torch.addcmul, an oracle-verified correctly-rounded FMA).
    No tolerance path.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("rows", [1, 2, 7, 300])
    def test_forward_vs_reference(self, dtype, rows):
        reference = NativeTxtInRMSNormLinearOp()
        backend = TritonTxtInRMSNormLinearOp()
        x, gamma, W, b = _make_inputs(rows, dtype=dtype)
        out_cpu = reference(x, gamma, W, bias=b)
        out_gpu = backend(x.cuda(), gamma.cuda(), W.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(out_gpu, out_cpu, f"triton vs reference rows={rows}")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward_vs_reference(self, dtype):
        reference = NativeTxtInRMSNormLinearOp()
        backend = TritonTxtInRMSNormLinearOp()
        x, gamma, W, b = _make_inputs(5, dtype=dtype)
        dY = torch.randn(5, TXT_IN_OUT, generator=torch.Generator().manual_seed(99)).to(dtype)
        grads_ref = _grads_on(reference, x, gamma, W, b, dY, "cpu")
        grads_gpu = _grads_on(backend, x, gamma, W, b, dY, "cuda")
        for ref, cur, name in zip(grads_ref, grads_gpu, ("dx", "dgamma", "dW", "db")):
            assert_bitwise_equal(cur.cpu(), ref, f"triton {name} vs reference")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_invariance_on_gpu(self, dtype):
        backend = TritonTxtInRMSNormLinearOp()
        x, gamma, W, b = _make_inputs(9, dtype=dtype)
        full = backend(x.cuda(), gamma.cuda(), W.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(
            backend(x[:3].cuda(), gamma.cuda(), W.cuda(), bias=b.cuda()).cpu(),
            full[:3],
            "triton slice",
        )
        pad = torch.randn(5, TXT_IN_HIDDEN).to(dtype)
        padded = torch.cat([x, pad], dim=0)
        assert_bitwise_equal(
            backend(padded.cuda(), gamma.cuda(), W.cuda(), bias=b.cuda()).cpu()[:9],
            full,
            "triton padded",
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_zero_rows(self, dtype):
        x = torch.zeros(0, TXT_IN_HIDDEN).to(dtype).cuda()
        gamma = torch.randn(TXT_IN_HIDDEN).to(dtype).cuda()
        W = torch.randn(TXT_IN_OUT, TXT_IN_HIDDEN).to(dtype).cuda()
        b = torch.randn(TXT_IN_OUT).to(dtype).cuda()
        out = TritonTxtInRMSNormLinearOp()(x, gamma, W, bias=b)
        assert out.shape == (0, TXT_IN_OUT)


@requires_cuda_ext
class TestCudaBackendBitwise:
    """CUDA backend vs the FP32 CPU reference (and vs the Triton backend)."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("rows", [1, 2, 7, 300])
    def test_forward_vs_reference(self, dtype, rows):
        reference = NativeTxtInRMSNormLinearOp()
        backend = CudaTxtInRMSNormLinearOp()
        x, gamma, W, b = _make_inputs(rows, dtype=dtype)
        out_cpu = reference(x, gamma, W, bias=b)
        out_gpu = backend(x.cuda(), gamma.cuda(), W.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(out_gpu, out_cpu, f"cuda vs reference rows={rows}")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward_vs_reference(self, dtype):
        reference = NativeTxtInRMSNormLinearOp()
        backend = CudaTxtInRMSNormLinearOp()
        x, gamma, W, b = _make_inputs(5, dtype=dtype)
        dY = torch.randn(5, TXT_IN_OUT, generator=torch.Generator().manual_seed(99)).to(dtype)
        grads_ref = _grads_on(reference, x, gamma, W, b, dY, "cpu")
        grads_gpu = _grads_on(backend, x, gamma, W, b, dY, "cuda")
        for ref, cur, name in zip(grads_ref, grads_gpu, ("dx", "dgamma", "dW", "db")):
            assert_bitwise_equal(cur.cpu(), ref, f"cuda {name} vs reference")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.skipif(
        not (_HAS_TRITON_OP and torch.cuda.is_available()),
        reason="cross-backend needs triton",
    )
    def test_cuda_vs_triton_bitwise(self, dtype):
        cuda_op = CudaTxtInRMSNormLinearOp()
        triton_op = TritonTxtInRMSNormLinearOp()
        x, gamma, W, b = _make_inputs(9, dtype=dtype)
        out_cuda = cuda_op(x.cuda(), gamma.cuda(), W.cuda(), bias=b.cuda()).cpu()
        out_triton = triton_op(x.cuda(), gamma.cuda(), W.cuda(), bias=b.cuda()).cpu()
        assert_bitwise_equal(out_cuda, out_triton, f"cuda vs triton {dtype}")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_zero_rows(self, dtype):
        x = torch.zeros(0, TXT_IN_HIDDEN).to(dtype).cuda()
        gamma = torch.randn(TXT_IN_HIDDEN).to(dtype).cuda()
        W = torch.randn(TXT_IN_OUT, TXT_IN_HIDDEN).to(dtype).cuda()
        b = torch.randn(TXT_IN_OUT).to(dtype).cuda()
        out = CudaTxtInRMSNormLinearOp()(x, gamma, W, bias=b)
        assert out.shape == (0, TXT_IN_OUT)
