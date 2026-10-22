# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 projection rows: ``audio_input_projection_fp32``, ``audio_output_projection_fp32``,
``text_context_projection`` and ``h3_qkv_gemm``.

* accuracy: forward and every gradient against an FP64 golden, on synthetic data and on
  the pinned checkpoint tensors (shard 1), with the provider (``F.linear``) as yardstick;
* invariance: a token's bytes do not depend on the batch size, the row position or the other
  rows; a tensor-parallel output shard equals the same columns of the full product;
* QKV layouts: the fused ``[Q; K; V]`` product equals the separate projections column for
  column, and their backwards agree bitwise;
* determinism and fail-closed contract (dtype, bias, widths, devices, missing extension).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from rl_engine.reference.minimax_h3 import projection as ref

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
SHARD1 = "diffusion_pytorch_model-00001-of-00014.safetensors"

ROWS = {
    "audio_input_projection_fp32": (ref.AUDIO_INPUT, "audio_proj_in"),
    "audio_output_projection_fp32": (ref.AUDIO_OUTPUT, "audio_proj_out"),
    "text_context_projection": (ref.TEXT_CONTEXT, "context_embedder"),
}
REFERENCE = {
    "audio_input_projection_fp32": ref.NativeH3AudioInputProjectionOp,
    "audio_output_projection_fp32": ref.NativeH3AudioOutputProjectionOp,
    "text_context_projection": ref.NativeH3TextContextProjectionOp,
}


def _cuda_ops():
    from rl_engine.backends.cuda.model_specific.minimax_h3 import projection as cu

    if not cu.gemm.available():
        pytest.skip("needs triton and a CUDA device")
    return cu


def _cuda_op(row):
    cu = _cuda_ops()
    return {
        "audio_input_projection_fp32": cu.H3AudioInputProjectionCudaOp,
        "audio_output_projection_fp32": cu.H3AudioOutputProjectionCudaOp,
        "text_context_projection": cu.H3TextContextProjectionCudaOp,
    }[row]()


def _params(spec, n=None, seed=0, device="cuda"):
    n = n or spec.out_features
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(n, spec.in_features, generator=g) / spec.in_features**0.5).to(spec.dtype)
    b = (torch.randn(n, generator=g) * 0.1).to(spec.dtype) if spec.has_bias else None
    return w.to(device), None if b is None else b.to(device)


def _x(spec, t, seed=1, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(t, spec.in_features, generator=g).to(spec.dtype).to(device)


def _real(names):
    root = os.environ.get("RL_KERNEL_H3_WEIGHTS")
    path = Path(root) / "hf" / "transformer" / SHARD1 if root else None
    if path is None or not path.exists():
        pytest.skip("RL_KERNEL_H3_WEIGHTS/hf/transformer shard 1 not available")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {n: f.get_tensor(n).cuda() for n in names}


def _floor(spec):
    """Absolute slack (relative to max|golden|) on top of 1.25x the provider's error.

    A sequential FP32 chain over K = 5376 lands near 2-3e-6 (cuBLAS's blocked sums reach
    ~5e-7); BF16 outputs round once at 2**-9 so 4e-3 covers a one-ulp flip at the maximum.
    """

    if spec.dtype == torch.float32:
        return 4e-6 if spec.in_features > 1024 else 1e-6
    return 4e-3


def _bits(t):
    return t.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _golden_grads(x, w, grad):
    xd, wd, gd = x.double(), w.double(), grad.double()
    return gd @ wd, gd.t() @ xd, gd.sum(0)


def _rel(value, golden):
    g = golden.double()
    return float((value.double() - g).abs().max() / g.abs().max().clamp_min(1e-30))


# --- CPU: reference and contract ------------------------------------------------------------


@pytest.mark.parametrize("row", sorted(ROWS))
def test_reference_provider_and_golden_agree(row):
    spec, _ = ROWS[row]
    op = REFERENCE[row]()
    w, b = _params(spec, device="cpu")
    x = _x(spec, 9, device="cpu")
    provider, golden = op.forward(x, w, b), op.forward_fp32(x, w, b)
    assert provider.dtype == golden.dtype == spec.dtype
    assert _rel(provider, golden) < (1e-5 if spec.dtype == torch.float32 else 2e-2)


@pytest.mark.parametrize("row", sorted(ROWS))
def test_reference_rejects_wrong_dtype_bias_and_shapes(row):
    spec, _ = ROWS[row]
    op = REFERENCE[row]()
    w, b = _params(spec, device="cpu")
    x = _x(spec, 4, device="cpu")
    other = torch.bfloat16 if spec.dtype == torch.float32 else torch.float32
    for bad in (
        (x.to(other), w, b),
        (x, w.to(other), b),
        (x, w, b.to(other)),
        (x, w, None),  # these modules have a bias
        (x[:0], w, b),
        (x[:, :-1], w, b),
        (x.flatten(), w, b),
        (x, w[:-1], b[:-1]),  # not a legal width
        (x, w, b[:-1]),
    ):
        with pytest.raises((TypeError, ValueError)):
            op.forward(*bad)


def test_qkv_reference_contract():
    spec = ref.QKV
    op = ref.NativeH3QKVGemmOp()
    ws = [_params(spec, seed=s, device="cpu")[0] for s in (1, 2, 3)]
    x = _x(spec, 5, device="cpu")
    q, k, v = op.forward(x, *ws)
    assert all(t.shape == (5, 7168) and t.dtype == torch.bfloat16 for t in (q, k, v))
    fused = op.forward_fused(x, ref.fuse_qkv_weights(*ws))
    assert fused.shape == (5, 3 * 7168)
    assert torch.equal(fused[:, :7168], q) or _rel(fused[:, :7168], q) < 1e-2  # CPU BLAS order
    with pytest.raises(ValueError):
        op.forward(x, ws[0], ws[1], ws[2][:-128])  # widths differ
    with pytest.raises(ValueError):
        op.forward(x, *[w[:100] for w in ws])  # not whole heads
    with pytest.raises(ValueError):
        op.forward_fused(x, ws[0][:-1])
    shard = op.forward(x, *[w[:1792] for w in ws])  # TP4 head shard
    assert shard[0].shape == (5, 1792)


def test_registry_dispatches_reference_on_cpu():
    from rl_engine.runtime.registry import KernelRegistry

    registry = KernelRegistry()
    for row, cls in REFERENCE.items():
        assert type(registry.get_op(row, device="cpu")) is cls
    assert type(registry.get_op("h3_qkv_gemm", device="cpu")) is ref.NativeH3QKVGemmOp


# --- CUDA: accuracy -----------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("row", sorted(ROWS))
@pytest.mark.parametrize("tokens", [1, 7, 8, 9, 257, 1025])
def test_forward_against_fp64_golden(row, tokens):
    spec, _ = ROWS[row]
    op, native = _cuda_op(row), REFERENCE[row]()
    w, b = _params(spec)
    x = _x(spec, tokens)
    out = op.forward(x, w, b)
    golden = native.forward_fp32(x, w, b)
    assert out.dtype == spec.dtype and out.shape == golden.shape
    with (
        torch.backends.cuda.matmul.flags(allow_tf32=False)
        if hasattr(torch.backends.cuda.matmul, "flags")
        else _noop()
    ):
        provider = native.forward(x, w, b)
    floor = _floor(spec)
    assert _rel(out, golden) <= 1.25 * _rel(provider, golden) + floor


class _noop:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@requires_cuda
@pytest.mark.parametrize("row", sorted(ROWS))
def test_forward_on_pinned_checkpoint_weights(row):
    spec, module = ROWS[row]
    t = _real([f"{module}.weight", f"{module}.bias"])
    w, b = t[f"{module}.weight"], t[f"{module}.bias"]
    assert (w.dtype, tuple(w.shape)) == (spec.dtype, (spec.out_features, spec.in_features))
    x = _x(spec, 300, seed=5)
    out = _cuda_op(row).forward(x, w, b)
    golden = REFERENCE[row]().forward_fp32(x, w, b)
    floor = _floor(spec)
    provider = F.linear(x, w, b)
    assert _rel(out, golden) <= 1.25 * _rel(provider, golden) + floor


@requires_cuda
def test_qkv_on_pinned_checkpoint_weights():
    names = [f"transformer_blocks.0.attn.to_{c}.weight" for c in "qkv"]
    t = _real(names)
    x = _x(ref.QKV, 130, seed=2)
    cu = _cuda_ops()
    outs = cu.H3QKVGemmCudaOp().forward(x, *[t[n] for n in names])
    golden = ref.NativeH3QKVGemmOp().forward_fp32(x, *[t[n] for n in names])
    for out, gold, n in zip(outs, golden, names):
        assert _rel(out, gold) <= 1.25 * _rel(F.linear(x, t[n]), gold) + 4e-3


@requires_cuda
@pytest.mark.parametrize("row", sorted(ROWS))
def test_backward_against_fp64_golden(row):
    spec, _ = ROWS[row]
    op = _cuda_op(row)
    w, b = _params(spec)
    w.requires_grad_(), b.requires_grad_()
    x = _x(spec, 129).requires_grad_()
    out = op.forward(x, w, b)
    grad = (
        torch.randn(out.shape, generator=torch.Generator("cpu").manual_seed(3)).to(out.dtype).cuda()
    )
    out.backward(grad)
    dx, dw, db = _golden_grads(x.detach(), w.detach(), grad)
    floor = 1e-5 if spec.dtype == torch.float32 else 2e-2
    assert _rel(x.grad, dx) < floor
    assert _rel(w.grad, dw) < floor
    assert _rel(b.grad, db) < floor
    assert x.grad.dtype == w.grad.dtype == b.grad.dtype == spec.dtype


# --- CUDA: invariance and determinism ----------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("row", sorted(ROWS))
def test_rows_are_batch_and_position_invariant(row):
    spec, _ = ROWS[row]
    op = _cuda_op(row)
    w, b = _params(spec)
    x = _x(spec, 67)
    full = op.forward(x, w, b)
    for start, stop in ((0, 1), (3, 4), (60, 67), (1, 64), (8, 17)):
        assert _bits(op.forward(x[start:stop].contiguous(), w, b)) == _bits(full[start:stop])
    perm = torch.randperm(67, generator=torch.Generator("cpu").manual_seed(0)).cuda()
    assert _bits(op.forward(x[perm].contiguous(), w, b)) == _bits(full[perm])
    padded = torch.cat([x, _x(spec, 33, seed=9)])
    assert _bits(op.forward(padded, w, b)[:67]) == _bits(full)


def _splittable(row):
    spec, _ = ROWS[row]
    return spec.out_features % (2 * spec.shard_unit) == 0


@requires_cuda
@pytest.mark.parametrize("row", [r for r in sorted(ROWS) if _splittable(r)])
def test_output_shard_equals_columns_of_full_product(row):
    spec, _ = ROWS[row]
    op = _cuda_op(row)
    w, b = _params(spec)
    x = _x(spec, 40)
    full = op.forward(x, w, b)
    half = spec.out_features // 2
    assert _bits(op.forward(x, w[:half].contiguous(), b[:half].contiguous())) == _bits(
        full[:, :half]
    )
    assert _bits(op.forward(x, w[half:].contiguous(), b[half:].contiguous())) == _bits(
        full[:, half:]
    )


@requires_cuda
@pytest.mark.parametrize("row", [r for r in sorted(ROWS) if not _splittable(r)])
def test_single_unit_output_is_not_shardable_and_fails_closed(row):
    """A row whose full width is one ``shard_unit`` (``audio_output_projection_fp32``: 32 columns)
    has no legal output shard: tensor parallelism replicates it (or splits the reduction axis, which
    the WS2 row-parallel path covers). Any narrower weight must be rejected, not silently computed.
    """
    spec, _ = ROWS[row]
    assert spec.out_features == spec.shard_unit
    op = _cuda_op(row)
    w, b = _params(spec)
    x = _x(spec, 8)
    assert op.forward(x, w, b).shape == (8, spec.out_features)
    half = spec.out_features // 2
    with pytest.raises(ValueError):
        op.forward(x, w[:half].contiguous(), b[:half].contiguous())


@requires_cuda
@pytest.mark.parametrize("row", sorted(ROWS))
def test_repeat_and_backward_are_deterministic(row):
    spec, _ = ROWS[row]
    op = _cuda_op(row)
    w, b = _params(spec)

    def run():
        x = _x(spec, 70).requires_grad_()
        wp, bp = w.clone().requires_grad_(), b.clone().requires_grad_()
        out = op.forward(x, wp, bp)
        out.backward(torch.ones_like(out))
        return [_bits(t) for t in (out, x.grad, wp.grad, bp.grad)]

    assert run() == run()


@requires_cuda
def test_dx_rows_are_batch_invariant():
    spec = ref.TEXT_CONTEXT
    op = _cuda_op("text_context_projection")
    w, b = _params(spec)
    g = (
        torch.randn(40, spec.out_features, generator=torch.Generator("cpu").manual_seed(7))
        .to(spec.dtype)
        .cuda()
    )

    def dx(rows):
        x = _x(spec, 40)[rows].requires_grad_()
        out = op.forward(x, w, b)
        out.backward(g[rows])
        return x.grad

    full = dx(slice(0, 40))
    assert _bits(dx(slice(5, 9))) == _bits(full[5:9])


# --- CUDA: h3_qkv_gemm layouts -----------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 9, 130])
def test_fused_qkv_equals_separate_columns_bitwise(tokens):
    cu = _cuda_ops()
    op = cu.H3QKVGemmCudaOp()
    ws = [_params(ref.QKV, seed=s)[0] for s in (1, 2, 3)]
    x = _x(ref.QKV, tokens)
    q, k, v = op.forward(x, *ws)
    fused = op.forward_fused(x, ref.fuse_qkv_weights(*ws))
    n = ref.QKV.out_features
    assert _bits(fused[:, :n]) == _bits(q)
    assert _bits(fused[:, n : 2 * n]) == _bits(k)
    assert _bits(fused[:, 2 * n :]) == _bits(v)


@requires_cuda
def test_fused_and_separate_qkv_backward_are_bitwise_equal():
    cu = _cuda_ops()
    op = cu.H3QKVGemmCudaOp()
    n = ref.QKV.out_features
    base = [_params(ref.QKV, seed=s)[0] for s in (1, 2, 3)]
    gq, gk, gv = (
        torch.randn(66, n, generator=torch.Generator("cpu").manual_seed(s)).bfloat16().cuda()
        for s in (4, 5, 6)
    )

    ws = [w.clone().requires_grad_() for w in base]
    x1 = _x(ref.QKV, 66).requires_grad_()
    q, k, v = op.forward(x1, *ws)
    torch.autograd.backward([q, k, v], [gq, gk, gv])

    wf = ref.fuse_qkv_weights(*base).clone().requires_grad_()
    x2 = _x(ref.QKV, 66).requires_grad_()
    fused = op.forward_fused(x2, wf)
    fused.backward(torch.cat([gq, gk, gv], dim=1))

    assert _bits(x1.grad) == _bits(x2.grad)
    assert _bits(torch.cat([w.grad for w in ws])) == _bits(wf.grad)


@requires_cuda
def test_qkv_tp_head_shards_equal_full_columns_and_backward_matches_golden():
    cu = _cuda_ops()
    op = cu.H3QKVGemmCudaOp()
    ws = [_params(ref.QKV, seed=s)[0] for s in (1, 2, 3)]
    x = _x(ref.QKV, 33)
    full = op.forward(x, *ws)
    for rank in range(4):  # TP4: 14 heads per rank
        sl = slice(rank * 1792, (rank + 1) * 1792)
        shard = op.forward(x, *[w[sl].contiguous() for w in ws])
        for s, f in zip(shard, full):
            assert _bits(s) == _bits(f[:, sl])
    # a missing output gradient is zero, not an error
    wq = ws[0].clone().requires_grad_()
    xg = x.clone().requires_grad_()
    q, k, v = op.forward(xg, wq, ws[1], ws[2])
    q.sum().backward()
    gold_dx = torch.ones(33, 7168, device="cuda", dtype=torch.float64) @ ws[0].double()
    assert _rel(xg.grad, gold_dx) < 2e-2
    assert wq.grad.shape == ws[0].shape


# --- CUDA: fail closed ----------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("row", sorted(ROWS))
def test_cuda_rejects_cpu_mixed_devices_and_wrong_dtype(row):
    spec, _ = ROWS[row]
    op = _cuda_op(row)
    w, b = _params(spec)
    x = _x(spec, 4)
    with pytest.raises(ValueError):
        op.forward(x.cpu(), w.cpu(), b.cpu())
    with pytest.raises(ValueError):
        op.forward(x, w.cpu(), b)
    other = torch.bfloat16 if spec.dtype == torch.float32 else torch.float32
    with pytest.raises(TypeError):
        op.forward(x.to(other), w, b)
    with pytest.raises(ValueError):
        op.forward(x, w, None)


@requires_cuda
def test_registry_dispatches_cuda():
    from rl_engine.backends.cuda.model_specific.minimax_h3 import projection as cu
    from rl_engine.runtime.registry import KernelRegistry

    _cuda_ops()
    registry = KernelRegistry()
    assert type(registry.get_op("audio_input_projection_fp32", device="cuda")) is (
        cu.H3AudioInputProjectionCudaOp
    )
    assert type(registry.get_op("h3_qkv_gemm", device="cuda")) is cu.H3QKVGemmCudaOp
