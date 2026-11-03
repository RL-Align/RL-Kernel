# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton fused MoE MLP kernels (rl_engine/kernels/ops/triton/moe/fused_mlp.py).

These are the portable counterparts of the CUDA kernels, written for ROCm.
They are deliberately NOT compared against the CUDA kernels byte for byte:
no Triton exponential reproduces nvcc's ``expf``, and ``tl.dot`` reduces a
BLOCK_K tile at a time rather than walking det_gemm's BF16 K-tree. What is
asserted instead:

* agreement with an FP32/oracle reference to within the MX noise floor;
* byte-level batch invariance, which is the property the contract actually
  requires and which must hold on any backend;
* byte-level TP invariance of the shared expert: sharded launches summed with
  the deterministic all-reduce's fixed tree equal the TP=1 launch.
"""

from __future__ import annotations

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

pytest.importorskip("triton", reason="Triton required")


@pytest.fixture(scope="module")
def tk():
    from rl_engine.kernels.ops.triton.moe import fused_mlp

    return fused_mlp


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.float(), want.float()
    return ((got - want).abs().max() / (want.abs().max() + 1e-30)).item()


# --- shared expert: fc1 + SwiGLU -------------------------------------------


def _shared_operands(t: int, h: int, f: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(t, h, generator=g) * 0.7).to(torch.bfloat16).cuda()
    w1 = (torch.randn(2 * f, h, generator=g) / h**0.5).to(torch.bfloat16).cuda()
    return x, w1


def _shared_reference(x: torch.Tensor, w1: torch.Tensor) -> torch.Tensor:
    """FP32 reference: no clamp and no route weight, per start-kit decision D6."""
    f = w1.shape[0] // 2
    z = x.float() @ w1.float().t()
    gate, up = z[:, :f], z[:, f:]
    return ((gate * torch.sigmoid(gate)) * up).to(torch.bfloat16)


@requires_cuda
@pytest.mark.parametrize("shape", [(128, 256, 64), (200, 512, 128), (1, 4096, 2048), (2048, 4096, 2048)])
def test_shared_matches_fp32_reference(tk, shape):
    x, w1 = _shared_operands(*shape)
    got = tk.fused_shared_fc1_swiglu(x, w1)
    assert got.dtype is torch.bfloat16 and got.shape == (shape[0], shape[2])
    # BF16 eps is ~7.8e-3; a reduction-order difference stays well inside it,
    # while a transposed operand or a swapped gate/up lands at O(1).
    assert _rel(got, _shared_reference(x, w1)) < 5e-3


@requires_cuda
def test_shared_batch_invariance(tk):
    x, w1 = _shared_operands(512, 4096, 2048, seed=1)
    full = tk.fused_shared_fc1_swiglu(x, w1)
    for t in (0, 1, 63, 64, 65, 300, 511):  # across the BLOCK_M = 64 boundary
        one = tk.fused_shared_fc1_swiglu(x[t : t + 1].contiguous(), w1)
        assert torch.equal(one[0], full[t]), f"row {t} diverged"
    assert torch.equal(tk.fused_shared_fc1_swiglu(x[100:164].contiguous(), w1), full[100:164])


@requires_cuda
def test_shared_run_to_run(tk):
    x, w1 = _shared_operands(256, 1024, 512, seed=2)
    first = tk.fused_shared_fc1_swiglu(x, w1)
    for _ in range(4):
        assert torch.equal(tk.fused_shared_fc1_swiglu(x, w1), first)


@requires_cuda
def test_shared_fails_closed(tk):
    x, w1 = _shared_operands(64, 128, 32)
    with pytest.raises(TypeError, match="must be BF16"):
        tk.fused_shared_fc1_swiglu(x.float(), w1)
    with pytest.raises(ValueError, match="K mismatch"):
        tk.fused_shared_fc1_swiglu(x, w1[:, :64].contiguous())
    with pytest.raises(ValueError, match="even number of rows"):
        tk.fused_shared_fc1_swiglu(x, w1[:-1].contiguous())


# --- shared expert: fc3, TP-invariant ---------------------------------------


def _fixed_tree(parts: list[torch.Tensor]) -> torch.Tensor:
    """The deterministic all-reduce's tree over 1/2/4/8 BF16 partials.

    ``fixed_tree_reduce`` in csrc/cuda/distributed/deterministic_collective.cu:
    adjacent pairs first, then pairs of pairs. A BF16 ``+`` on the GPU is an
    FP32 add rounded to nearest even, which is what ``__hadd`` does too.
    """
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
    return parts[0]


def _shard(t: torch.Tensor, rank: int, tp: int, dim: int) -> torch.Tensor:
    return t.chunk(tp, dim=dim)[rank].contiguous()


def _shard_w_fc1(w1: torch.Tensor, rank: int, tp: int) -> torch.Tensor:
    """Column-parallel fc1 shard: this rank's gate rows on top of its up rows."""
    f = w1.shape[0] // 2
    return torch.cat([_shard(w1[:f], rank, tp, 0), _shard(w1[f:], rank, tp, 0)])


@requires_cuda
@pytest.mark.parametrize("shape", [(128, 256, 64), (200, 512, 136), (1, 4096, 2048), (512, 4096, 2048)])
def test_shared_fc3_matches_fp32_reference(tk, shape):
    t, h, f = shape
    g = torch.Generator().manual_seed(10)
    hid = (torch.randn(t, f, generator=g)).to(torch.bfloat16).cuda()
    w2 = (torch.randn(h, f, generator=g) / f**0.5).to(torch.bfloat16).cuda()
    got = tk.shared_fc3(hid, w2)
    assert got.dtype is torch.bfloat16 and got.shape == (t, h)
    # Eight BF16 leaves and seven BF16 adds: a few BF16 ulps of the output.
    assert _rel(got, hid.float() @ w2.float().t()) < 2e-2


@requires_cuda
@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("f", [2048, 136])  # 136/8 = 17: leaves are not whole BK tiles
def test_shared_fc3_tp_invariance(tk, tp, f):
    """TP=tp partials, summed with the collective's tree, equal TP=1 byte for byte."""
    g = torch.Generator().manual_seed(11)
    hid = (torch.randn(300, f, generator=g)).to(torch.bfloat16).cuda()
    w2 = (torch.randn(512, f, generator=g) / f**0.5).to(torch.bfloat16).cuda()
    full = tk.shared_fc3(hid, w2)
    parts = [tk.shared_fc3(_shard(hid, r, tp, 1), _shard(w2, r, tp, 1), tp) for r in range(tp)]
    assert torch.equal(_fixed_tree(parts), full)


@requires_cuda
@pytest.mark.parametrize("tp", [2, 4, 8])
def test_shared_chain_tp_invariance(tk, tp):
    """The whole shared expert, sharded fc1 -> sharded fc3 -> tree, equals TP=1."""
    x, w1 = _shared_operands(130, 1024, 512, seed=12)
    g = torch.Generator().manual_seed(13)
    w2 = (torch.randn(1024, 512, generator=g) / 512**0.5).to(torch.bfloat16).cuda()
    h_full = tk.fused_shared_fc1_swiglu(x, w1)
    y_full = tk.shared_fc3(h_full, w2)
    parts = []
    for r in range(tp):
        h_r = tk.fused_shared_fc1_swiglu(x, _shard_w_fc1(w1, r, tp))
        assert torch.equal(h_r, _shard(h_full, r, tp, 1)), f"fc1 shard {r} diverged"
        parts.append(tk.shared_fc3(h_r, _shard(w2, r, tp, 1), tp))
    assert torch.equal(_fixed_tree(parts), y_full)


@requires_cuda
def test_shared_fc3_batch_invariance(tk):
    g = torch.Generator().manual_seed(14)
    hid = (torch.randn(512, 2048, generator=g)).to(torch.bfloat16).cuda()
    w2 = (torch.randn(4096, 2048, generator=g) / 2048**0.5).to(torch.bfloat16).cuda()
    for tp in (1, 4):
        h_r, w_r = _shard(hid, 0, tp, 1), _shard(w2, 0, tp, 1)
        full = tk.shared_fc3(h_r, w_r, tp)
        for t in (0, 1, 63, 64, 65, 127, 128, 129, 300, 511):
            assert torch.equal(tk.shared_fc3(h_r[t : t + 1], w_r, tp)[0], full[t]), f"row {t}"
        assert torch.equal(tk.shared_fc3(h_r[100:229], w_r, tp), full[100:229])


@requires_cuda
def test_shared_fc3_fails_closed(tk):
    hid = torch.zeros(4, 64, dtype=torch.bfloat16, device="cuda")
    w2 = torch.zeros(32, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(TypeError, match="must be BF16"):
        tk.shared_fc3(hid.float(), w2)
    with pytest.raises(ValueError, match="K mismatch"):
        tk.shared_fc3(hid, w2[:, :32].contiguous())
    with pytest.raises(ValueError, match="tp_size"):
        tk.shared_fc3(hid, w2, 3)
    with pytest.raises(ValueError, match="K-tree leaves"):
        tk.shared_fc3(hid[:, :60].contiguous(), w2[:, :60].contiguous())


# --- routed expert: fc1 + SwiGLU + quant, then fc3 -------------------------


def _routed_case(m: int, e: int, h: int, f: int, seed: int = 0):
    from rl_engine.moe.mx_format import mx_quantize

    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(m, h, generator=g) * 0.5).to(torch.bfloat16).cuda()
    w1 = (torch.randn(e, 2 * f, h, generator=g) / h**0.5).to(torch.bfloat16).cuda()
    w2 = (torch.randn(e, h, f, generator=g) / f**0.5).to(torch.bfloat16).cuda()
    p_s = torch.rand(m, generator=g).cuda()
    offsets = torch.tensor([0] + [m * (i + 1) // e for i in range(e)], dtype=torch.int32).cuda()
    return (
        mx_quantize(x, "e4m3"),
        mx_quantize(w1, "e2m1"),
        mx_quantize(w2, "e2m1"),
        offsets,
        p_s,
    )


@requires_cuda
def test_routed_matches_the_oracle_at_fixture_geometry(tk):
    """At fixture width the whole chain is byte-equal to the FP32 oracle.

    Not a general property -- the FP32 accumulator has headroom at H=128 that it
    runs out of at production width, the same way the P5-3 Triton path does --
    but it pins the recipe: the MX codes, the E8M0 scales, the clamp bounds and
    the association order all have to be exactly right for this to hold.
    """
    from rl_engine.moe import oracle
    from rl_engine.moe.mx_format import mx_quantize

    x_q, w1, w2, offsets, p_s = _routed_case(64, 2, 128, 64)
    z = oracle.mxfp8_mxfp4_grouped_gemm_fwd(x_q, w1, offsets)
    f = w2.shape[2]
    h_ref, _ = oracle.clamp_swiglu_weighted_fwd(z[:, :f], z[:, f:], p_s)
    q2 = mx_quantize(h_ref, "e4m3")
    y_ref = oracle.mxfp8_mxfp4_grouped_gemm_fwd(q2, w2, offsets).to(torch.bfloat16)

    h_q = tk.routed_fc1_swiglu_quant(x_q, w1, offsets, p_s)
    assert torch.equal(h_q.codes, q2.codes), "h codes diverged from the oracle"
    assert torch.equal(h_q.scales, q2.scales), "h scales diverged from the oracle"
    assert torch.equal(tk.routed_fc3(h_q, w2, offsets), y_ref)


@requires_cuda
def test_routed_matches_fp64_at_production_width(tk):
    """At H=4096 the gap is the MX quantization floor, not an error."""
    from rl_engine.moe.mx_format import mx_dequantize

    x_q, w1, w2, offsets, p_s = _routed_case(512, 8, 4096, 2048, seed=3)
    y = tk.routed_mlp_forward(x_q, w1, w2, offsets, p_s)

    a, w1d, w2d = mx_dequantize(x_q).double(), mx_dequantize(w1).double(), mx_dequantize(w2).double()
    f = w2.shape[2]
    ref = torch.zeros_like(y, dtype=torch.float64)
    offs = offsets.tolist()
    for e in range(len(offs) - 1):
        lo, hi = offs[e], offs[e + 1]
        if lo == hi:
            continue
        z = a[lo:hi] @ w1d[e].t()
        gate = z[:, :f].clamp(max=10.0)
        up = z[:, f:].clamp(-10.0, 10.0)
        h = ((gate * torch.sigmoid(gate)) * up) * p_s[lo:hi, None].double()
        ref[lo:hi] = h @ w2d[e].t()
    # E4M3 carries 4 significand bits, so ~6% relative per element; the measured
    # max over the tensor sits near 4e-2 for both this and the CUDA kernel.
    assert _rel(y, ref) < 8e-2


@requires_cuda
def test_routed_batch_invariance(tk):
    """Every boundary, byte-for-byte, with rows on both sides of an expert edge."""
    from rl_engine.moe.mx_format import MXTensor

    m, e, h = 256, 2, 128
    x_q, w1, w2, _, p_s = _routed_case(m, e, h, 64, seed=4)
    offsets = torch.tensor([0, 100, m], dtype=torch.int32).cuda()
    h_full = tk.routed_fc1_swiglu_quant(x_q, w1, offsets, p_s)
    y_full = tk.routed_fc3(h_full, w2, offsets)

    for t in (0, 1, 63, 64, 99, 100, 101, 255):
        expert = int((offsets[1:] <= t).sum().item())
        one_off = torch.tensor(
            [0] * (expert + 1) + [1] * (e - expert), dtype=torch.int32, device="cuda"
        )
        x_one = MXTensor(
            codes=x_q.codes[t : t + 1].contiguous(),
            scales=x_q.scales[t : t + 1].contiguous(),
            elem_format="e4m3",
            shape=(1, h),
        )
        h_one = tk.routed_fc1_swiglu_quant(x_one, w1, one_off, p_s[t : t + 1].contiguous())
        assert torch.equal(h_one.codes[0], h_full.codes[t]), f"h codes, row {t}"
        assert torch.equal(h_one.scales[0], h_full.scales[t]), f"h scales, row {t}"
        assert torch.equal(tk.routed_fc3(h_one, w2, one_off)[0], y_full[t]), f"y, row {t}"


@requires_cuda
def test_routed_run_to_run(tk):
    x_q, w1, w2, offsets, p_s = _routed_case(128, 2, 256, 64, seed=5)
    first = tk.routed_mlp_forward(x_q, w1, w2, offsets, p_s)
    for _ in range(3):
        assert torch.equal(tk.routed_mlp_forward(x_q, w1, w2, offsets, p_s), first)


@requires_cuda
def test_routed_handles_empty_experts(tk):
    """An expert with no rows must not shift any other expert's output."""
    m, e, h, f = 192, 4, 128, 64
    x_q, w1, w2, _, p_s = _routed_case(m, e, h, f, seed=6)
    dense = torch.tensor([0, 64, 128, 160, m], dtype=torch.int32).cuda()
    empty = torch.tensor([0, 64, 64, 160, m], dtype=torch.int32).cuda()
    y_dense = tk.routed_mlp_forward(x_q, w1, w2, dense, p_s)
    y_empty = tk.routed_mlp_forward(x_q, w1, w2, empty, p_s)
    assert torch.isfinite(y_empty).all()
    # Rows 0..63 use expert 0 in both layouts, so they must not move.
    assert torch.equal(y_dense[:64], y_empty[:64])


@requires_cuda
def test_routed_many_experts_match_per_expert_launches(tk):
    """DSv4-style expert count with ragged and empty experts.

    The program -> expert map is a binary search over a device-side block
    prefix; each expert's rows must equal a launch that holds only that expert.
    """
    from rl_engine.moe.mx_format import MXTensor

    e, h, f = 37, 128, 64
    g = torch.Generator().manual_seed(15)
    counts = torch.randint(0, 150, (e,), generator=g)
    counts[[0, 5, 6, 36]] = 0  # empty at the front, in a run, and at the back
    counts[7] = 1
    m = int(counts.sum())
    x_q, w1, w2, _, p_s = _routed_case(m, e, h, f, seed=16)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]).int().cuda()
    h_all = tk.routed_fc1_swiglu_quant(x_q, w1, offsets, p_s)
    y_all = tk.routed_fc3(h_all, w2, offsets)

    def rows(t: MXTensor, lo: int, hi: int) -> MXTensor:
        return MXTensor(codes=t.codes[lo:hi].contiguous(), scales=t.scales[lo:hi].contiguous(),
                        elem_format=t.elem_format, shape=(hi - lo, t.shape[1]))

    def expert(t: MXTensor, i: int) -> MXTensor:
        return MXTensor(codes=t.codes[i : i + 1].contiguous(), scales=t.scales[i : i + 1].contiguous(),
                        elem_format=t.elem_format, shape=(1, *t.shape[1:]))

    offs = offsets.tolist()
    for i in range(e):
        lo, hi = offs[i], offs[i + 1]
        if lo == hi:
            continue
        one = torch.tensor([0, hi - lo], dtype=torch.int32, device="cuda")
        h_one = tk.routed_fc1_swiglu_quant(rows(x_q, lo, hi), expert(w1, i), one, p_s[lo:hi].contiguous())
        assert torch.equal(h_one.codes, h_all.codes[lo:hi]), f"expert {i} h codes"
        assert torch.equal(tk.routed_fc3(h_one, expert(w2, i), one), y_all[lo:hi]), f"expert {i} y"


def _require_fp8(tk):
    if not tk.fp8_available(torch.device("cuda")):
        pytest.skip("native FP8 lane needs e4m3fnuz MFMA (gfx94x)")


@requires_cuda
def test_routed_fp8_lane_matches_fp64_and_is_batch_invariant(tk):
    """Opt-in FP8 lane: MX-floor accuracy, and byte-level batch invariance."""
    from rl_engine.moe.mx_format import MXTensor, mx_dequantize

    _require_fp8(tk)
    x_q, w1, w2, offsets, p_s = _routed_case(512, 8, 4096, 2048, seed=17)
    r1, r2 = tk.prepare_mxfp4_fp8(w1), tk.prepare_mxfp4_fp8(w2)
    y = tk.routed_mlp_forward(x_q, w1, w2, offsets, p_s, fp8=True, w1_ref=r1, w2_ref=r2)

    a, w1d, w2d = mx_dequantize(x_q).double(), mx_dequantize(w1).double(), mx_dequantize(w2).double()
    f = w2.shape[2]
    ref = torch.zeros_like(y, dtype=torch.float64)
    offs = offsets.tolist()
    for e in range(len(offs) - 1):
        lo, hi = offs[e], offs[e + 1]
        z = a[lo:hi] @ w1d[e].t()
        h = (torch.nn.functional.silu(z[:, :f].clamp(max=10.0)) * z[:, f:].clamp(-10.0, 10.0))
        ref[lo:hi] = (h * p_s[lo:hi, None].double()) @ w2d[e].t()
    assert _rel(y, ref) < 8e-2

    for t in (0, 63, 64, 300, 511):
        e = int((offsets[1:] <= t).sum().item())
        one = torch.tensor([0] * (e + 1) + [1] * (8 - e), dtype=torch.int32, device="cuda")
        x_one = MXTensor(codes=x_q.codes[t : t + 1].contiguous(), scales=x_q.scales[t : t + 1].contiguous(),
                         elem_format="e4m3", shape=(1, 4096))
        y_one = tk.routed_mlp_forward(x_one, w1, w2, one, p_s[t : t + 1].contiguous(),
                                      fp8=True, w1_ref=r1, w2_ref=r2)
        assert torch.equal(y_one[0], y[t]), f"row {t}"


@requires_cuda
def test_routed_fp8_lane_handles_negative_zero_and_subnormals(tk):
    """0x80 (-0 in fn, NaN in fnuz) and the fn subnormals must decode exactly."""
    from rl_engine.moe import oracle
    from rl_engine.moe.mx_format import MXTensor

    _require_fp8(tk)
    _, w1, w2, offsets, p_s = _routed_case(64, 2, 128, 64, seed=18)
    codes = torch.randint(0, 256, (64, 128), dtype=torch.uint8, device="cuda")
    codes[(codes & 0x7F) == 0x7F] = 0x80  # drop fn NaN codes; plenty of -0 and subnormals
    x_q = MXTensor(codes=codes, scales=torch.full((64, 4), 120, dtype=torch.uint8, device="cuda"),
                   elem_format="e4m3", shape=(64, 128))
    got = tk.routed_fc1_swiglu_quant(x_q, w1, offsets, p_s, fp8=True)
    want = tk.routed_fc1_swiglu_quant(x_q, w1, offsets, p_s, fp8=False)
    assert torch.isfinite(oracle.mxfp8_mxfp4_grouped_gemm_fwd(got, w2, offsets)).all()
    # Different MFMA rounding, so not byte-equal; but no NaN and the same MX floor.
    from rl_engine.moe.mx_format import mx_dequantize

    assert _rel(mx_dequantize(got), mx_dequantize(want)) < 8e-2


@requires_cuda
def test_prepare_mxfp4_fp8_fails_closed(tk):
    from rl_engine.moe.mx_format import MXTensor

    codes = torch.zeros(1, 4, 64, dtype=torch.uint8, device="cuda")
    scales = torch.full((1, 4, 4), 120, dtype=torch.uint8, device="cuda")
    ok = MXTensor(codes=codes, scales=scales, elem_format="e2m1", shape=(1, 4, 128))
    assert tk.prepare_mxfp4_fp8(ok).tolist() == [[115] * 4]  # max - 5
    wide = scales.clone()
    wide[0, 0, 0] = 120 - 15  # 15 binades below the column max
    with pytest.raises(ValueError, match="span"):
        tk.prepare_mxfp4_fp8(MXTensor(codes=codes, scales=wide, elem_format="e2m1", shape=(1, 4, 128)))


@requires_cuda
def test_routed_fails_closed(tk):
    x_q, w1, w2, offsets, p_s = _routed_case(64, 2, 128, 64)
    with pytest.raises(ValueError, match="K mismatch"):
        tk.routed_fc3(x_q, w2, offsets)  # x_q has K=H, w2 expects K=F
    with pytest.raises(ValueError, match="K mismatch"):
        tk.routed_fc1_swiglu_quant(x_q, w2, offsets, p_s)


# --- provider wiring -------------------------------------------------------


@requires_cuda
def test_routed_provider_matches_the_kernels(tk):
    from dataclasses import replace

    from rl_engine.moe.backends import TritonFusedMoeMlp
    from rl_engine.moe.contract import ExpertBatch

    x_q, w1, w2, offsets, p_s = _routed_case(128, 2, 256, 64, seed=7)
    provider = TritonFusedMoeMlp()
    batch = ExpertBatch(
        x=torch.zeros(128, 256, dtype=torch.bfloat16, device="cuda"),
        w1=w1,
        w2=w2,
        expert_offsets=offsets,
        p_s=p_s,
        lora=None,
        output_slot=torch.arange(128, dtype=torch.int32, device="cuda"),
        numeric_profile=provider.numeric_profile,
    )
    assert torch.equal(
        provider.forward(batch, x_q), tk.routed_mlp_forward(x_q, w1, w2, offsets, p_s)
    )
    assert provider.provenance()["batch_invariant"] is True
    with pytest.raises(NotImplementedError, match="fail-closed"):
        provider.forward(replace(batch, numeric_profile="oracle-fp32-serial-v1"), x_q)
    with pytest.raises(ValueError, match="two_launch"):
        provider.forward(batch, x_q, path="fused")


@requires_cuda
def test_shared_provider_is_forward_only(tk):
    from rl_engine.moe.backends import TritonFusedSharedExpertProvider
    from rl_engine.moe.contract import SharedBatch

    provider = TritonFusedSharedExpertProvider()
    gen = torch.Generator().manual_seed(8)
    t, h, f = 128, 256, 64
    batch = SharedBatch(
        x=(torch.randn(t, h, generator=gen) * 0.7).to(torch.bfloat16).cuda(),
        w_fc1=(torch.randn(2 * f, h, generator=gen) / h**0.5).to(torch.bfloat16).cuda(),
        w_fc2=(torch.randn(h, f, generator=gen) / f**0.5).to(torch.bfloat16).cuda(),
        numeric_profile=provider.numeric_profile,
    )
    y, saved = provider.shared_expert_mlp_fwd(batch)
    assert y.shape == (t, h) and y.dtype is torch.bfloat16
    assert "z32" not in saved, "the fused kernel must not materialize z"
    assert torch.equal(saved["h_bf16"], tk.fused_shared_fc1_swiglu(batch.x, batch.w_fc1))
    assert torch.equal(y, tk.shared_fc3(saved["h_bf16"], batch.w_fc2))
    assert provider.provenance()["tp_equivalent"] is True
    with pytest.raises(NotImplementedError, match="forward-only"):
        provider.shared_expert_mlp_bwd(y.float(), batch, saved)
