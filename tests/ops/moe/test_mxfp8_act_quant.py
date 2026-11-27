# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bit-wise alignment tests for P5-1 ``mxfp8_act_quant``.

Both backends (CUDA and Triton) must reproduce the start-kit oracle
(``rl_engine.moe.mx_format.mx_quantize(x, "e4m3")``) byte for byte — not
within a tolerance — because the P5 contract requires train/infer byte
equality on one numeric profile. The tests therefore compare raw ``uint8``
codes and E8M0 scale bytes, the committed golden manifest, and the two
backends against each other.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.moe import fixtures
from rl_engine.moe.contract import tensor_sha256
from rl_engine.moe.mx_format import MX_BLOCK, mx_quantize
from rl_engine.moe.oracle import mxfp8_act_quant_bwd, mxfp8_act_quant_fwd

_GPU_OK = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
pytestmark = pytest.mark.skipif(not _GPU_OK, reason="mxfp8_act_quant needs an E4M3 GPU (SM89+)")

DEV = "cuda"


def _backends():
    backends = []
    from rl_engine.kernels.ops.cuda.moe import mxfp8_act_quant_bwd_cuda, mxfp8_act_quant_fwd_cuda

    # The CUDA backend needs either the AOT rl_engine._C symbols or a working
    # JIT toolchain (nvcc + ninja); without both it is skipped, not failed.
    cuda_error = None
    if _GPU_OK:
        try:
            mxfp8_act_quant_fwd_cuda(torch.zeros(1, MX_BLOCK, device=DEV, dtype=torch.bfloat16))
        except RuntimeError as exc:
            cuda_error = str(exc).splitlines()[0]
        else:
            backends.append(("cuda", mxfp8_act_quant_fwd_cuda, mxfp8_act_quant_bwd_cuda))
    return backends, cuda_error


BACKENDS, _CUDA_BACKEND_ERROR = _backends()
BACKEND_IDS = [name for name, _, _ in BACKENDS]
_HAS_CUDA_BACKEND = "cuda" in BACKEND_IDS
FWD = [(name, fwd) for name, fwd, _ in BACKENDS]
BWD = [(name, bwd) for name, _, bwd in BACKENDS]


def _assert_byte_equal(got, want, what: str) -> None:
    assert got.codes.dtype == torch.uint8 and got.scales.dtype == torch.uint8
    assert got.shape == want.shape, what
    assert torch.equal(got.scales, want.scales), f"{what}: E8M0 scale bytes differ"
    assert torch.equal(got.codes, want.codes), f"{what}: E4M3 element bytes differ"


def _wide_range(rows: int, cols: int, device: str = "cpu", seed: int = 7) -> torch.Tensor:
    """Random values spanning ~2**+-60 so scale selection is exercised hard."""
    g = torch.Generator(device=device).manual_seed(seed)
    mant = torch.randn(rows, cols, device=device, generator=g)
    exp = torch.randint(-60, 60, (rows, cols), device=device, generator=g).float()
    return (mant * torch.exp2(exp)).to(DEV)


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 32), torch.bfloat16),
        ((1, 128), torch.bfloat16),
        ((24, 128), torch.bfloat16),
        ((37, 64), torch.bfloat16),
        ((512, 4096), torch.bfloat16),
        ((24, 128), torch.float32),
        ((24, 128), torch.float16),
        ((4, 8, 64), torch.bfloat16),
    ],
)
def test_forward_bitwise_matches_oracle(name, fwd, shape, dtype):
    torch.manual_seed(0)
    x = (torch.randn(*shape, device=DEV) * 3.0).to(dtype)
    _assert_byte_equal(fwd(x), mx_quantize(x, "e4m3"), f"{name} {shape} {dtype}")


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_forward_bitwise_edge_fixtures(name, fwd):
    """Start-kit edge inputs: powers of two, RNE ties, zero rows, clamp bounds."""
    x = fixtures.make_act_quant_edge_inputs().to(DEV)
    _assert_byte_equal(fwd(x), mxfp8_act_quant_fwd(x), f"{name} edge fixtures")


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_forward_matches_golden_manifest(name, fwd):
    """The committed CPU-anchored golden hashes, recomputed by the kernels."""
    golden = fixtures.load_manifest()["cases"]["act_quant_edges"]
    got = fwd(fixtures.make_act_quant_edge_inputs().to(DEV))
    assert tensor_sha256(got.codes) == golden["act_quant.codes"]
    assert tensor_sha256(got.scales) == golden["act_quant.scales"]


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
@pytest.mark.parametrize(
    "case",
    ["zeros", "wide_range", "subnormal", "saturating", "all_negative"],
)
def test_forward_bitwise_special_values(name, fwd, case):
    if case == "zeros":
        x = torch.zeros(3, 64, device=DEV, dtype=torch.bfloat16)
    elif case == "wide_range":
        x = _wide_range(64, 128)
    elif case == "subnormal":
        # amax below FLT_MIN: the oracle clamps before floor(log2(.)).
        x = torch.full((4, 32), 1e-40, device=DEV, dtype=torch.float32)
        x[1] = 0.0
    elif case == "saturating":
        # mantissa > 1.75 makes x / scale exceed 448 -> satfinite clamp fires.
        x = torch.full((2, 32), 1.9, device=DEV, dtype=torch.float32)
        x[1, 0] = 1.99
    else:
        x = -(torch.rand(4, 64, device=DEV, dtype=torch.float32) + 0.5)
    _assert_byte_equal(fwd(x), mx_quantize(x, "e4m3"), f"{name} {case}")


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_forward_is_batch_invariant(name, fwd):
    """A row's bytes never depend on how many rows are quantized with it.

    The amax reduction is row-local by contract, so one row alone, the same row
    inside a large batch, and a non-contiguous slice of it must agree bitwise.
    """
    torch.manual_seed(1)
    batch = (torch.randn(257, 256, device=DEV) * 2.0).to(torch.bfloat16)
    full = fwd(batch)
    for row in (0, 1, 42, 256):
        single = fwd(batch[row : row + 1])
        assert torch.equal(single.codes[0], full.codes[row]), f"{name} row {row} codes"
        assert torch.equal(single.scales[0], full.scales[row]), f"{name} row {row} scales"
    strided = fwd(batch[::2])
    assert torch.equal(strided.codes, full.codes[::2]), f"{name} strided codes"


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_forward_is_deterministic(name, fwd):
    x = _wide_range(128, 512).to(torch.bfloat16)
    first = fwd(x)
    for _ in range(5):
        again = fwd(x)
        assert torch.equal(again.codes, first.codes)
        assert torch.equal(again.scales, first.scales)


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_matches_torchao_reference_cast(name, fwd):
    """Third-party cross-check: torchao's OCP-MX cast in FLOOR scaling mode.

    torchao derives the E8M0 scale independently of this repo, so agreeing with
    it byte for byte validates the P5 recipe itself, not just our own oracle.
    """
    to_mx = pytest.importorskip(
        "torchao.prototype.mx_formats.mx_tensor", reason="torchao is an optional baseline"
    ).to_mx
    x = _wide_range(64, 256).to(torch.bfloat16)
    scales, codes = to_mx(x, torch.float8_e4m3fn, MX_BLOCK)
    got = fwd(x)
    assert torch.equal(got.codes, codes.view(torch.uint8))
    assert torch.equal(got.scales, scales.view(torch.uint8).reshape(got.scales.shape))


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_matches_triton_kernels_reference_cast(name, fwd):
    """Cross-check against the MX quantizer vLLM runs for its MXFP4 path.

    ``triton_kernels`` ships with the Triton repo; ROUND_DOWN is the scale mode
    that matches the P5 floor(log2(amax)) recipe.
    """
    mxfp = pytest.importorskip(
        "triton_kernels.numerics_details.mxfp", reason="triton_kernels is an optional baseline"
    )
    x = _wide_range(64, 256).to(torch.bfloat16)
    codes, scales = mxfp.downcast_to_mxfp(
        x,
        torch.float8_e4m3fn,
        axis=-1,
        DEQUANT_SCALE_ROUNDING_MODE=mxfp.DequantScaleRoundingMode.ROUND_DOWN,
    )
    got = fwd(x)
    assert torch.equal(got.codes, codes.view(torch.uint8))
    assert torch.equal(got.scales, scales.view(torch.uint8).reshape(got.scales.shape))


@pytest.mark.parametrize("name,bwd", BWD, ids=BACKEND_IDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_backward_is_straight_through(name, bwd, dtype):
    """STE: dX = dY exactly for any floating dtype (P5-1 spec contract table)."""
    dy = (torch.randn(64, 128, device=DEV) * 4.0).to(dtype)
    dx = bwd(dy)
    assert dx.dtype == dy.dtype
    assert torch.equal(dx, dy)
    assert torch.equal(dx, mxfp8_act_quant_bwd(dy))


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_forward_is_fail_closed_on_non_finite(name, fwd, bad):
    x = torch.randn(2, 64, device=DEV, dtype=torch.float32)
    x[1, 5] = bad
    with pytest.raises(ValueError, match="non-finite"):
        fwd(x)


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_forward_rejects_bad_shapes_and_devices(name, fwd):
    with pytest.raises(ValueError, match="MX block"):
        fwd(torch.randn(2, MX_BLOCK - 1, device=DEV, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="CUDA"):
        fwd(torch.randn(2, MX_BLOCK, dtype=torch.bfloat16))
    with pytest.raises(TypeError):
        fwd(torch.randn(2, MX_BLOCK, device=DEV, dtype=torch.float64))


def test_cuda_kernel_is_fast_math_immune():
    """A --use_fast_math build must emit the same bytes as the oracle.

    nvcc turns fmaxf/fabsf/__fdiv_rn into .ftz forms under fast-math; the
    kernel keeps every subnormal-sensitive step on integer bit patterns or
    explicit non-.ftz PTX, so subnormal inputs and the 2**-127 scale survive.
    """
    if not _HAS_CUDA_BACKEND:
        pytest.skip(f"CUDA backend unavailable: {_CUDA_BACKEND_ERROR}")
    from torch.utils.cpp_extension import load

    from rl_engine.kernels.ops.cuda.moe import mxfp8_act_quant as wrapper

    # No build_directory: torch caches the build by name and only recompiles
    # when the source or flags change, instead of ~50 s of nvcc per session.
    module = load(
        name="rl_engine_p5_mxfp8_act_quant_fastmath",
        sources=[str(wrapper._CU_SOURCE)],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-DRL_KERNEL_P5_STANDALONE"],
        verbose=False,
    )
    cases = [
        torch.full((4, 32), 1e-40, device=DEV, dtype=torch.float32),  # subnormal amax
        _wide_range(64, 128),
        fixtures.make_act_quant_edge_inputs().to(DEV),
    ]
    for x in cases:
        ref = mx_quantize(x, "e4m3")
        codes, scales, flag = module.mxfp8_act_quant_forward(x.contiguous(), True)
        assert int(flag.item()) == 0
        assert torch.equal(scales, ref.scales), "fast-math build changed E8M0 scale bytes"
        assert torch.equal(codes, ref.codes), "fast-math build changed E4M3 element bytes"


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
@pytest.mark.parametrize("shape", [(4096, 7168), (16384, 7168)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_forward_bitwise_at_production_shapes(name, fwd, shape, dtype):
    """The benchmark shapes (DeepSeek-V3 hidden size), asserted byte-equal, not just timed.

    Random exponents over 2**+-60 so every tile takes a different scale path.
    """
    x = _wide_range(*shape, device=DEV, seed=11).to(dtype)
    _assert_byte_equal(fwd(x), mx_quantize(x, "e4m3"), f"{name} {shape} {dtype}")
