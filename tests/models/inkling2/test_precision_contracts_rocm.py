# SPDX-License-Identifier: Apache-2.0
"""INK-04/05/30/31/52/67/69 precision contracts executed on the ROCm HIP device."""

import pytest
import torch

from rl_engine.models.inkling2.precision_contracts import (
    PrecisionContractError,
    compare_checkpoint_fingerprints,
    dequantize_per_tensor,
    fingerprint_checkpoint,
    quantize_per_tensor,
    quantized_linear,
)
from rl_engine.models.inkling2.precision_contracts_rocm import (
    FP8_FNUZ_FORMATS,
    PrecisionProfile,
    RocmRuntime,
    build_rocm_quant_matrix,
    detect_rocm_runtime,
    fingerprint_bf16_dequant_checkpoint,
    is_strict,
    profile_for_formats,
    quantize_per_tensor_rocm,
    quantized_linear_rocm,
    require_strict,
)

HIP = getattr(torch.version, "hip", None) is not None and torch.cuda.is_available()
needs_hip = pytest.mark.skipif(not HIP, reason="ROCm GPU is required")
DEV = "cuda"


def _fake_runtime(arch: str) -> RocmRuntime:
    return RocmRuntime("7.2.0", 8, "fake", arch, 192 * 2**30, True)


# ---------------------------------------------------------------- runtime / matrix
@needs_hip
def test_runtime_readback_matches_torch():
    rt = detect_rocm_runtime()
    assert rt.is_rocm and rt.hip_version == torch.version.hip
    assert rt.device_count == torch.cuda.device_count() >= 1
    assert rt.gcn_arch_name == torch.cuda.get_device_properties(0).gcnArchName
    assert rt.arch.startswith("gfx")
    assert rt.total_memory_bytes == torch.cuda.get_device_properties(0).total_memory
    assert rt.rccl_available == torch.distributed.is_nccl_available()


@needs_hip
def test_matrix_is_derived_from_the_real_arch():
    rt = detect_rocm_runtime()
    matrix = build_rocm_quant_matrix(rt)
    for entry in matrix.entries:
        assert entry.platform in {"rocm:gfx942", "rocm:gfx950"}
        if entry.available:
            assert entry.platform == f"rocm:{rt.arch}"
            assert entry.reason == ""
        else:
            assert entry.reason
    if rt.arch == "gfx942":
        for fmt in ("int8_symmetric", *FP8_FNUZ_FORMATS):
            cap = matrix.require(platform="rocm:gfx942", format=fmt, purpose="inference")
            assert cap.available
        for fmt in ("float8_e4m3fn", "mxfp8", "mxfp4", "nvfp4"):
            with pytest.raises(RuntimeError, match="gated"):
                matrix.require(platform="rocm:gfx942", format=fmt, purpose="inference")
        # gfx950 is declared but not present on this host.
        for fmt in ("float8_e4m3fn", "float8_e5m2", "mxfp8", "mxfp4"):
            with pytest.raises(RuntimeError, match="not present on this host"):
                matrix.require(platform="rocm:gfx950", format=fmt, purpose="inference")


def test_matrix_gfx950_host_declares_ocp_and_mx_but_never_nvfp4():
    matrix = build_rocm_quant_matrix(_fake_runtime("gfx950"), fnuz_probe=False)
    for fmt in ("float8_e4m3fn", "float8_e5m2", "mxfp8", "mxfp4", "int8_symmetric"):
        assert matrix.require(platform="rocm:gfx950", format=fmt, purpose="inference").available
    with pytest.raises(RuntimeError, match="NVFP4"):
        matrix.require(platform="rocm:gfx950", format="nvfp4", purpose="inference")
    # gfx942 entries are gated when the host is gfx950.
    with pytest.raises(RuntimeError, match="not present on this host"):
        matrix.require(platform="rocm:gfx942", format="float8_e4m3fnuz", purpose="inference")


def test_matrix_gfx942_fnuz_requires_a_passing_probe_and_no_device_fails_closed():
    rt = _fake_runtime("gfx942")
    ok = build_rocm_quant_matrix(rt, fnuz_probe=True)
    assert ok.require(platform="rocm:gfx942", format="float8_e5m2fnuz", purpose="training")
    bad = build_rocm_quant_matrix(rt, fnuz_probe=False)
    with pytest.raises(RuntimeError, match="probe failed"):
        bad.require(platform="rocm:gfx942", format="float8_e4m3fnuz", purpose="inference")
    none = build_rocm_quant_matrix(RocmRuntime(None, 0, "", "", 0, False))
    assert not any(entry.available for entry in none.entries)
    with pytest.raises(RuntimeError, match="no ROCm device detected"):
        none.require(platform="rocm:gfx942", format="int8_symmetric", purpose="training")


@needs_hip
def test_ocp_fp8_is_refused_by_the_quantizer_on_gfx942():
    rt = detect_rocm_runtime()
    if rt.arch != "gfx942":
        pytest.skip("gfx942 specific")
    x = torch.randn(8, device=DEV)
    for fmt in ("float8_e4m3fn", "mxfp8", "nvfp4"):
        with pytest.raises(RuntimeError, match="gated"):
            quantize_per_tensor_rocm(x, format=fmt)
    with pytest.raises(PrecisionContractError):
        quantize_per_tensor_rocm(x, format="bogus")


# ---------------------------------------------------------------- strictness rule
def test_mixed_and_approx_are_never_strict():
    assert profile_for_formats(()) is PrecisionProfile.STRICT
    assert profile_for_formats(("bf16", "fp32")) is PrecisionProfile.STRICT
    assert profile_for_formats(("bf16", "int8_symmetric")) is PrecisionProfile.MIXED
    assert profile_for_formats(("float8_e4m3fnuz",)) is PrecisionProfile.MIXED
    assert profile_for_formats(("bf16", "mxfp8")) is PrecisionProfile.APPROX
    assert profile_for_formats(("nvfp4",)) is PrecisionProfile.APPROX
    assert is_strict(PrecisionProfile.STRICT)
    for profile in (PrecisionProfile.MIXED, PrecisionProfile.APPROX):
        assert not is_strict(profile)
        with pytest.raises(PrecisionContractError, match="not strict"):
            require_strict(profile)
    require_strict(PrecisionProfile.STRICT)


# ---------------------------------------------------------------- fingerprints
@needs_hip
def test_bf16_checkpoint_fingerprint_device_equals_cpu():
    gen = torch.Generator().manual_seed(4)
    cpu = {
        "embed": torch.randn(33, 65, generator=gen).bfloat16(),
        "odd": torch.randn(7, generator=gen).bfloat16(),
        "special": torch.tensor(
            [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 1e-40, 3e38]
        ).bfloat16(),
        "fp32": torch.randn(5, 3, generator=gen),
        "fp16": torch.randn(9, generator=gen).half(),
    }
    dev = {k: v.to(DEV) for k, v in cpu.items()}
    a = fingerprint_checkpoint(cpu, source="s", revision="r")
    b = fingerprint_checkpoint(dev, source="s", revision="r")
    assert a.identity_sha256 == b.identity_sha256
    assert compare_checkpoint_fingerprints(a, b) == ()
    # A one-bit flip on the device is detected.
    flipped = {k: v.clone() for k, v in dev.items()}
    flipped["odd"].view(torch.int16)[0] ^= 1
    assert compare_checkpoint_fingerprints(
        a, fingerprint_checkpoint(flipped, source="s", revision="r")
    ) == ("tensor fingerprint differs: odd",)


@needs_hip
@pytest.mark.parametrize("fmt", ["int8_symmetric", *FP8_FNUZ_FORMATS])
def test_bf16_dequant_fingerprint_is_device_independent(fmt):
    torch.manual_seed(31)
    quantized = {
        f"layer{i}.weight": quantize_per_tensor_rocm(torch.randn(17 + i, 40), format=fmt)
        for i in range(3)
    }
    cpu_fp = fingerprint_bf16_dequant_checkpoint(quantized, source="miles", revision="v1")
    dev_fp = fingerprint_bf16_dequant_checkpoint(
        quantized, source="miles", revision="v1", device=DEV
    )
    assert cpu_fp.identity_sha256 == dev_fp.identity_sha256
    assert all(t.dtype == "torch.bfloat16" for t in dev_fp.tensors)


# ---------------------------------------------------------------- round trips
@needs_hip
def test_int8_per_tensor_device_equals_cpu_bitwise_and_exact_values():
    torch.manual_seed(52)
    x = torch.randn(64, 96)
    cpu_q = quantize_per_tensor_rocm(x, format="int8_symmetric")
    dev_q = quantize_per_tensor_rocm(x.to(DEV), format="int8_symmetric")
    assert dev_q.values.is_cuda and dev_q.values.dtype == torch.int8
    assert torch.equal(dev_q.values.cpu(), cpu_q.values)
    assert dev_q.scale.cpu().view(torch.int32) == cpu_q.scale.view(torch.int32)
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        assert torch.equal(
            dequantize_per_tensor(dev_q, dtype=dt).cpu(), dequantize_per_tensor(cpu_q, dtype=dt)
        )
    # Exact expectation: amax == 127 gives scale 1.0 and integers round trip exactly.
    ints = torch.tensor([127.0, -127.0, 0.0, 5.0, -64.0, 1.0], device=DEV)
    q = quantize_per_tensor_rocm(ints, format="int8_symmetric")
    assert q.scale.item() == 1.0
    assert torch.equal(dequantize_per_tensor(q, dtype=torch.float32), ints)


@needs_hip
@pytest.mark.parametrize(
    "fmt,exact",
    [
        ("float8_e4m3fnuz", [240.0, -120.0, 60.0, 30.0, 7.5, 0.0625, 0.0]),
        ("float8_e5m2fnuz", [57344.0, -28672.0, 7168.0, 1.5, 0.25, 0.0]),
    ],
)
def test_fp8_fnuz_per_tensor_round_trip_exact_and_cpu_parity(fmt, exact):
    dtype = getattr(torch, fmt)
    x = torch.tensor(exact, device=DEV)
    q = quantize_per_tensor_rocm(x, format=fmt)
    assert q.values.dtype == dtype and q.values.is_cuda
    assert q.scale.item() == 1.0  # amax equals the format max
    assert torch.equal(dequantize_per_tensor(q, dtype=torch.float32), x)
    assert not torch.isnan(q.values.float()).any()
    # Random data: the HIP result equals the CPU result byte for byte.
    torch.manual_seed(67)
    r = torch.randn(48, 80) * 3
    cpu_q = quantize_per_tensor_rocm(r, format=fmt)
    dev_q = quantize_per_tensor_rocm(r.to(DEV), format=fmt)
    assert torch.equal(dev_q.values.cpu().view(torch.uint8), cpu_q.values.view(torch.uint8))
    assert dev_q.scale.cpu().view(torch.int32) == cpu_q.scale.view(torch.int32)
    assert torch.equal(
        dequantize_per_tensor(dev_q, dtype=torch.bfloat16).cpu(),
        dequantize_per_tensor(cpu_q, dtype=torch.bfloat16),
    )
    # Quantize error is bounded by half a step of the format (relative 2^-mantissa).
    err = (dequantize_per_tensor(dev_q, dtype=torch.float32).cpu() - r).abs().max()
    assert err <= cpu_q.scale.item() * (16.0 if "e4m3" in fmt else 8192.0)


@needs_hip
@pytest.mark.parametrize("fmt", ["int8_symmetric", *FP8_FNUZ_FORMATS])
def test_low_bit_linear_exact_on_device(fmt):
    # Operands are exactly representable small dyadic values. A pinned row/column
    # holding the format max (and zeros elsewhere) forces scale == 1.0 exactly, and
    # every partial sum is exactly representable in FP32, so the result is
    # order independent and has an exact float64 expectation.
    fmax = {"int8_symmetric": 127.0, "float8_e4m3fnuz": 240.0, "float8_e5m2fnuz": 57344.0}[fmt]
    pool = {
        "int8_symmetric": [-100.0, -64.0, -3.0, 0.0, 1.0, 5.0, 100.0],
        "float8_e4m3fnuz": [-16.0, -8.0, -1.0, 0.0, 0.5, 2.0, 4.0, 16.0],
        "float8_e5m2fnuz": [-2.0, -1.0, 0.0, 0.5, 1.0, 4.0, 8.0],
    }[fmt]
    g = torch.Generator().manual_seed(69)
    k = 64
    values = torch.tensor(pool)
    x = values[torch.randint(0, len(pool), (5, k), generator=g)]
    w = values[torch.randint(0, len(pool), (7, k), generator=g)]
    pin = torch.zeros(1, k)
    pin[0, 0] = fmax
    x = torch.cat([x, pin])
    w = torch.cat([w, pin])
    expected = (x.double() @ w.double().T).to(torch.bfloat16)
    out = quantized_linear_rocm(
        x.to(DEV), w.to(DEV), input_format=fmt, weight_format=fmt, output_dtype=torch.bfloat16
    )
    assert out.is_cuda and out.dtype == torch.bfloat16
    assert torch.equal(out.cpu(), expected)
    # Device result equals the CPU result of the same contract.
    cpu_out = quantized_linear_rocm(
        x, w, input_format=fmt, weight_format=fmt, output_dtype=torch.bfloat16
    )
    assert torch.equal(out.cpu(), cpu_out)


@needs_hip
def test_int8_linear_agrees_with_cuda_contract_module():
    torch.manual_seed(5)
    x = torch.randn(4, 128, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(6, 128, device=DEV, dtype=torch.bfloat16)
    cuda_contract = quantized_linear(x.cpu(), w.cpu())  # CUDA-contract module on CPU
    out = quantized_linear_rocm(x, w).cpu()
    torch.testing.assert_close(out, cuda_contract, atol=0.1, rtol=0.05)
    ref = torch.nn.functional.linear(x.float(), w.float()).bfloat16()
    torch.testing.assert_close(quantized_linear_rocm(x, w), ref, atol=0.5, rtol=0.25)
    with pytest.raises(ValueError, match="incompatible"):
        quantized_linear_rocm(x, w[:, :64])


def test_int8_quantizer_equals_cuda_contract_quantizer_on_cpu():
    torch.manual_seed(2)
    x = torch.randn(33, 47)
    ours = quantize_per_tensor_rocm(x, format="int8_symmetric")
    theirs = quantize_per_tensor(x, format="int8_symmetric")
    assert torch.equal(ours.values, theirs.values)
    assert ours.scale.view(torch.int32) == theirs.scale.view(torch.int32)


@needs_hip
def test_scales_use_true_division_on_every_device():
    # amax / 57344 and amax / 240 differ by 1 ulp on the GPU when the divisor is a scalar
    # (x * (1 / c)); the tensor-divisor scale must be bit-identical for many amax values.
    g = torch.Generator().manual_seed(8)
    for fmt in ("int8_symmetric", *FP8_FNUZ_FORMATS):
        for _ in range(64):
            x = torch.randn(16, generator=g) * float(torch.rand(1, generator=g) * 1000 + 0.01)
            cpu = quantize_per_tensor_rocm(x, format=fmt).scale
            dev = quantize_per_tensor_rocm(x.to(DEV), format=fmt).scale.cpu()
            assert cpu.view(torch.int32) == dev.view(torch.int32), (fmt, x.abs().max())
