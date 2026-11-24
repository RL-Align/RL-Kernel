# SPDX-License-Identifier: Apache-2.0
"""ROCm counterpart of the Inkling 2 Mini precision contracts.

Covers INK-04, INK-05, INK-30, INK-31, INK-52, INK-67 and INK-69 on AMD
Instinct GPUs. Everything here is derived from the real runtime (HIP version,
``gcnArchName``, device count, RCCL) instead of being assumed:

* gfx942 (MI300 series): FP8 is the *FNUZ* family (``float8_e4m3fnuz`` max 240,
  ``float8_e5m2fnuz`` max 57344). There is no OCP ``float8_e4m3fn`` MXFP8
  hardware path and no FP4 of any kind.
* gfx950 (MI350 / MI355): OCP FP8 and the MX formats are declared, but they are
  only reported available when the detected arch really is gfx950. On any other
  host they are gated with an explicit reason, never faked as passing.
* NVFP4 is an NVIDIA format and is never claimed native on AMD.

The per-tensor quantization maths is bit-identical to the CUDA contract module
(FP32 scale = amax / format_max, FP32 round-to-nearest-even casts), so a tensor
quantized on the HIP device must equal the CPU result byte for byte.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Mapping

import torch

from .precision_contracts import (
    CheckpointFingerprint,
    PrecisionContractError,
    QuantBackendCapability,
    QuantBackendMatrix,
    QuantizedTensor,
    dequantize_per_tensor,
    fingerprint_checkpoint,
)

FP8_FNUZ_FORMATS = {
    "float8_e4m3fnuz": ("float8_e4m3fnuz", 240.0),
    "float8_e5m2fnuz": ("float8_e5m2fnuz", 57344.0),
}
INT8_FORMAT = "int8_symmetric"
OCP_FP8_FORMATS = ("float8_e4m3fn", "float8_e5m2")
MX_FORMATS = ("mxfp8", "mxfp4")
NVFP4_FORMAT = "nvfp4"

# arch family -> (product line, FP8 flavour)
_GFX_FAMILIES = {
    "gfx942": ("MI300X/MI300A/MI325X (CDNA3)", "fnuz"),
    "gfx950": ("MI350X/MI355X (CDNA4)", "ocp"),
}


# --------------------------------------------------------------------------
# Runtime readback
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RocmRuntime:
    hip_version: str | None
    device_count: int
    device_name: str
    gcn_arch_name: str  # full string, e.g. "gfx942:sramecc+:xnack-"
    total_memory_bytes: int
    rccl_available: bool

    @property
    def arch(self) -> str:
        """Arch family without feature flags, e.g. ``gfx942``."""
        return self.gcn_arch_name.split(":", 1)[0]

    @property
    def is_rocm(self) -> bool:
        return self.hip_version is not None and self.device_count > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "hip_version": self.hip_version,
            "device_count": self.device_count,
            "device_name": self.device_name,
            "gcn_arch_name": self.gcn_arch_name,
            "arch": self.arch,
            "total_memory_bytes": self.total_memory_bytes,
            "rccl_available": self.rccl_available,
        }


def detect_rocm_runtime(device_index: int = 0) -> RocmRuntime:
    """Read the ROCm runtime state from torch; fields are empty when not on ROCm."""
    hip = getattr(torch.version, "hip", None)
    count = torch.cuda.device_count() if hip is not None and torch.cuda.is_available() else 0
    name, arch, total = "", "", 0
    if count > 0:
        props = torch.cuda.get_device_properties(device_index)
        name = props.name
        arch = getattr(props, "gcnArchName", "")
        total = int(props.total_memory)
    try:
        rccl = bool(hip is not None and torch.distributed.is_nccl_available())
    except Exception:  # pragma: no cover - defensive, torch.distributed may be absent
        rccl = False
    return RocmRuntime(hip, count, name, arch, total, rccl)


# --------------------------------------------------------------------------
# Quant backend matrix
# --------------------------------------------------------------------------
def probe_fp8_fnuz_on_device(device: torch.device | str = "cuda") -> bool:
    """Execute a real FNUZ FP8 cast round trip on the device (exact values)."""
    try:
        probe = torch.tensor([1.0, -2.0, 0.5, 240.0], device=device)
        out = probe.to(torch.float8_e4m3fnuz).float()
        return bool(torch.equal(out, probe))
    except Exception:
        return False


def build_rocm_quant_matrix(
    runtime: RocmRuntime | None = None, *, fnuz_probe: bool | None = None
) -> QuantBackendMatrix:
    """Quant backend matrix for ``rocm:gfx942`` and ``rocm:gfx950`` platforms.

    An entry is ``available`` only when the detected gcnArchName family equals
    the entry's platform arch (and, for FNUZ, a device probe succeeded). Every
    other entry carries the reason it is gated.
    """
    runtime = runtime if runtime is not None else detect_rocm_runtime()
    detected = runtime.arch if runtime.is_rocm else ""
    if fnuz_probe is None:
        fnuz_probe = probe_fp8_fnuz_on_device() if detected == "gfx942" else False

    def on_host(arch: str) -> tuple[bool, str]:
        if detected == arch:
            return True, ""
        where = f"detected {detected}" if detected else "no ROCm device detected"
        return False, f"{arch} ({_GFX_FAMILIES[arch][0]}) not present on this host ({where})"

    entries: list[QuantBackendCapability] = []

    def add(platform, backend, fmt, training, inference, available, reason):
        cap = QuantBackendCapability(platform, backend, fmt, training, inference, available, reason)
        entries.append(cap)

    nvfp4_why = "NVFP4 is an NVIDIA format; never claimed native on AMD"
    # --- gfx942: int8 + FNUZ FP8 (torch emulated per-tensor quant + FP32 dequant)
    present, why = on_host("gfx942")
    add("rocm:gfx942", "torch", INT8_FORMAT, True, True, present, why)
    for fmt in FP8_FNUZ_FORMATS:
        ok = present and fnuz_probe
        reason = why or ("" if ok else "FNUZ FP8 device probe failed")
        add("rocm:gfx942", "torch", fmt, True, True, ok, reason)
    gated_942 = {
        "float8_e4m3fn": "OCP FP8 has no gfx942 hardware path (FNUZ only)",
        "float8_e5m2": "OCP FP8 has no gfx942 hardware path (FNUZ only)",
        "mxfp8": "MXFP8 needs OCP FP8 hardware (gfx950); gfx942 has FNUZ only",
        "mxfp4": "MXFP4 needs gfx950 hardware; gfx942 has no FP4",
        NVFP4_FORMAT: nvfp4_why,
    }
    for fmt, extra in gated_942.items():
        add("rocm:gfx942", "none", fmt, False, False, False, extra)
    # --- gfx950: OCP FP8 + MX declared, gated unless this host really is gfx950
    present, why = on_host("gfx950")
    add("rocm:gfx950", "torch", INT8_FORMAT, True, True, present, why)
    for fmt in (*OCP_FP8_FORMATS, *MX_FORMATS):
        backend = "aiter" if fmt in MX_FORMATS else "torch"
        reason = why or "declared; not validated on this project's hardware"
        add("rocm:gfx950", backend, fmt, False, True, present, reason)
    for fmt in FP8_FNUZ_FORMATS:
        add("rocm:gfx950", "none", fmt, False, False, False, "gfx950 uses OCP FP8, not FNUZ")
    add("rocm:gfx950", "none", NVFP4_FORMAT, False, False, False, nvfp4_why)
    return QuantBackendMatrix(tuple(entries))


# --------------------------------------------------------------------------
# Strictness classification
# --------------------------------------------------------------------------
class PrecisionProfile(enum.Enum):
    STRICT = "STRICT"  # FP32/BF16 compute, byte-deterministic, no quantized compute
    MIXED = "MIXED"  # quantized storage, BF16 dequant before compute
    APPROX = "APPROX"  # quantized compute / emulated formats / approximate kernels


def profile_for_formats(formats: tuple[str, ...]) -> PrecisionProfile:
    """Lowest-fidelity profile implied by the formats a run touches."""
    if not formats:
        return PrecisionProfile.STRICT
    if all(fmt in {"bf16", "fp16", "fp32"} for fmt in formats):
        return PrecisionProfile.STRICT
    if any(fmt in {*MX_FORMATS, NVFP4_FORMAT, *OCP_FP8_FORMATS} for fmt in formats):
        return PrecisionProfile.APPROX
    return PrecisionProfile.MIXED


def is_strict(profile: PrecisionProfile) -> bool:
    """MIXED and APPROX profiles are never reported as strict."""
    return profile is PrecisionProfile.STRICT


def require_strict(profile: PrecisionProfile) -> None:
    if not is_strict(profile):
        raise PrecisionContractError(
            f"{profile.value} precision profile is not strict; it must not be reported as strict"
        )


# --------------------------------------------------------------------------
# Quantization (device-agnostic, bit-identical to the CPU result)
# --------------------------------------------------------------------------
def _per_tensor_scale(source: torch.Tensor, fmax: float) -> torch.Tensor:
    """FP32 scale = amax / fmax using a tensor divisor.

    Dividing by a Python scalar lets the GPU compile ``x / c`` into ``x * (1 / c)``,
    which differs from the CPU result by 1 ulp for divisors such as 240 or 57344.
    A tensor divisor is a true IEEE division on every device.
    """
    amax = source.abs().amax().clamp_min(torch.finfo(torch.float32).eps)
    return amax / torch.full_like(amax, fmax)


def quantize_per_tensor_rocm(
    tensor: torch.Tensor, *, format: str = INT8_FORMAT, runtime: RocmRuntime | None = None
) -> QuantizedTensor:
    """Per-tensor quantization for the formats the detected arch supports.

    Operates on whichever device ``tensor`` lives on and returns bytes identical to
    the CPU result. OCP FP8 is refused unless the detected arch is gfx950;
    NVFP4/MX are never supported here.
    """
    if tensor.numel() == 0 or not tensor.is_floating_point():
        raise PrecisionContractError("quantization needs a nonempty floating point tensor")
    if format == INT8_FORMAT:
        source = tensor.float()
        scale = _per_tensor_scale(source, 127.0)
        values = torch.round(source / scale).clamp(-127, 127).to(torch.int8)
        return QuantizedTensor(values, scale.reshape(()), format)
    if format in FP8_FNUZ_FORMATS:
        dtype_name, fmax = FP8_FNUZ_FORMATS[format]
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise RuntimeError(f"PyTorch {dtype_name} is unavailable")
        source = tensor.float()
        scale = _per_tensor_scale(source, fmax)
        return QuantizedTensor((source / scale).to(dtype), scale.reshape(()), format)
    if format in OCP_FP8_FORMATS or format in MX_FORMATS or format == NVFP4_FORMAT:
        runtime = runtime if runtime is not None else detect_rocm_runtime()
        matrix = build_rocm_quant_matrix(runtime)
        platform = f"rocm:{runtime.arch}" if runtime.arch in _GFX_FAMILIES else "rocm:gfx942"
        matrix.require(platform=platform, format=format, purpose="inference")
        raise RuntimeError(f"{format} quantization is declared but not implemented in this module")
    raise PrecisionContractError(f"unsupported quantization format {format!r}")


def quantized_linear_rocm(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    input_format: str = INT8_FORMAT,
    weight_format: str = INT8_FORMAT,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Low-bit linear: per-tensor quantize both operands, FP32 dequant, FP32 linear."""
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[-1]:
        raise ValueError("input and weight have incompatible linear shapes")
    input_dq = dequantize_per_tensor(
        quantize_per_tensor_rocm(input, format=input_format), dtype=torch.float32
    )
    weight_dq = dequantize_per_tensor(
        quantize_per_tensor_rocm(weight, format=weight_format), dtype=torch.float32
    )
    return torch.nn.functional.linear(input_dq, weight_dq).to(output_dtype)


# --------------------------------------------------------------------------
# BF16-DEQUANT checkpoint fingerprints
# --------------------------------------------------------------------------
def dequantize_checkpoint_bf16(
    quantized: Mapping[str, QuantizedTensor], *, device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Dequantize a quantized checkpoint to BF16 on ``device``."""
    out = {}
    for name, q in quantized.items():
        moved = QuantizedTensor(q.values.to(device), q.scale.to(device), q.format)
        out[name] = dequantize_per_tensor(moved, dtype=torch.bfloat16)
    return out


def fingerprint_bf16_dequant_checkpoint(
    quantized: Mapping[str, QuantizedTensor],
    *,
    source: str,
    revision: str,
    device: torch.device | str = "cpu",
) -> CheckpointFingerprint:
    """Fingerprint of the BF16-DEQUANT view, dequantized on ``device``.

    Dequantization is an exact elementwise ``float(value) * float(scale)`` cast
    to BF16, so the fingerprint is identical whichever device computed it.
    """
    return fingerprint_checkpoint(
        dequantize_checkpoint_bf16(quantized, device=device), source=source, revision=revision
    )
