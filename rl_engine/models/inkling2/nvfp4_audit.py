# SPDX-License-Identifier: Apache-2.0
"""INK-29: exact-format NVFP4 conversion reference, not a native GEMM."""

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant_nvfp4(packed, scale, amax, denominator):
    """Released [expert, output, packed-K] layout; low nibble first.

    Match Miles converter's two FP32 multiply stages and final BF16 cast.
    Scales are numeric E4M3 values, not raw scale-byte codes.
    """
    if packed.ndim != 3 or packed.dtype != torch.uint8 or packed.shape[-1] % 8:
        raise ValueError("expected uint8 [expert, output, packed-K] with K divisible by 16")
    expected = (*packed.shape[:-1], packed.shape[-1] // 8)
    if scale.shape != expected or amax.shape != (packed.shape[0],):
        raise ValueError("invalid NVFP4 scale geometry")
    if denominator.shape != amax.shape:
        raise ValueError("invalid outer scale denominator")
    if any(t.device != packed.device or not t.is_floating_point()
           for t in (scale, amax, denominator)):
        raise ValueError("numeric scales must share packed device")
    s, a, d = scale.float(), amax.float(), denominator.float()
    if not bool(torch.isfinite(s).all() and torch.isfinite(a).all() and torch.isfinite(d).all()):
        raise ValueError("nonfinite NVFP4 scales")
    if bool((s < 0).any() or (a < 0).any() or (d <= 0).any()):
        raise ValueError("invalid NVFP4 scales")
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    table = torch.tensor(E2M1, device=packed.device, dtype=torch.float32)
    values = table[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
    values = values.reshape(*expected, 16) * s.unsqueeze(-1)
    return (values.flatten(-2) * (a / d).view(-1, 1, 1)).bfloat16()


def native_capability(device):
    """Report observed architecture; never certify native FP4 from a device name."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("actual GPU required")
    props = torch.cuda.get_device_properties(device)
    hip = torch.version.hip is not None
    arch = props.gcnArchName.split(":")[0] if hip else f"sm{props.major}{props.minor}"
    return {"platform": "rocm" if hip else "cuda", "arch": arch,
            "native_nvfp4_qualified": False, "conversion_profile": "bf16-dequant"}
