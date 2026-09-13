# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Accelerator abstraction shared by the WS1 gates (C3-C11 of #266).

The WS1 harness was written against CUDA. Adding the Ascend BF16 profile means
every gate needs the same small set of device facts on either vendor:
availability, the device handle, a human-readable device name, an architecture
key for evidence, TF32 policy enforcement, seeding, and synchronization.

The rules the contract cares about are vendor-independent and enforced here:

- A required profile never silently falls back. Asking for an Ascend profile on
  a host with no NPU is an error, not a CPU run.
- TF32 is disabled on every backend. CUDA has real TF32 switches; Ascend has no
  TF32 equivalent at all, so the policy is satisfied by construction and both
  report ``candidate_tf32_enabled=False``.
- ``arch_key`` is the evidence field that pins "which silicon": the SM version
  on CUDA (``sm90``), the SoC version on Ascend (``ascend910b``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# backend_profile id -> (torch device type, contract backend_family).
PROFILE_DEVICE_TYPES: dict[str, str] = {
    "cuda_bf16": "cuda",
    "triton_cuda_bf16": "cuda",
    "ascend_bf16": "npu",
}
PROFILE_FAMILIES: dict[str, str] = {
    "cuda_bf16": "cuda",
    "triton_cuda_bf16": "triton",
    "ascend_bf16": "ascend",
}
ACCELERATOR_TYPES = ("cuda", "npu")


class AcceleratorUnavailable(RuntimeError):
    """A required profile's accelerator is absent; the gate must fail, not fall back."""


def _npu() -> Any:
    """Return the ``torch.npu`` namespace, or None when Ascend is unavailable.

    ``torch.npu`` is installed onto the torch module by importing torch_npu, so
    it cannot be referenced statically. Every NPU call in this module goes
    through here.
    """

    try:
        import torch_npu  # noqa: F401
    except Exception:
        return None
    return getattr(torch, "npu", None)


def npu_available() -> bool:
    npu = _npu()
    try:
        return bool(npu is not None and npu.is_available())
    except Exception:
        return False


def is_available(device_type: str) -> bool:
    if device_type == "cuda":
        return bool(torch.cuda.is_available())
    if device_type == "npu":
        return npu_available()
    return False


def device_type_for_profile(profile: str) -> str:
    """Return the torch device type a backend profile executes on."""

    try:
        return PROFILE_DEVICE_TYPES[profile]
    except KeyError:
        raise ValueError(f"unknown backend_profile {profile!r}") from None


def family_for_profile(profile: str) -> str:
    """Return the C1 ``backend_family`` a backend profile must report."""

    try:
        return PROFILE_FAMILIES[profile]
    except KeyError:
        raise ValueError(f"unknown backend_profile {profile!r}") from None


def candidate_family(candidate: str) -> str:
    """Map a C2 ``expected_backend_id`` to its contract backend family."""

    if candidate.startswith("cuda"):
        return "cuda"
    if candidate == "triton":
        return "triton"
    if candidate.startswith("ascend") or candidate == "npu":
        return "ascend"
    return candidate


@dataclass(frozen=True)
class AcceleratorInfo:
    """Device facts a WS1 report persists so evidence names real silicon."""

    device_type: str
    device: torch.device
    name: str
    arch_key: str
    runtime_version: str | None

    @property
    def device_str(self) -> str:
        return str(self.device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "device": str(self.device),
            "name": self.name,
            "arch_key": self.arch_key,
            "runtime_version": self.runtime_version,
        }


def resolve_device(
    device: torch.device | str | None, *, profile: str | None = None
) -> torch.device:
    """Resolve the device a gate runs on, failing closed when it is absent.

    ``device=None`` picks the profile's device type. An explicit device that
    disagrees with the profile is an error: running an Ascend profile on CUDA
    would be exactly the undeclared fallback the contract forbids.
    """

    if device is None:
        if profile is None:
            raise ValueError("resolve_device needs a device or a profile")
        device_type = device_type_for_profile(profile)
    else:
        # torch.device() only knows "npu" once torch_npu has registered it, so
        # read the type from the string before handing it to torch.
        device_type = str(device).split(":", 1)[0]
        if profile is not None:
            expected = device_type_for_profile(profile)
            if device_type != expected:
                raise AcceleratorUnavailable(
                    f"profile {profile!r} executes on {expected!r}, got device {device}"
                )
    if device_type not in ACCELERATOR_TYPES:
        raise AcceleratorUnavailable(f"WS1 gates require an accelerator device, got {device}")
    if not is_available(device_type):
        hint = (
            "install torch_npu and run on an Ascend host"
            if device_type == "npu"
            else "run on a CUDA host"
        )
        raise AcceleratorUnavailable(
            f"{device_type} is not available; {hint}. Required profiles never "
            "fall back to CPU."
        )
    resolved = torch.device(device_type) if device is None else torch.device(device)
    if resolved.index is None:
        resolved = torch.device(resolved.type, current_device(resolved.type))
    return resolved


def current_device(device_type: str) -> int:
    if device_type == "cuda":
        return int(torch.cuda.current_device())
    if device_type == "npu":
        return int(_npu().current_device())
    return 0


def set_device(device: torch.device) -> None:
    if device.index is None:
        return
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu":
        _npu().set_device(device)


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return str(torch.cuda.get_device_name(device))
    if device.type == "npu":
        try:
            return str(_npu().get_device_name(device.index or 0))
        except Exception:
            return "Ascend NPU"
    return device.type


def arch_key(device: torch.device) -> str:
    """Architecture key for evidence: ``sm90`` on CUDA, ``ascend910b`` on NPU."""

    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return f"sm{major}{minor}"
    if device.type == "npu":
        # torch_npu exposes the SoC through several names across releases;
        # fall back to the device name, which already carries "Ascend910B*".
        npu = _npu()
        for getter in ("get_soc_version", "get_device_name"):
            fn = getattr(npu, getter, None)
            if fn is None:
                continue
            try:
                value = fn(device.index or 0) if getter == "get_device_name" else fn()
            except Exception:
                continue
            text = str(value).strip().lower().replace(" ", "").replace("-", "")
            if text:
                return text
    return device.type


def compute_capability(device: torch.device) -> str:
    """Dotted capability string for CUDA; the SoC key on Ascend."""

    if device.type == "cuda":
        return ".".join(str(x) for x in torch.cuda.get_device_capability(device))
    return arch_key(device)


def runtime_version(device_type: str) -> str | None:
    if device_type == "cuda":
        return getattr(torch.version, "cuda", None)
    if device_type == "npu":
        try:
            import torch_npu

            return str(getattr(torch_npu, "__version__", None) or "") or None
        except Exception:
            return None
    return None


def describe(device: torch.device) -> AcceleratorInfo:
    return AcceleratorInfo(
        device_type=device.type,
        device=device,
        name=device_name(device),
        arch_key=arch_key(device),
        runtime_version=runtime_version(device.type),
    )


def disable_tf32(device_type: str) -> bool:
    """Enforce the contract TF32 policy and report the resulting candidate flag.

    Returns the value a report must persist as ``candidate_tf32_enabled``.
    Ascend has no TF32 mode, so the policy holds with nothing to switch off.
    """

    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        return bool(torch.backends.cuda.matmul.allow_tf32)
    return False


def manual_seed_all(device_type: str, seed: int) -> None:
    torch.manual_seed(seed)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif device_type == "npu" and npu_available():
        _npu().manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "npu" and npu_available():
        _npu().synchronize(device)


def empty_cache(device_type: str) -> None:
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "npu" and npu_available():
        _npu().empty_cache()
