# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Standalone harness for the shared Hash/Learned route backward arithmetic.

Raw tensors here are synthetic experiment inputs, NOT SavedRouteSealedV1.
This harness does not register an RL-Kernel operator or claim a P3 verdict.
"""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _load_extension():
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires NVIDIA CUDA; no fallback is provided")
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    return load(
        name="rlk_p3_t06_arithmetic_core",
        sources=[
            str(root / "examples/p3_router_backward/bindings.cpp"),
            str(root / "csrc/cuda/moe/router_backward_core.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "-lineinfo",
        ],
    )


def fixed_tree6(x: torch.Tensor) -> torch.Tensor:
    """Explicit six-term tree, also used to construct synthetic forward state."""
    return ((x[:, 0] + x[:, 1]) + (x[:, 2] + x[:, 3])) + (x[:, 4] + x[:, 5])


@torch.no_grad()
def diagnostic_reference(dweights, ids, p, z, row_active):
    """CPU eager reference for local tests; this is NOT the official T01 oracle.

    Use FP32 for operation-order checks and FP64 for mathematical comparisons.
    Only active rows participate, and duplicate slots accumulate sequentially.
    """
    if any(t.device.type != "cpu" for t in (dweights, ids, p, z, row_active)):
        raise ValueError("diagnostic_reference expects CPU tensors")
    result = torch.zeros((dweights.shape[0], 256), dtype=dweights.dtype)
    for token in row_active.nonzero(as_tuple=True)[0].tolist():
        g = dweights[token]
        gp = g * p[token]
        c = ((gp[0] + gp[1]) + (gp[2] + gp[3])) + (gp[4] + gp[5])
        # Tensor.__rtruediv__ may compute reciprocal * scalar. Use true divide
        # to match one correctly rounded FP32 division, __fdiv_rn(1.5, Z).
        scale_over_z = torch.div(torch.full_like(z[token], 1.5), z[token])
        da = scale_over_z * (g - c)
        for slot in range(6):
            expert = int(ids[token, slot])
            result[token, expert] = result[token, expert] + da[slot]
    # Include signed zero in bitwise comparisons.
    result[result == 0] = 0.0
    return result


@torch.no_grad()
def route_backward_core(dweights, ids, p, z, row_active, *, threads=256):
    """Validate raw experiment inputs and run the CUDA arithmetic core.

    All inputs are contiguous and on the same NVIDIA CUDA device:
    dweights/p FP32[T,6], ids INT32[T,6], z FP32[T], row_active BOOL[T].
    Here p is the saved *unscaled* probability, and z is saved capital Z,
    the normalization denominator (not the upstream gate logits).

    Validation uses synchronous checks. Do not interpret end-to-end harness
    latency as kernel latency. Official identity/status handling is pending T01.
    """
    if torch.version.cuda is None or not dweights.is_cuda:
        raise ValueError("dweights must be on NVIDIA CUDA")
    if dweights.ndim != 2 or dweights.shape[1] != 6:
        raise ValueError("dweights must have shape [T, 6]")
    if threads not in (128, 256):
        raise ValueError("threads must be 128 or 256")
    tokens = dweights.shape[0]
    for name, tensor, dtype, shape in (
        ("dweights", dweights, torch.float32, (tokens, 6)),
        ("ids", ids, torch.int32, (tokens, 6)),
        ("p", p, torch.float32, (tokens, 6)),
        ("z", z, torch.float32, (tokens,)),
        ("row_active", row_active, torch.bool, (tokens,)),
    ):
        if tensor.device != dweights.device:
            raise ValueError(f"{name} must be on the same CUDA device as dweights")
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if tokens == 0 or not row_active.any().item():
        return torch.zeros((tokens, 256), device=dweights.device, dtype=torch.float32)
    for name, tensor in (("dweights", dweights), ("p", p), ("z", z)):
        if not torch.isfinite(tensor[row_active]).all().item():
            raise ValueError(f"active {name} must be finite")
    active_ids = ids[row_active]
    if not ((active_ids >= 0) & (active_ids < 256)).all().item():
        raise ValueError("active ids must be in [0, 256)")
    if not (z[row_active] > 0).all().item() or not (p[row_active] >= 0).all().item():
        raise ValueError("active z must be positive and p must be nonnegative")

    output = torch.empty((tokens, 256), device=dweights.device, dtype=torch.float32)
    _load_extension().route_backward_core_out(dweights, ids, p, z, row_active, output, threads)
    if not torch.isfinite(output).all().item():
        raise RuntimeError("route backward arithmetic produced a non-finite result")
    return output
