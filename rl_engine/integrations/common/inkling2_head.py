# SPDX-License-Identifier: Apache-2.0
"""Shared BF16 unembedding boundary for the pinned Inkling 2 Mini engines."""

import torch

PADDED_VOCAB = 201024
VALID_VOCAB = 200058
HIDDEN = 2304


def require_platform(tensor: torch.Tensor, platform: str) -> None:
    if platform not in ("cuda", "rocm"):
        raise ValueError("expected cuda or rocm platform")
    if not tensor.is_cuda:
        raise RuntimeError("Inkling strict head requires a real GPU tensor")
    actual = "rocm" if torch.version.hip is not None else "cuda"
    if platform != actual:
        raise RuntimeError(f"requested {platform}, got {actual}")
    if actual == "rocm":
        arch = torch.cuda.get_device_properties(tensor.device).gcnArchName.split(":")[0]
        if arch != "gfx950":
            raise RuntimeError("initial ROCm profile requires gfx950 (MI350X/MI355X)")


def raw_head(hidden: torch.Tensor, weight: torch.Tensor, *, platform: str) -> torch.Tensor:
    """Use INK-32's FP32 GEMM and BF16-gradient boundary, without silent fallback."""
    require_platform(hidden, platform)
    if hidden.ndim != 2 or hidden.shape[1] != HIDDEN or not hidden.shape[0]:
        raise ValueError("expected nonempty [tokens, 2304] hidden states")
    if hasattr(weight, "placements"):
        raise RuntimeError("distributed weight requires a separate WS2 binding")
    if weight.shape != (PADDED_VOCAB, HIDDEN) or weight.device != hidden.device:
        raise ValueError("initial head path requires the full unsharded padded weight")
    if hidden.dtype != torch.bfloat16 or weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("expected BF16 hidden and BF16/FP32 head weight")
    # INK-32 owns the arithmetic, including the VJP; never duplicate its implementation.
    from rl_engine.models.inkling2.lm_head_logprob import _UnembedFP32

    return _UnembedFP32.apply(hidden, weight.to(torch.bfloat16))
