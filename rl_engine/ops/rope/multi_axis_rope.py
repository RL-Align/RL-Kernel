# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image interleaved three-axis RoPE (RFC #386).

Tables are immutable inputs, built before capture/replay. The rotation accepts
BSHD tensors, unlike the legacy BHSD, rotate-half RoPE operator.
"""

from __future__ import annotations

import torch

CONTRACT = "qwenimage-interleaved-rope-fp32-v1"
AXES_DIMS = (16, 56, 56)
UPSTREAM_REVISION = "40f63690da440e8c382ab53e67508835deb1ed2f"


def qwen_image_positions(
    height: int, width: int, text_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU int64 positions for one image's *packed latent* H/W, then text.

    Matches QwenEmbedRope(scale_rope=True), single-frame text-to-image. Odd
    grids have the extra negative coordinate: H=3 gives [-2, -1, 0].
    Return separate tables so callers explicitly choose their stream order.
    """
    for name, value in (
        ("height", height),
        ("width", width),
        ("text_length", text_length),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if not 1 <= height <= 8192 or not 1 <= width <= 8192 or text_length < 0:
        raise ValueError("height/width must be in [1, 8192], text_length must be nonnegative")
    start = max(height // 2, width // 2)
    if start + text_length > 4096:
        raise ValueError("text positions exceed the pinned upstream 4096-entry table")
    h = torch.arange(-(height - height // 2), height // 2, device="cpu")
    w = torch.arange(-(width - width // 2), width // 2, device="cpu")
    hh, ww = torch.meshgrid(h, w, indexing="ij")
    image = torch.stack((torch.zeros_like(hh), hh, ww), dim=-1).reshape(-1, 3)
    text = (
        torch.arange(start, start + text_length, device="cpu")[:, None].expand(-1, 3).contiguous()
    )
    return image, text


def build_multi_axis_cos_sin(
    positions: torch.Tensor, *, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build canonical FP32 tables on CPU, then copy to the requested device.

    positions: int32/int64 [S, 3] or [B, S, 3], range [-4096, 4095].
    The CPU torch.polar construction follows the pinned Diffusers reference;
    no GPU transcendental or per-batch table recomputation enters the hot path.
    """
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must be int32 or int64")
    if positions.ndim not in (2, 3) or positions.shape[-1] != 3:
        raise ValueError("positions must be [S, 3] or [B, S, 3]")
    cpu = positions.detach().to(device="cpu", dtype=torch.float32)
    if torch.any((cpu < -4096) | (cpu >= 4096)):
        raise ValueError("positions exceed the pinned upstream [-4096, 4095] range")
    phases = []
    for axis, dim in enumerate(AXES_DIMS):
        inv_freq = 1.0 / torch.pow(10000, torch.arange(0, dim, 2, device="cpu").float().div(dim))
        phases.append(cpu[..., axis, None] * inv_freq)
    phase = torch.cat(phases, dim=-1)
    cis = torch.polar(torch.ones_like(phase), phase)
    target = positions.device if device is None else device
    return cis.real.contiguous().to(target), cis.imag.contiguous().to(target)


def validate_inputs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    if x.ndim != 4 or x.shape[-1] != 128 or x.shape[2] <= 0:
        raise ValueError("x must be [B, S, H, 128] with H > 0")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("x must be float16, bfloat16 or float32")
    shared = (x.shape[1], 64)
    batched = (x.shape[0], x.shape[1], 64)
    if cos.shape not in (shared, batched) or sin.shape != cos.shape:
        raise ValueError("cos/sin must have matching [S, 64] or [B, S, 64] shapes")
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise TypeError("cos/sin must be float32")
    if x.device != cos.device or x.device != sin.device:
        raise ValueError("x, cos and sin must be on the same device")
    if cos.requires_grad or sin.requires_grad:
        raise ValueError("cos/sin are fixed tables; table gradients are unsupported")


def get_multi_axis_rope_op(backend: str = "auto", *, device="cpu"):
    """Resolve an op exposing trace(); explicit backends never silently fall back.

    Strict callers must request cuda or triton and retain the returned trace.
    auto is a diagnostic/convenience mode with a PyTorch reference fallback.
    """
    if backend == "auto":
        from rl_engine.runtime.registry import kernel_registry

        return kernel_registry.get_op("multi_axis_rope", device=device)
    if backend == "cuda":
        from rl_engine.backends.cuda.rope.multi_axis_rope import MultiAxisRoPECudaOp

        return MultiAxisRoPECudaOp()
    if backend == "triton":
        from rl_engine.backends.shared.triton.rope.multi_axis_rope import MultiAxisRoPETritonOp

        return MultiAxisRoPETritonOp()
    if backend == "reference":
        from rl_engine.reference.rope.multi_axis_rope import NativeMultiAxisRoPEOp

        return NativeMultiAxisRoPEOp()
    raise ValueError(f"unknown multi_axis_rope backend: {backend!r}")


def multi_axis_rope(x, cos, sin, *, backend="auto"):
    return get_multi_axis_rope_op(backend, device=x.device)(x, cos, sin)


class MultiAxisRoPEMetadata:
    backend_id: str
    kernel_id: str

    def trace(self) -> dict:
        return {
            "contract": CONTRACT,
            "actual_backend": self.backend_id,
            "kernel_id": self.kernel_id,
            "layout": "BSHD, adjacent channel pairs",
            "arithmetic": "fp32 separate multiply/add, one output cast, no reduction",
            "table_revision": UPSTREAM_REVISION,
        }
