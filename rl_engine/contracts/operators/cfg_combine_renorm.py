# SPDX-License-Identifier: Apache-2.0
import math

import torch
from torch import Tensor


def cfg_platform(device_type: str) -> str:
    if device_type == "cpu" or (device_type == "cuda" and torch.version.hip is None):
        return device_type
    raise RuntimeError(
        f"CFG requires CPU or NVIDIA CUDA, got {device_type} (HIP={torch.version.hip})"
    )


def request_layout(
    v_main: Tensor,
    v_text: Tensor | None,
    v_img: Tensor | None,
    cu_seqlens: Tensor | None,
    cfg_text_scale: float,
    cfg_img_scale: float,
    timestep: float | Tensor | None,
    cfg_interval: tuple[float, float],
    renorm_type: str,
    cfg_renorm_min: float,
    eps: float,
    strict: bool,
) -> tuple[list[int], list[bool]]:
    cfg_platform(v_main.device.type)
    if v_main.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("v_main must use fp32, bf16 or fp16")
    if v_main.ndim not in (2, 3):
        raise ValueError("expected [B, S, D] or [N, D] velocities")
    for name, branch in (("v_text", v_text), ("v_img", v_img)):
        if branch is None:
            continue
        if branch.dtype != v_main.dtype:
            raise TypeError(f"{name} must match v_main dtype")
        if branch.shape != v_main.shape or branch.device != v_main.device:
            raise ValueError(f"{name} must match v_main shape and device")
    if renorm_type not in ("global", "channel", "text_channel"):
        raise ValueError(f"unknown renorm_type: {renorm_type}")
    if not all(math.isfinite(x) for x in (cfg_text_scale, cfg_img_scale, cfg_renorm_min, eps)):
        raise ValueError("CFG parameters must be finite")
    if not 0 <= cfg_renorm_min <= 1 or torch.tensor(eps, dtype=torch.float32).item() <= 0:
        raise ValueError("cfg_renorm_min must be in [0, 1] and eps must be positive in FP32")
    if v_main.ndim == 3:
        if cu_seqlens is not None:
            raise ValueError("cu_seqlens is only valid for packed inputs")
        batch, seq, _ = v_main.shape
        offsets = [b * seq for b in range(batch + 1)]
    elif cu_seqlens is None:
        if strict and renorm_type == "global":
            raise ValueError("strict packed global renorm requires cu_seqlens")
        offsets = [0, v_main.shape[0]]
    else:
        if (
            cu_seqlens.ndim != 1
            or cu_seqlens.device != v_main.device
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("cu_seqlens must be a 1D integer tensor on the input device")
        offsets = cu_seqlens.tolist()
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != v_main.shape[0]
            or any(a >= b for a, b in zip(offsets, offsets[1:]))
        ):
            raise ValueError("cu_seqlens must increase strictly from 0 to N")
    batch = len(offsets) - 1
    lo, hi = cfg_interval
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise ValueError("cfg_interval must contain ordered finite endpoints")
    # 对齐 ByteDance-Seed/Bagel 的 modeling/bagel/bagel.py 与 RFC #435。
    # cfg_text_scale <= 1.0 是独立的 identity profile。即使 cfg_img_scale=1.5
    # 且 v_img=None，也直接输出 v_main。
    active = [cfg_text_scale > 1] * batch
    if timestep is not None:
        t = torch.as_tensor(
            timestep,
            dtype=torch.float32,
            device=timestep.device if isinstance(timestep, Tensor) else v_main.device,
        )
        if t.device != v_main.device or (t.ndim != 0 and t.shape != (batch,)):
            raise ValueError("timestep must be scalar or [B] on the input device")
        if not torch.isfinite(t).all():
            raise ValueError("timestep must be finite")
        active = ((t > lo) & (t <= hi) & (cfg_text_scale > 1)).expand(batch).tolist()
    if any(active) and v_text is None:
        raise ValueError("v_text is required for active text guidance")
    return offsets, active
