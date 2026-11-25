# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rl_engine.contracts.operators.cfg_combine_renorm import request_layout


def _pairwise_sum(values: Tensor) -> Tensor:
    # 每层按相邻两项相加，奇数尾项补零。每 256 个叶子自然构成一个固定子树。
    while values.shape[-1] > 1:
        if values.shape[-1] % 2:
            values = F.pad(values, (0, 1))
        values = values[..., ::2] + values[..., 1::2]
    return values


class _Renorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, main, mixed, global_norm, minimum, eps):
        main_rows = main.reshape(1, -1) if global_norm else main
        mixed_rows = mixed.reshape_as(main_rows)
        a = _pairwise_sum(main_rows.square()).sqrt()
        b = _pairwise_sum(mixed_rows.square()).sqrt()
        denom = b + eps
        ratio = a / denom
        scale = ratio.clamp(minimum, 1.0)
        interior = (ratio > minimum) & (ratio < 1.0)
        ctx.save_for_backward(main_rows, mixed_rows, a, b, denom, scale, interior)
        ctx.shape = main.shape
        return (mixed_rows * scale).reshape_as(main)

    @staticmethod
    def backward(ctx, grad):
        main, mixed, a, b, denom, scale, interior = ctx.saved_tensors
        grad = grad.reshape_as(mixed)
        dot = _pairwise_sum(grad * mixed)
        factor = torch.where(interior, dot, 0.0) / denom
        # 在零范数处采用零次梯度，避免除零。
        main_unit = main / torch.where(a == 0, 1.0, a)
        mixed_unit = mixed / torch.where(b == 0, 1.0, b)
        dmain = factor * main_unit
        dmixed = grad * scale - (factor * scale) * mixed_unit
        return dmain.reshape(ctx.shape), dmixed.reshape(ctx.shape), None, None, None


class _SkipBranches(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, *unused):
        # 未参与计算的分支即使有 NaN，也不影响输出，同时保留显式零梯度。
        ctx.save_for_backward(*unused)
        return value

    @staticmethod
    def backward(ctx, grad):
        return (grad, *(torch.zeros_like(branch) for branch in ctx.saved_tensors))


class NativeCfgCombineRenormOp(nn.Module):
    """CPU and NVIDIA CUDA gold path. The FP32 tree uses no TF32-capable operations."""

    op_class = "reduction"

    def forward(self, v_main, v_text=None, v_img=None, **policy):
        return self.forward_fp32(v_main, v_text, v_img, **policy).to(v_main.dtype)

    def forward_fp32(
        self,
        v_main: Tensor,
        v_text: Tensor | None = None,
        v_img: Tensor | None = None,
        *,
        cu_seqlens: Tensor | None = None,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        timestep: float | Tensor | None = None,
        cfg_interval: tuple[float, float] = (0.4, 1.0),
        renorm_type: str = "global",
        cfg_renorm_min: float = 0.0,
        eps: float = 1e-8,
        strict: bool = True,
    ) -> Tensor:
        offsets, active = request_layout(
            v_main,
            v_text,
            v_img,
            cu_seqlens,
            cfg_text_scale,
            cfg_img_scale,
            timestep,
            cfg_interval,
            renorm_type,
            cfg_renorm_min,
            eps,
            strict,
        )
        main = v_main.flatten(0, -2).float()
        text = v_text.flatten(0, -2).float() if v_text is not None else None
        image = v_img.flatten(0, -2).float() if v_img is not None else None
        if main.numel() == 0:
            unused = [branch for branch in (text, image) if branch is not None]
            return _SkipBranches.apply(main, *unused).reshape_as(v_main)
        outputs = []
        for start, end, guided in zip(offsets[:-1], offsets[1:], active, strict=True):
            v = main[start:end]
            t = text[start:end] if text is not None else None
            i = image[start:end] if image is not None else None
            if not guided:
                unused = [branch for branch in (t, i) if branch is not None]
                outputs.append(_SkipBranches.apply(v, *unused))
                continue
            mixed = t + cfg_text_scale * (v - t)
            if renorm_type == "text_channel":
                mixed = _Renorm.apply(v, mixed, False, cfg_renorm_min, eps)
            # ByteDance-Seed/Bagel 的 modeling/bagel/bagel.py 与 RFC #435 使用同一分支规则。
            # 只有 cfg_img_scale > 1.0 才混合图像分支，否则保留两分支文本引导。
            if i is not None:
                if cfg_img_scale > 1.0:
                    mixed = i + cfg_img_scale * (mixed - i)
                else:
                    mixed = _SkipBranches.apply(mixed, i)
            if renorm_type != "text_channel":
                mixed = _Renorm.apply(v, mixed, renorm_type == "global", cfg_renorm_min, eps)
            outputs.append(mixed)
        return torch.cat(outputs).reshape_as(v_main)
