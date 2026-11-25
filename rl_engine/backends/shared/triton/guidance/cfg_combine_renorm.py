# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor, nn

from rl_engine.contracts.operators.cfg_combine_renorm import request_layout


class _Request(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, t, image, active, st, si, mode, minimum, eps, kernels):
        ctx.guided = active and v.numel() != 0
        if not ctx.guided:
            return v.clone()
        rows = 1 if mode == "global" else v.shape[0]
        post_image = mode == "text_channel"
        with torch.cuda.device(v.device):
            out, mixed, a2, b2 = kernels.forward(
                v, t, image, rows, st, si, post_image, minimum, eps
            )
        ctx.save_for_backward(v, mixed, a2, b2)
        ctx.policy = st, si, post_image, image is not None and si > 1, minimum, eps
        ctx.kernels = kernels
        return out

    @staticmethod
    def backward(ctx, grad):
        needs = ctx.needs_input_grad[:3]
        if ctx.guided:
            with torch.cuda.device(grad.device):
                grads = ctx.kernels.backward(
                    grad.contiguous(), *ctx.saved_tensors, *ctx.policy, needs
                )
        else:
            grads = (
                grad if needs[0] else None,
                torch.zeros_like(grad) if needs[1] else None,
                torch.zeros_like(grad) if needs[2] else None,
            )
        return (*grads, None, None, None, None, None, None, None)


class TritonCfgCombineRenormOp(nn.Module):
    op_class = "reduction"

    def forward(
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
        if (
            v_main.device.type != "cuda"
            or torch.version.hip is not None
            or torch.version.cuda is None
        ):
            raise RuntimeError("Triton CFG requires an NVIDIA CUDA tensor and CUDA PyTorch build")
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
        from rl_engine.backends.shared.triton.guidance import _kernels

        main = v_main.flatten(0, -2).contiguous()
        text = v_text.flatten(0, -2).contiguous() if v_text is not None else None
        image = v_img.flatten(0, -2).contiguous() if v_img is not None else None
        policy = cfg_text_scale, cfg_img_scale, renorm_type, cfg_renorm_min, eps, _kernels
        if main.numel() == 0:
            return _Request.apply(main, text, image, False, *policy).reshape_as(v_main)
        outputs = []
        for start, end, guided in zip(offsets[:-1], offsets[1:], active, strict=True):
            t = text[start:end] if text is not None else None
            i = image[start:end] if image is not None else None
            outputs.append(_Request.apply(main[start:end], t, i, guided, *policy))
        return torch.cat(outputs).reshape_as(v_main)
