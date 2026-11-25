# SPDX-License-Identifier: Apache-2.0
from torch import Tensor

from rl_engine.backends.shared.triton.guidance import TritonCfgCombineRenormOp
from rl_engine.reference.guidance import NativeCfgCombineRenormOp
from rl_engine.runtime.registry import kernel_registry


def cfg_combine_renorm(
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
    backend: str = "auto",
) -> Tensor:
    if backend == "auto":
        op = kernel_registry.get_op("cfg_combine_renorm", device=v_main.device)
    elif backend == "pytorch":
        op = NativeCfgCombineRenormOp()
    elif backend == "triton":
        op = TritonCfgCombineRenormOp()
    else:
        raise ValueError(f"unsupported CFG backend: {backend}")
    return op(
        v_main,
        v_text,
        v_img,
        cu_seqlens=cu_seqlens,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        timestep=timestep,
        cfg_interval=cfg_interval,
        renorm_type=renorm_type,
        cfg_renorm_min=cfg_renorm_min,
        eps=eps,
        strict=strict,
    )
