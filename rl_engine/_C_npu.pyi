# rl_engine/_C_npu.pyi
# Type stub for the compiled Ascend C (CANN) extension module.
# Built only when KERNEL_ALIGN_FORCE_ASCEND=1 on a machine with CANN + torch_npu.
import torch

def batch_invariant_logp_ascend(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> list[torch.Tensor]: ...

def rope_apply_ascend(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sin_sign: float,
) -> torch.Tensor: ...
