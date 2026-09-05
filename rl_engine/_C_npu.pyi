# rl_engine/_C_npu.pyi
# Type stub for the compiled Ascend C (CANN) extension module.
# Built only when KERNEL_ALIGN_FORCE_ASCEND=1 on a machine with CANN + torch_npu.
import torch

def swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor: ...
def swiglu_backward(
    grad_out: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> list[torch.Tensor]: ...
def batch_invariant_logp_ascend(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> list[torch.Tensor]: ...
