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

def deterministic_attention_ascend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    scale: float,
    key_padding_mask: torch.Tensor | None,
) -> list[torch.Tensor]: ...

def prefix_shared_attention_ascend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor: ...

def deterministic_collective_create(
    staging: torch.Tensor,
    world_size: int,
    rank: int,
) -> int: ...

def deterministic_collective_destroy(handle: int) -> None: ...

def deterministic_collective_stage(handle: int, input: torch.Tensor) -> None: ...

def deterministic_collective_reduce(
    handle: int,
    gathered: torch.Tensor,
    output: torch.Tensor,
    slice_offset: int,
) -> None: ...

def rmsnorm_ascend(
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> torch.Tensor: ...

