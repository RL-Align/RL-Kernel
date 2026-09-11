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

def embedding_ascend(
    token_ids: torch.Tensor,
    weight: torch.Tensor,
    output_fp32: bool,
) -> torch.Tensor: ...

def fused_logp_ascend(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor: ...
def lm_head_ascend(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    output_fp32: bool,
) -> torch.Tensor: ...

def fused_linear_logp_ascend(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    target: torch.Tensor,
) -> torch.Tensor: ...

def det_gemm_ascend_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor: ...
def det_gemm_ascend_fwd_rhs_transposed(
    a: torch.Tensor,
    bt: torch.Tensor,
) -> torch.Tensor: ...
def det_gemm_ascend_fwd_fp32(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor: ...
def det_gemm_ascend_da(
    dc: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor: ...
def det_gemm_ascend_db(
    a: torch.Tensor,
    dc: torch.Tensor,
) -> torch.Tensor: ...
def det_gemm_ascend_db_transposed(
    a: torch.Tensor,
    dc: torch.Tensor,
) -> torch.Tensor: ...
