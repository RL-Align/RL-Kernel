# rl_engine/_C.pyi
# This file is a type stub for the compiled C++ extension module.
import torch

def deterministic_collective_ipc_meta(
    tensor: torch.Tensor,
) -> tuple[list[int], int]: ...
def deterministic_collective_create(
    staging: torch.Tensor,
    handles: list[list[int]],
    offsets: list[int],
    rank: int,
) -> int: ...
def deterministic_collective_destroy(handle: int) -> None: ...
def deterministic_collective_stage(handle: int, input: torch.Tensor) -> None: ...
def deterministic_collective_all_reduce(handle: int, output: torch.Tensor) -> None: ...
def deterministic_collective_all_reduce_fused(
    handle: int, input: torch.Tensor, output: torch.Tensor
) -> None: ...
def deterministic_collective_reduce_scatter(handle: int, output: torch.Tensor) -> None: ...
def deterministic_collective_all_gather(handle: int, output: torch.Tensor) -> None: ...
def deterministic_collective_all_gather_fused(
    handle: int, input: torch.Tensor, output: torch.Tensor
) -> None: ...
def fused_logp(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor: ...
def fused_logp_sm90(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...
def batch_invariant_logp_sm90(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> list[torch.Tensor]: ...
def fused_linear_logp_sm90(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None,
) -> list[torch.Tensor]: ...
def fused_linear_logp_sm90_global_target(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def fused_linear_logp_sm90_backward(
    grad_logp: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    bias: torch.Tensor | None,
    vocab_start_index: int,
    compute_grad_hidden: bool,
    compute_grad_weight: bool,
    compute_grad_bias: bool,
    use_global_lse: bool,
) -> list[torch.Tensor]: ...
def linear_logp_probs_bf16_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def linear_logp_bf16_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def linear_logp_local_probs_bf16_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def linear_logp_local_bf16_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def linear_logp_probs_bf16_to_dlogits_(
    probs: torch.Tensor,
    target: torch.Tensor,
    grad_logp: torch.Tensor,
    vocab_start_index: int,
) -> torch.Tensor: ...
def linear_logp_local_probs_bf16_to_dlogits_(
    probs: torch.Tensor,
    target: torch.Tensor,
    grad_logp: torch.Tensor,
    local_lse: torch.Tensor,
    global_lse: torch.Tensor,
    vocab_start_index: int,
) -> torch.Tensor: ...
def linear_logp_logits_bf16_to_dlogits(
    logits: torch.Tensor,
    dlogits: torch.Tensor,
    target: torch.Tensor,
    grad_logp: torch.Tensor,
    lse: torch.Tensor,
    vocab_start_index: int,
) -> torch.Tensor: ...
def embedding_sm90_forward(token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...
def embedding_sm90_forward_fp32(token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor: ...
def lm_head_sm90_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor: ...
def lm_head_sm90_forward_fp32(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor: ...
def fused_logp_forward_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_fp32(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor: ...
def fused_logp_forward_indexed_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_indexed_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_online_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_online_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_online_indexed_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_online_indexed_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_logp(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor: ...
def deterministic_logp_forward_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_logp_forward_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_logp_forward_indexed_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_logp_forward_indexed_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    scale: float,
    key_padding_mask: torch.Tensor | None,
) -> list[torch.Tensor]:
    """Returns [out, lse, P]."""
    ...

def deterministic_attention_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    P: torch.Tensor,
    causal: bool,
    scale: float,
    key_padding_mask: torch.Tensor | None,
) -> list[torch.Tensor]: ...
def det_gemm_sm90_compiled() -> bool: ...
def det_gemm_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
def det_gemm_fwd_rhs_transposed(
    a: torch.Tensor,
    bt: torch.Tensor,
) -> torch.Tensor: ...
def det_gemm_da(dc: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
def det_gemm_db(a: torch.Tensor, dc: torch.Tensor) -> torch.Tensor: ...
def det_gemm_db_transposed(a: torch.Tensor, dc: torch.Tensor) -> torch.Tensor: ...
def silu_forward(x: torch.Tensor) -> torch.Tensor: ...
def silu_backward(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...
def swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor: ...
def swiglu_backward(
    dy: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> list[torch.Tensor]: ...
def p5_strict_gemm(a: torch.Tensor, b: torch.Tensor, trans_b: bool) -> torch.Tensor: ...
def p5_swiglu_shared_forward(z: torch.Tensor) -> torch.Tensor: ...
def p5_swiglu_shared_backward(dh: torch.Tensor, z: torch.Tensor) -> torch.Tensor: ...
def rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> list[torch.Tensor]: ...
def rmsnorm_backward_dx(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> torch.Tensor: ...
def rmsnorm_backward_dw(
    dy: torch.Tensor,
    x: torch.Tensor,
    rstd: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor: ...
