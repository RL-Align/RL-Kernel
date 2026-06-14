# rl_engine/_C.pyi
# This file is a type stub for the compiled C++ extension module.
import torch

def fused_logp(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor: ...
def fused_logp_sm90(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...
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
