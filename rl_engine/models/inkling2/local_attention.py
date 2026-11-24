# SPDX-License-Identifier: Apache-2.0
"""INK-14: bounded, fixed-window causal GQA attention candidate."""

import torch

from rl_engine.ops.norm.fixed_tree import fixed_sum_last

LOCAL_WINDOW = 1024


def local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    relative_logits: torch.Tensor,
    query_key_indices: torch.Tensor,
    key_documents: torch.Tensor,
    *,
    window: int = LOCAL_WINDOW,
    query_chunk: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pre-sink context (input dtype) and real-key natural-log LSE (FP32).

    Token-major Q [Q,H,D], K/V [K,Hkv,D], relative logits [Q,H,extent].
    Query indices refer to their own entries in a contiguous logical KV
    sequence, including the current token. Packed documents are contiguous
    with unique IDs. Each window includes the current token and W-1 older
    slots. Positions within a document advance by one. Prefill and decode
    use the same oldest-to-newest fixed reduction graph with padded slots.

    Eager CPU/CUDA/ROCm arithmetic candidate. No full Q*K score matrix is
    materialized. Backward uses eager autograd (gather VJPs are NOT certified
    deterministic on GPU); production paged kernels and performance remain
    separate work. INK-13 consumes the real-key LSE and context.
    """
    if query.ndim != 3 or key.ndim != 3 or key.shape != value.shape:
        raise ValueError("expected token-major Q and matching K/V [tokens, heads, dim]")
    nq, hq, dim = query.shape
    nk, hk, kd = key.shape
    if min(nq, hq, dim, nk, hk, kd) <= 0 or dim != kd or hq % hk:
        raise ValueError("invalid GQA geometry")
    if not 1 <= window <= LOCAL_WINDOW or query_chunk <= 0:
        raise ValueError("invalid window or query chunk")
    if relative_logits.ndim != 3 or relative_logits.shape[:2] != (nq, hq):
        raise ValueError("relative logits must be [queries, query_heads, extent]")
    extent = relative_logits.shape[-1]
    if extent <= 0 or extent > LOCAL_WINDOW:
        raise ValueError("invalid relative-logit extent")
    if query_key_indices.shape != (nq,) or key_documents.shape != (nk,):
        raise ValueError("query indices or key document IDs have incorrect shape")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (query_key_indices, key_documents)):
        raise TypeError("query indices and key document IDs must be integers")
    if any(t.device != query.device for t in (
        key, value, relative_logits, query_key_indices, key_documents
    )):
        raise ValueError("attention tensors must share a device")
    if query.device.type not in ("cpu", "cuda"):
        raise RuntimeError("expected CPU or CUDA/ROCm")
    if query.dtype not in (torch.bfloat16, torch.float32) or key.dtype != query.dtype:
        raise TypeError("Q/K/V must share BF16 or FP32 dtype")
    if value.dtype != query.dtype or relative_logits.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("invalid value or relative-logit dtype")
    qids = query_key_indices.long()
    if bool(((qids < 0) | (qids >= nk)).any()):
        raise ValueError("query index must refer to an existing KV entry")

    distances = torch.arange(window - 1, -1, -1, device=query.device)
    heads = torch.arange(hq, device=query.device) // (hq // hk)
    bias_indices = distances.clamp(max=extent - 1)
    bias_valid = distances < extent
    outputs = []
    lses = []
    for start in range(0, nq, query_chunk):
        stop = min(start + query_chunk, nq)
        ids = qids[start:stop]
        indices = ids[:, None] - distances[None, :]
        safe_indices = indices.clamp(min=0)
        docs = key_documents[ids]
        valid = (indices >= 0) & (key_documents[safe_indices] == docs[:, None])
        # [chunk, window, query_heads, dim] -> [chunk, query_heads, window, dim]
        k = key[safe_indices[:, :, None], heads[None, None, :]].transpose(1, 2).float()
        v = value[safe_indices[:, :, None], heads[None, None, :]].transpose(1, 2).float()
        q = query[start:stop].float().unsqueeze(2)
        scores = fixed_sum_last(q * k).squeeze(-1) * (dim**-0.5)
        bias = relative_logits[start:stop, :, bias_indices].float()
        scores = scores + torch.where(bias_valid, bias, torch.zeros_like(bias))
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        peak = scores.amax(-1, keepdim=True)
        exp = (scores - peak).exp()
        denom = fixed_sum_last(exp)
        probs = exp / denom
        weighted = probs.unsqueeze(-1) * v
        out = fixed_sum_last(weighted.transpose(-1, -2)).squeeze(-1)
        outputs.append(out.to(query.dtype))
        lses.append((peak + denom.log()).squeeze(-1))
    return torch.cat(outputs, dim=0), torch.cat(lses, dim=0)
