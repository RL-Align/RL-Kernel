# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini global attention contract (INK-15)."""

from __future__ import annotations

import torch


def full_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    relative_bias: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    output_gate: torch.Tensor | None = None,
    query_documents: torch.Tensor | None = None,
    key_documents: torch.Tensor | None = None,
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute global GQA attention with sink, gate, and document boundaries.

    Q/K/V use ``[batch, sequence, heads, head_dim]``. K/V heads are repeated
    to match query heads. Scores and softmax run in FP32. ``sink`` is a learned
    virtual-key logit per query head, while ``output_gate`` is a learned
    per-head sigmoid gate applied after the value reduction.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("attention tensors must have shape [batch, sequence, heads, dim]")
    batch, query_len, query_heads, dim = query.shape
    key_batch, key_len, key_heads, key_dim = key.shape
    if value.shape != (key_batch, key_len, key_heads, dim):
        raise ValueError("key and value must share batch, sequence, heads and head dimension")
    if key_batch != batch or dim != key_dim or query_heads % key_heads:
        raise ValueError("attention geometry does not support the requested GQA layout")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key and value must share a device")

    if relative_bias is not None:
        if relative_bias.shape == (query_heads, query_len, key_len):
            bias = relative_bias.float().unsqueeze(0)
        elif relative_bias.shape == (batch, query_heads, query_len, key_len):
            bias = relative_bias.float()
        else:
            raise ValueError(
                "relative bias must have shape [heads, query, key] or [batch, heads, query, key]"
            )
        if bias.device != query.device:
            raise ValueError("relative bias must share the query device")
    else:
        bias = None

    if sink is not None and sink.shape != (query_heads,):
        raise ValueError("sink must have shape [query_heads]")
    if output_gate is not None and output_gate.shape != (query_heads,):
        raise ValueError("output gate must have shape [query_heads]")
    for tensor, name in ((sink, "sink"), (output_gate, "output gate")):
        if tensor is not None and tensor.device != query.device:
            raise ValueError(f"{name} must share the query device")

    repeat = query_heads // key_heads
    key_gqa = key.repeat_interleave(repeat, dim=2)
    value_gqa = value.repeat_interleave(repeat, dim=2)
    query_fp32 = query.float().transpose(1, 2)
    key_fp32 = key_gqa.float().transpose(1, 2)
    value_fp32 = value_gqa.float().transpose(1, 2)
    attention_scale = float(scale) if scale is not None else dim**-0.5
    scores = torch.matmul(query_fp32, key_fp32.transpose(-2, -1)) * attention_scale
    if bias is not None:
        scores = scores + bias

    allowed = torch.ones((batch, query_len, key_len), device=query.device, dtype=torch.bool)
    if causal:
        query_positions = torch.arange(query_len, device=query.device).view(query_len, 1)
        key_positions = torch.arange(key_len, device=query.device).view(1, key_len)
        offset = max(key_len - query_len, 0)
        allowed = allowed & (key_positions <= query_positions + offset)
    if query_documents is not None or key_documents is not None:
        if query_documents is None or key_documents is None:
            raise ValueError("query and key document IDs must be provided together")
        if query_documents.shape != (batch, query_len) or key_documents.shape != (batch, key_len):
            raise ValueError("document IDs must match query and key batch/sequence dimensions")
        allowed = allowed & query_documents.unsqueeze(-1).eq(key_documents.unsqueeze(-2))

    scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
    if sink is not None:
        sink_scores = sink.float().view(1, query_heads, 1, 1).expand(batch, -1, query_len, -1)
        scores = torch.cat((scores, sink_scores), dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities[..., :key_len], value_fp32)
    if output_gate is not None:
        output = output * torch.sigmoid(output_gate.float()).view(1, query_heads, 1, 1)
    return output.transpose(1, 2).to(dtype=query.dtype)


class _NativeFullAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        relative_bias,
        sink,
        output_gate,
        query_documents,
        key_documents,
        causal,
        scale,
    ):
        from rl_engine.backends.extension import _C

        output, probabilities, pre_gate, sink_probability = _C.inkling2_full_attention_forward(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            relative_bias.contiguous(),
            sink.contiguous(),
            output_gate.contiguous(),
            query_documents.contiguous(),
            key_documents.contiguous(),
            bool(causal),
            float(scale),
        )
        ctx.save_for_backward(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            probabilities,
            pre_gate,
            sink_probability,
            relative_bias.contiguous(),
            sink.contiguous(),
            output_gate.contiguous(),
            query_documents.contiguous(),
            key_documents.contiguous(),
        )
        ctx.causal = bool(causal)
        ctx.scale = float(scale)
        ctx.has_relative_bias = relative_bias.numel() != 0
        ctx.has_sink = sink.numel() != 0
        ctx.has_output_gate = output_gate.numel() != 0
        ctx.need_relative_bias = ctx.needs_input_grad[3]
        ctx.need_sink = ctx.needs_input_grad[4]
        ctx.need_output_gate = ctx.needs_input_grad[5]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from rl_engine.backends.extension import _C

        (
            query,
            key,
            value,
            probabilities,
            pre_gate,
            sink_probability,
            relative_bias,
            sink,
            output_gate,
            query_documents,
            key_documents,
        ) = ctx.saved_tensors
        d_query, d_key, d_value, d_bias, d_sink, d_gate = _C.inkling2_full_attention_backward(
            grad_output.contiguous(),
            query,
            key,
            value,
            probabilities,
            pre_gate,
            sink_probability,
            relative_bias,
            sink,
            output_gate,
            query_documents,
            key_documents,
            ctx.causal,
            ctx.scale,
        )
        if not ctx.has_relative_bias or not ctx.need_relative_bias:
            d_bias = None
        if not ctx.has_sink or not ctx.need_sink:
            d_sink = None
        if not ctx.has_output_gate or not ctx.need_output_gate:
            d_gate = None
        return d_query, d_key, d_value, d_bias, d_sink, d_gate, None, None, None, None


def full_attention_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    relative_bias: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    output_gate: torch.Tensor | None = None,
    query_documents: torch.Tensor | None = None,
    key_documents: torch.Tensor | None = None,
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """Run the native CUDA Inkling 2 Mini full-attention kernel."""
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_full_attention_forward"):
        raise RuntimeError("Inkling 2 full attention CUDA extension is unavailable")
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("native attention tensors must be [batch, sequence, heads, dim]")
    batch, query_len, query_heads, dim = query.shape
    key_len, key_heads = key.shape[1], key.shape[2]
    if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3] or query_heads % key_heads:
        raise ValueError("native attention geometry does not support the requested GQA layout")
    if query.device != key.device or query.device != value.device:
        raise ValueError("native attention tensors must share device")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("native attention supports float16, bfloat16, and float32")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("native attention tensors must share dtype")
    if relative_bias is None:
        relative_bias = torch.empty(0, device=query.device, dtype=torch.float32)
    elif relative_bias.shape == (query_heads, query_len, key_len):
        relative_bias = relative_bias.float().unsqueeze(0).expand(
            batch, -1, -1, -1
        ).contiguous()
    elif relative_bias.shape == (batch, query_heads, query_len, key_len):
        relative_bias = relative_bias.float().contiguous()
    else:
        raise ValueError(
            "relative bias must have shape [heads, query, key] or [batch, heads, query, key]"
        )
    if sink is None:
        sink = torch.empty(0, device=query.device, dtype=torch.float32)
    elif sink.shape != (query_heads,):
        raise ValueError("sink must have shape [query_heads]")
    else:
        sink = sink.float().contiguous()
    if output_gate is None:
        output_gate = torch.empty(0, device=query.device, dtype=torch.float32)
    elif output_gate.shape != (query_heads,):
        raise ValueError("output gate must have shape [query_heads]")
    else:
        output_gate = output_gate.float().contiguous()
    if query_documents is None and key_documents is None:
        query_documents = torch.empty(0, device=query.device, dtype=torch.int64)
        key_documents = torch.empty(0, device=query.device, dtype=torch.int64)
    elif query_documents is None or key_documents is None:
        raise ValueError("query and key document IDs must be provided together")
    else:
        if query_documents.shape != (batch, query_len) or key_documents.shape != (batch, key_len):
            raise ValueError("document IDs must match query and key batch/sequence dimensions")
        query_documents = query_documents.to(device=query.device, dtype=torch.int64).contiguous()
        key_documents = key_documents.to(device=query.device, dtype=torch.int64).contiguous()
    attention_scale = float(scale) if scale is not None else dim**-0.5
    return _NativeFullAttentionFunction.apply(
        query,
        key,
        value,
        relative_bias,
        sink,
        output_gate,
        query_documents,
        key_documents,
        bool(causal),
        attention_scale,
    )
