# SPDX-License-Identifier: Apache-2.0
"""INK-13: token-dependent virtual attention sink and output gating."""

import torch


def apply_attention_sink_gate(
    context: torch.Tensor,
    lse: torch.Tensor,
    sink_logits: torch.Tensor,
    gate_logits: torch.Tensor,
) -> torch.Tensor:
    """Apply FP32 sigmoid(gate), then sigmoid(LSE - sink), in that order.

    Context is the real-key softmax result after the attention kernel's
    declared output cast. LSE is the natural-log real-key denominator in
    FP32, excluding the sink. Sink logits are [T,H]; output gate logits are
    [T,H,D] or [T,H*D]. The output stays FP32 until INK-17's projection.
    This portable epilogue does not fuse a virtual value into attention.
    """
    if context.ndim != 3 or context.numel() == 0:
        raise ValueError("context must be nonempty [tokens, heads, head_dim]")
    tokens, heads, dim = context.shape
    if lse.shape != (tokens, heads) or sink_logits.shape != (tokens, heads):
        raise ValueError("LSE and sink logits must be [tokens, heads]")
    if gate_logits.shape not in (context.shape, (tokens, heads * dim)):
        raise ValueError("gate logits must be per-token, per-head, per-channel")
    if any(t.device != context.device for t in (lse, sink_logits, gate_logits)):
        raise ValueError("attention epilogue tensors must share a device")
    if lse.dtype != torch.float32:
        raise TypeError("LSE must be FP32")
    if any(t.dtype not in (torch.float32, torch.bfloat16) for t in (
        context, sink_logits, gate_logits
    )):
        raise TypeError("context and projected logits must be BF16 or FP32")
    if context.device.type not in ("cpu", "cuda"):
        raise RuntimeError("expected CPU or CUDA/ROCm")
    gate = torch.sigmoid(gate_logits.float()).reshape_as(context)
    sink_scale = torch.sigmoid(lse - sink_logits.float())
    return (context.float() * gate) * sink_scale.unsqueeze(-1)
