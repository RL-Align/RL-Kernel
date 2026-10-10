# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Experimental training bridge for the pinned vLLM GDN decode provider.

Forward uses the real packed decode kernel, with a private cache copy. Backward
recomputes the mathematical recurrence from the training inputs and preserves
initial-state gradients. This is a single operator bridge, not a model installer,
prefill/decode equivalence claim, or permission to bypass vLLM's BI guard.
"""

from functools import lru_cache
from importlib.metadata import version

import torch
from torch.autograd.function import once_differentiable

from rl_engine.reference.linear_attn.gated_delta_rule import (
    GatedDeltaRuleRecurrentStepOp,
    _validate_state_indices,
)


@lru_cache(maxsize=1)
def _provider():
    if version("vllm") != "0.30.0":
        raise RuntimeError("Qwen3-Next GDN bridge requires the audited vLLM 0.30.0 provider")
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    return fused_recurrent_gated_delta_rule_packed_decode


def _validate(qkv, a, b, A_log, dt_bias, state, indices, num_k_heads):
    if not qkv.is_cuda or qkv.ndim != 2 or qkv.dtype != torch.bfloat16:
        raise ValueError("qkv must be CUDA BF16 [B, D]")
    if state.ndim != 4 or state.dtype != torch.float32 or state.shape[-2:] != (128, 128):
        raise ValueError("state must be FP32 [blocks, HV, 128, 128]")
    if isinstance(num_k_heads, bool) or not isinstance(num_k_heads, int) or num_k_heads <= 0:
        raise ValueError("num_k_heads must be a positive integer")
    batch, hv = qkv.shape[0], state.shape[1]
    if hv <= 0 or hv % num_k_heads or qkv.shape[1] != 2 * num_k_heads * 128 + hv * 128:
        raise ValueError("packed qkv width and state heads are inconsistent")
    for name, tensor, shape, dtype in (
        ("a", a, (batch, hv), qkv.dtype),
        ("b", b, (batch, hv), qkv.dtype),
        ("A_log", A_log, (hv,), torch.float32),
        ("dt_bias", dt_bias, (hv,), torch.float32),
    ):
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != qkv.device:
            raise ValueError(f"{name} has an incompatible shape, dtype or device")
    if state.device != qkv.device:
        raise ValueError("state must share the qkv device")
    _validate_state_indices(indices, batch, state.shape[0], qkv.device)


class _PackedDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, a, b, A_log, dt_bias, state, indices, scale, num_k_heads):
        _validate(qkv, a, b, A_log, dt_bias, state, indices, num_k_heads)
        ctx.save_for_backward(qkv, a, b, A_log, dt_bias, state, indices)
        ctx.scale, ctx.num_k_heads = float(scale), num_k_heads
        new_state = state.contiguous().clone()
        out = torch.zeros(qkv.shape[0], 1, state.shape[1], 128, device=qkv.device, dtype=qkv.dtype)
        if qkv.shape[0]:
            _provider()(
                qkv.contiguous(),
                a.contiguous(),
                b.contiguous(),
                A_log.contiguous(),
                dt_bias.contiguous(),
                float(scale),
                new_state,
                out,
                indices.contiguous(),
                use_qk_l2norm_in_kernel=True,
            )
        return out, new_state

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out, grad_state):
        *saved, indices = ctx.saved_tensors
        with torch.enable_grad():
            inputs = [tensor.detach().requires_grad_(True) for tensor in saved]
            out, state = GatedDeltaRuleRecurrentStepOp()(
                *inputs,
                indices,
                scale=ctx.scale,
                num_k_heads=ctx.num_k_heads,
            )
            outputs_and_grads = [
                (value, torch.zeros_like(value) if grad is None else grad)
                for value, grad in ((out, grad_out), (state, grad_state))
                if value.requires_grad
            ]
            grads = torch.autograd.grad(
                tuple(value for value, _ in outputs_and_grads),
                inputs,
                tuple(grad for _, grad in outputs_and_grads),
                allow_unused=True,
            )
            grads = tuple(
                torch.zeros_like(value) if grad is None else grad
                for value, grad in zip(inputs, grads)
            )

        return (*grads, None, None, None)


def packed_decode_training_step(
    qkv, a, b, A_log, dt_bias, state, indices, *, num_k_heads, scale=None
):
    """One train-side token; pass returned state onward without detaching it.

    Supports first-order gradients only. FP32 recurrent state and Qwen3-Next
    128-wide heads are mandatory. This function never reads rollout tensors.
    """
    return _PackedDecode.apply(
        qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        indices,
        128**-0.5 if scale is None else float(scale),
        num_k_heads,
    )


def packed_recurrent_sequence(
    qkv, a, b, A_log, dt_bias, state, indices, cu_seqlens, *, num_k_heads, scale=None
):
    """Apply the decode provider to packed, variable-length sequences.

    ``cu_seqlens`` is CPU int32/int64 metadata delimiting contiguous sequences
    in ``qkv[T, D]``. ``indices[B]`` maps each sequence to a distinct positive
    cache slot; slot zero remains reserved by the decode provider. Returning
    the entire FP32 state bank makes chunk continuation and sequence reordering
    explicit. Callers must pass that state onward without detaching it for
    training. No rollout cache is read and no batch-invariance capability is
    registered. This intentionally slow recurrence is an integration reference,
    not equivalence evidence for the upstream chunk-prefill kernel.
    """
    if (
        not isinstance(cu_seqlens, torch.Tensor)
        or cu_seqlens.device.type != "cpu"
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
        or cu_seqlens.ndim != 1
        or cu_seqlens.numel() < 1
    ):
        raise ValueError("cu_seqlens must be a nonempty CPU int32/int64 vector")
    boundaries = cu_seqlens.tolist()
    if qkv.ndim != 2 or boundaries[0] != 0 or boundaries[-1] != qkv.shape[0]:
        raise ValueError("cu_seqlens must span all packed qkv tokens, starting at zero")
    lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    if any(length < 0 for length in lengths):
        raise ValueError("cu_seqlens must be nondecreasing")
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != qkv.shape[0] or b.shape[0] != qkv.shape[0]:
        raise ValueError("a and b must have one row per packed token")
    _validate(qkv[:0], a[:0], b[:0], A_log, dt_bias, state, indices[:0], num_k_heads)
    _validate_state_indices(indices, len(lengths), state.shape[0], qkv.device)
    if bool((indices <= 0).any().item()):
        raise ValueError("sequence cache indices must be positive; zero is reserved")
    outputs, positions = [], []
    for position in range(max(lengths, default=0)):
        active = [seq for seq, length in enumerate(lengths) if position < length]
        rows = [boundaries[seq] + position for seq in active]
        row_ids = torch.tensor(rows, device=qkv.device, dtype=torch.int64)
        seq_ids = torch.tensor(active, device=qkv.device, dtype=torch.int64)
        out, state = packed_decode_training_step(
            qkv.index_select(0, row_ids),
            a.index_select(0, row_ids),
            b.index_select(0, row_ids),
            A_log,
            dt_bias,
            state,
            indices.index_select(0, seq_ids),
            num_k_heads=num_k_heads,
            scale=scale,
        )
        outputs.append(out[:, 0])
        positions.extend(rows)
    if not outputs:
        return qkv.new_empty((0, state.shape[1], 128)), state
    # Restore packed sequence order after the time-major recurrent traversal.
    inverse = [0] * len(positions)
    for source, destination in enumerate(positions):
        inverse[destination] = source
    order = torch.tensor(inverse, device=qkv.device, dtype=torch.int64)
    return torch.cat(outputs).index_select(0, order), state
