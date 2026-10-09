# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Differentiable, explicit-state bridge to the pinned vLLM causal-conv provider.

Forward is the real vLLM update kernel for both prefill chunks and decode.
Backward recomputes the reference recurrence from training inputs. This module
does not install model hooks or advertise backend batch-invariance support.
"""

from functools import lru_cache
from importlib.metadata import version

import torch
from torch.autograd.function import once_differentiable

from rl_engine.kernels.ops.pytorch.linear_attn.causal_conv1d import CausalConv1dUpdateOp
from rl_engine.kernels.ops.pytorch.linear_attn.gated_delta_rule import _validate_state_indices


@lru_cache(maxsize=1)
def _provider():
    if version("vllm") != "0.30.0":
        raise RuntimeError("Qwen3-Next conv bridge requires the audited vLLM 0.30.0 provider")
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    return causal_conv1d_update


def _lengths(x, state, weight, indices, cu_seqlens):
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or x.shape[1] == 0:
        raise ValueError("x must be CUDA BF16 [tokens, dim] with positive dim")
    if weight.shape != (x.shape[1], 4) or weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError("weight must be BF16 [dim, 4] on the input device")
    if (
        state.ndim != 3
        or state.shape[1:] != (x.shape[1], 3)
        or state.device != x.device
        or state.dtype != x.dtype
    ):
        raise ValueError("conv state must be BF16 [blocks, dim, 3] on the input device")
    if (
        cu_seqlens.ndim != 1
        or cu_seqlens.device.type != "cpu"
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
        or cu_seqlens.numel() < 1
    ):
        raise ValueError("cu_seqlens must be a nonempty CPU integer vector")
    boundaries = cu_seqlens.tolist()
    if boundaries[0] != 0 or boundaries[-1] != x.shape[0] or boundaries[-1] >= 2**31:
        raise ValueError("cu_seqlens must cover all input tokens and fit int32")
    lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    if any(length < 0 for length in lengths):
        raise ValueError("cu_seqlens must be nondecreasing")
    _validate_state_indices(indices, len(lengths), state.shape[0], x.device)
    if bool((indices <= 0).any().item()):
        raise ValueError("sequence cache indices must be positive; zero is reserved")
    return lengths


class _CausalConvSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, state, weight, indices, cu_seqlens):
        lengths = _lengths(x, state, weight, indices, cu_seqlens)
        ctx.save_for_backward(x, state, weight, indices, cu_seqlens)
        new_state = state.contiguous().clone()
        output = torch.zeros_like(x)
        if x.shape[0]:
            # The upstream varlen path subtracts (max_len - sequence_len) from
            # its effective cache length, which is for speculative cache layouts.
            # A width-1 ordinary cache must instead use the dense sequence path.
            # Equal-length groups need neither padding nor a speculative cache.
            boundaries = cu_seqlens.tolist()
            for length in sorted(set(lengths) - {0}):
                active = [i for i, size in enumerate(lengths) if size == length]
                packed = torch.stack(
                    [x[boundaries[seq] : boundaries[seq + 1]].t() for seq in active]
                )
                values = _provider()(
                    packed,
                    new_state,
                    weight.contiguous(),
                    activation="silu",
                    conv_state_indices=indices[active].to(torch.int32).contiguous(),
                    out=torch.empty_like(packed),
                )
                for row, seq in enumerate(active):
                    output[boundaries[seq] : boundaries[seq + 1]] = values[row].t()
        return output, new_state

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_state):
        x, state, weight, indices, cu_seqlens = ctx.saved_tensors
        with torch.enable_grad():
            inputs = [tensor.detach().requires_grad_(True) for tensor in (x, state, weight)]
            x_ref, state_ref, weight_ref = inputs
            pieces = []
            boundaries = cu_seqlens.tolist()
            for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                for token in range(start, end):
                    out, state_ref = CausalConv1dUpdateOp()(
                        x_ref[token : token + 1], state_ref, weight_ref, indices[seq : seq + 1]
                    )
                    pieces.append(out)
            output = torch.cat(pieces) if pieces else x_ref[:0]
            grads = torch.autograd.grad(
                (output, state_ref),
                inputs,
                (
                    torch.zeros_like(output) if grad_output is None else grad_output,
                    torch.zeros_like(state_ref) if grad_state is None else grad_state,
                ),
                allow_unused=True,
            )
        return (
            *(
                torch.zeros_like(value) if grad is None else grad
                for value, grad in zip(inputs, grads)
            ),
            None,
            None,
        )


def causal_conv_sequence(x, state, weight, indices, cu_seqlens):
    """Return packed output and an independent cache; retain cache gradients.

    Qwen3-Next's width-four, bias-free SiLU convolution is the only contract.
    All tensors except CPU ``cu_seqlens`` live on the input CUDA device. The
    original cache and inputs are never mutated, including in inference mode.
    """
    return _CausalConvSequence.apply(x, state, weight, indices, cu_seqlens)
