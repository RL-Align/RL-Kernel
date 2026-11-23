# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Training bridge for the pinned vLLM GDN decode provider.

A single token uses the real packed decode kernel, with a private cache copy.
Packed sequences run the same per-token arithmetic with the time loop inside one
launch. Backward recomputes the mathematical recurrence from the training inputs
and preserves initial-state gradients. This is a single operator bridge, not a
model installer, or permission to bypass vLLM's BI guard.
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


def _validated_boundaries(qkv, a, b, A_log, dt_bias, state, indices, cu_seqlens, num_k_heads):
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
    return tuple(boundaries)


def _decode_loop(qkv, a, b, A_log, dt_bias, state, indices, boundaries, *, num_k_heads, scale):
    """The decode kernel once per token position (time-major), as autograd steps."""
    lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
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


@lru_cache(maxsize=1)
def _sequence_provider():
    """vLLM's recurrent kernel with the time loop inside one launch, and its gating.

    ``fused_recurrent_gated_delta_rule`` runs the same per-token update as the packed
    decode kernel (L2-normalised q/k, scale, exp(g) decay, delta update, output), with
    the same tiles (BK=128, BV=32, one warp, three stages), but takes g and beta as
    inputs. The gating kernel below computes them with the decode kernel's own
    expressions; recomputing them in PyTorch changes low bits.
    """
    if version("vllm") != "0.30.0":
        raise RuntimeError("Qwen3-Next GDN bridge requires the audited vLLM 0.30.0 provider")
    import triton
    import triton.language as tl
    from vllm.third_party.flash_linear_attention.ops import fused_recurrent_gated_delta_rule

    @triton.jit
    def gating_kernel(
        a,
        b,
        A_log,
        dt_bias,
        g,
        beta,
        stride_a_tok,
        stride_b_tok,
        HV: tl.constexpr,
        BLOCK_HV: tl.constexpr,
        SOFTPLUS_THRESHOLD: tl.constexpr,
    ):
        # Same loads, casts and expressions as
        # fused_recurrent_gated_delta_rule_packed_decode_kernel in vLLM 0.30.0.
        i_n = tl.program_id(0)
        i_hv = tl.arange(0, BLOCK_HV)
        mask = i_hv < HV
        a_val = tl.load(a + i_n * stride_a_tok + i_hv, mask=mask, other=0).to(tl.float32)
        b_val = tl.load(b + i_n * stride_b_tok + i_hv, mask=mask, other=0).to(tl.float32)
        A_log_val = tl.load(A_log + i_hv, mask=mask, other=0).to(tl.float32)
        dt_bias_val = tl.load(dt_bias + i_hv, mask=mask, other=0).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        g_val = -tl.exp(A_log_val) * softplus_x
        beta_val = tl.sigmoid(b_val)
        tl.store(g + i_n * HV + i_hv, g_val, mask=mask)
        tl.store(beta + i_n * HV + i_hv, beta_val, mask=mask)

    def gating(a, b, A_log, dt_bias):
        tokens, hv = a.shape
        g = torch.empty((tokens, hv), device=a.device, dtype=torch.float32)
        beta = torch.empty_like(g)
        if tokens:
            gating_kernel[(tokens,)](
                a,
                b,
                A_log,
                dt_bias,
                g,
                beta,
                a.stride(0),
                b.stride(0),
                HV=hv,
                BLOCK_HV=triton.next_power_of_2(hv),
                SOFTPLUS_THRESHOLD=20.0,
            )
        return g, beta

    return fused_recurrent_gated_delta_rule, gating


def _fused_sequence(qkv, a, b, A_log, dt_bias, state, indices, boundaries, *, num_k_heads, scale):
    lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    if not max(lengths, default=0):
        return qkv.new_empty((0, state.shape[1], 128)), state
    recurrent, gating = _sequence_provider()
    tokens, heads = qkv.shape[0], state.shape[1]
    width = num_k_heads * 128
    qkv = qkv.contiguous()
    q = qkv[:, :width].reshape(1, tokens, num_k_heads, 128).contiguous()
    k = qkv[:, width : 2 * width].reshape(1, tokens, num_k_heads, 128).contiguous()
    v = qkv[:, 2 * width :].reshape(1, tokens, heads, 128).contiguous()
    g, beta = gating(a.contiguous(), b.contiguous(), A_log.contiguous(), dt_bias.contiguous())
    new_state = state.contiguous().clone()
    # Each token of sequence n stores its running state into slot indices[n]; the
    # last token's store is the final state, as after the decode loop.
    slots = indices.to(torch.int64)[:, None].expand(len(lengths), max(lengths)).contiguous()
    output, final_state = recurrent(
        q,
        k,
        v,
        g=g.reshape(1, tokens, heads),
        beta=beta.reshape(1, tokens, heads),
        scale=scale,
        initial_state=new_state,
        inplace_final_state=True,
        cu_seqlens=torch.tensor(boundaries, device=qkv.device, dtype=torch.int64),
        ssm_state_indices=slots,
        use_qk_l2norm_in_kernel=True,
    )
    # vLLM 0.30.0 returns initial_state itself when inplace_final_state is set; a version
    # that returned a separate tensor would leave new_state stale, so refuse it.
    if final_state.data_ptr() != new_state.data_ptr():
        raise RuntimeError("fused GDN recurrence did not update the state bank in place")
    return output.reshape(tokens, heads, 128), new_state


class _PackedSequence(torch.autograd.Function):
    """Forward in one launch per layer; backward differentiates the per-token decode loop.

    Both forwards give the same bits (checked in check_qwen3_next_gdn_sequence.py), so
    the gradient is the one the decode loop has always produced.
    """

    @staticmethod
    def forward(ctx, qkv, a, b, A_log, dt_bias, state, indices, boundaries, scale, num_k_heads):
        ctx.save_for_backward(qkv, a, b, A_log, dt_bias, state, indices)
        ctx.boundaries, ctx.scale, ctx.num_k_heads = boundaries, scale, num_k_heads
        return _fused_sequence(
            qkv,
            a,
            b,
            A_log,
            dt_bias,
            state,
            indices,
            boundaries,
            num_k_heads=num_k_heads,
            scale=scale,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out, grad_state):
        *saved, indices = ctx.saved_tensors
        with torch.enable_grad():
            inputs = [
                tensor.detach().requires_grad_(needed)
                for tensor, needed in zip(saved, ctx.needs_input_grad)
            ]
            out, state = _decode_loop(
                *inputs,
                indices,
                ctx.boundaries,
                num_k_heads=ctx.num_k_heads,
                scale=ctx.scale,
            )
            pairs = [
                (value, torch.zeros_like(value) if grad is None else grad)
                for value, grad in ((out, grad_out), (state, grad_state))
                if value.requires_grad
            ]
            wanted = [tensor for tensor in inputs if tensor.requires_grad]
            grads = iter(
                torch.autograd.grad(
                    tuple(value for value, _ in pairs),
                    wanted,
                    tuple(grad for _, grad in pairs),
                    allow_unused=True,
                )
                if pairs and wanted
                else ()
            )
        result = [next(grads, None) if tensor.requires_grad else None for tensor in inputs]
        return (*result, None, None, None, None)


def packed_recurrent_sequence(
    qkv, a, b, A_log, dt_bias, state, indices, cu_seqlens, *, num_k_heads, scale=None
):
    """Run the GDN recurrence over packed, variable-length sequences.

    ``cu_seqlens`` is CPU int32/int64 metadata delimiting contiguous sequences
    in ``qkv[T, D]``. ``indices[B]`` maps each sequence to a distinct positive
    cache slot; slot zero remains reserved by the decode provider. Returning
    the entire FP32 state bank makes chunk continuation and sequence reordering
    explicit; the caller's ``state`` is not modified. Callers must pass the
    returned state onward without detaching it for training. No rollout cache is
    read and no batch-invariance capability is registered.

    The forward runs every sequence's tokens inside one kernel launch with the
    decode kernel's per-token arithmetic, so its bits equal one decode step per
    token (``decode_loop_sequence``); prompt prefill and decode stay on one
    numerical path. It is not the upstream chunk-prefill kernel.
    """
    boundaries = _validated_boundaries(
        qkv, a, b, A_log, dt_bias, state, indices, cu_seqlens, num_k_heads
    )
    return _PackedSequence.apply(
        qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        indices,
        boundaries,
        128**-0.5 if scale is None else float(scale),
        num_k_heads,
    )


def decode_loop_sequence(
    qkv, a, b, A_log, dt_bias, state, indices, cu_seqlens, *, num_k_heads, scale=None
):
    """The decode kernel once per token position: the reference the fused forward equals."""
    boundaries = _validated_boundaries(
        qkv, a, b, A_log, dt_bias, state, indices, cu_seqlens, num_k_heads
    )
    return _decode_loop(
        qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        indices,
        boundaries,
        num_k_heads=num_k_heads,
        scale=128**-0.5 if scale is None else float(scale),
    )
