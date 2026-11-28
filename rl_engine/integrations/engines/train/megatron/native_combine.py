# SPDX-License-Identifier: Apache-2.0
"""Native dispatch mapping into the existing fixed FP32 Top6 P6 operator."""

import types

import torch
from torch.autograd.function import once_differentiable

from rl_engine.contracts.operators.fixed_output_combine import prepare_fixed_output_combine
from rl_engine.contracts.operators.moe import MoEOutputCombineInputs, MoEOutputCombinePlan
from rl_engine.ops.moe.output_combine import output_combine


def prepare_combine_rows(expert_output, tokens, experts, num_tokens, run_id):
    order = (tokens * 256 + experts).argsort(stable=True)
    slots = torch.empty_like(tokens)
    slots[order] = torch.arange(tokens.numel(), device=tokens.device, dtype=torch.int64) % 6
    plan = MoEOutputCombinePlan(6, expert_output.shape[1], expert_weight_applied=True)
    inputs = MoEOutputCombineInputs(
        expert_output,
        tokens,
        experts,
        slots,
        mapping_run_id=run_id,
        producer_backend="RLK.native.W4A16",
        producer_build_fingerprint="dsv4-fixed-p6-expert-ascending-v1",
        num_tokens=num_tokens,
    )
    prepared = prepare_fixed_output_combine(
        plan, inputs, expected_mapping_run_id=run_id, num_experts=256
    )
    return prepared


def combine_rows(expert_output, tokens, experts, num_tokens, run_id):
    prepared = prepare_combine_rows(expert_output, tokens, experts, num_tokens, run_id)
    return output_combine(prepared, expert_output, mapping_run_id=run_id)


class _FixedTop6RowGather(torch.autograd.Function):
    """Forward: the native ``index_select`` rows. Backward: the fixed P6 combine.

    Autograd's ``index_select`` VJP is a BF16 atomic ``index_add_`` of each token's
    six routed rows, so its rounding follows hardware arrival order. The VJP of a
    Top-6 gather is exactly the P6 combine of the row gradients: FP32 in
    expert-ascending slot order with one BF16 store, over the same expert-major rows.
    """

    @staticmethod
    def forward(ctx, hidden, sorted_indices, experts, run_id):
        ctx.save_for_backward(sorted_indices, experts)
        ctx.num_tokens, ctx.run_id = hidden.shape[0], run_id
        return hidden.index_select(0, sorted_indices)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        tokens, experts = ctx.saved_tensors
        dx = combine_rows(grad.contiguous(), tokens, experts, ctx.num_tokens, ctx.run_id)
        return dx, None, None, None


def _with_globals(function, **names):
    cloned = types.FunctionType(
        function.__code__,
        dict(function.__globals__, **names),
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    cloned.__kwdefaults__ = function.__kwdefaults__
    return cloned


def _bind_fixed_permute_vjp(dispatcher):
    original = dispatcher.dispatch_preprocess.__func__
    if "permute" not in original.__code__.co_names:
        raise ValueError("expected original native all-to-all permute boundary")
    native_permute = original.__globals__["permute"]
    calls = [0]

    def permute(
        tokens,
        routing_map,
        probs=None,
        num_out_tokens=None,
        fused=False,
        drop_and_pad=False,
        tokens_per_expert=None,
        align_size=0,
        return_batch_invariant_inverse_map=False,
    ):
        if (
            fused
            or drop_and_pad
            or tokens_per_expert is not None
            or align_size
            or return_batch_invariant_inverse_map
        ):
            raise ValueError("fixed P6 permute VJP supports ordinary dropless native dispatch")
        # Native routing metadata and permuted probs, unchanged; the zero-width
        # proxy only skips the native gather, which the function below repeats.
        proxy, permuted_probs, sorted_indices, inverse_map, counts = native_permute(
            tokens.detach()[:, :0], routing_map, probs=probs, num_out_tokens=num_out_tokens
        )
        if proxy.shape != (sorted_indices.numel(), 0) or inverse_map is not None:
            raise ValueError("native permute metadata changed shape")
        experts, rows = routing_map.T.nonzero(as_tuple=True)
        if not torch.equal(sorted_indices, rows):
            raise ValueError("native sorted rows do not match actual expert-major routing map")
        calls[0] += 1
        permuted = _FixedTop6RowGather.apply(
            tokens, sorted_indices, experts.long(), f"native-p6-permute-vjp-{calls[0]}"
        )
        if calls[0] == 1:
            print(
                "RLK_NATIVE_FIXED_PERMUTE_VJP_BOUND",
                __file__,
                "index_select forward; VJP = output_combine FP32 expert-ascending slots",
                flush=True,
            )
        return permuted, permuted_probs, sorted_indices, inverse_map, counts

    dispatcher.dispatch_preprocess = types.MethodType(
        _with_globals(original, permute=permute), dispatcher
    )


def bind_native_combine(moe):
    dispatcher = moe.token_dispatcher
    original = dispatcher.combine_postprocess.__func__
    if "unpermute" not in original.__code__.co_names:
        raise ValueError("expected original native all-to-all unpermute boundary")
    _bind_fixed_permute_vjp(dispatcher)
    calls = [0]

    def unpermute(
        permuted_tokens,
        sorted_indices,
        restore_shape,
        probs=None,
        routing_map=None,
        fused=False,
        drop_and_pad=False,
        pad_offsets=None,
        batch_invariant_inverse_map=None,
        ep_invariant_combine=False,
    ):
        if (
            probs is not None
            or fused
            or drop_and_pad
            or pad_offsets is not None
            or batch_invariant_inverse_map is not None
            or ep_invariant_combine
        ):
            raise ValueError("initial P6 adapter supports ordinary weight-applied native dispatch")
        experts, tokens = routing_map.T.nonzero(as_tuple=True)
        if not torch.equal(sorted_indices, tokens):
            raise ValueError("native sorted rows do not match actual expert-major routing map")
        calls[0] += 1
        result = combine_rows(
            permuted_tokens,
            tokens.long(),
            experts.long(),
            routing_map.shape[0],
            f"native-p6-{calls[0]}",
        )
        if calls[0] == 1:
            print(
                "RLK_NATIVE_FIXED_COMBINE_EXECUTED",
                __file__,
                "output_combine FP32 expert-ascending slots, one BF16 store",
                flush=True,
            )
        return result.reshape(restore_shape)

    dispatcher.combine_postprocess = types.MethodType(
        _with_globals(original, unpermute=unpermute), dispatcher
    )
    return dispatcher
