# SPDX-License-Identifier: Apache-2.0
"""Native dispatch mapping into the existing fixed FP32 Top6 P6 operator."""

import types

import torch

from rl_engine.contracts.operators.fixed_output_combine import prepare_fixed_output_combine
from rl_engine.contracts.operators.moe import MoEOutputCombineInputs, MoEOutputCombinePlan
from rl_engine.ops.moe.output_combine import output_combine


def combine_rows(expert_output, tokens, experts, num_tokens, run_id):
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
    return output_combine(prepared, expert_output, mapping_run_id=run_id)


def bind_native_combine(moe):
    dispatcher = moe.token_dispatcher
    original = dispatcher.combine_postprocess.__func__
    if "unpermute" not in original.__code__.co_names:
        raise ValueError("expected original native all-to-all unpermute boundary")
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

    cloned = types.FunctionType(
        original.__code__,
        dict(original.__globals__, unpermute=unpermute),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    cloned.__kwdefaults__ = original.__kwdefaults__
    dispatcher.combine_postprocess = types.MethodType(cloned, dispatcher)
    return dispatcher
