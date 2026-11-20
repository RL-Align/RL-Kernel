# SPDX-License-Identifier: Apache-2.0
"""Use the actor's expert-ascending FP32 combine at the pinned ROCm boundary.

The upstream 3.8 unfused W4A16 path uses an atomic index_add_. Its addition
order can change with batch shape and cross a BF16 rounding midpoint. Keep
upstream routing, packed GEMMs, activation and shared-expert handling; scope
only that one reduction to RLK's existing fixed Top6 operator.
"""

import inspect
import types

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rl_engine.contracts.operators.fixed_output_combine import (
    prepare_fixed_output_combine,
)
from rl_engine.contracts.operators.moe import (
    MoEOutputCombineInputs,
    MoEOutputCombinePlan,
)
from rl_engine.ops.moe.output_combine import output_combine


class _FixedReduce(TorchDispatchMode):
    def __init__(self, topk_ids):
        super().__init__()
        if topk_ids.ndim != 2 or topk_ids.shape[1] != 6:
            raise ValueError("fixed native combine requires an ordinary Top6 batch")
        self.ids = topk_ids
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is not torch.ops.aten.index_add_.default:
            return func(*args, **kwargs)
        target, dim, tokens, values = args
        if (
            self.calls
            or dim != 0
            or kwargs.get("alpha", 1) != 1
            or values.dtype != torch.float32
            or target.dtype != torch.float32
            or tuple(target.shape) != (self.ids.shape[0], values.shape[1])
            or tokens.shape != (self.ids.numel(),)
            or values.shape[0] != self.ids.numel()
        ):
            raise ValueError("unexpected native unfused expert reduction ABI")
        order = self.ids.reshape(-1).argsort(stable=True)
        expected_tokens = (
            torch.arange(self.ids.shape[0], device=tokens.device)[:, None]
            .expand_as(self.ids)
            .reshape(-1)[order]
        )
        if not torch.equal(tokens, expected_tokens):
            raise ValueError(
                "native expert-major rows do not match actual Top6 routing"
            )
        experts = self.ids.reshape(-1)[order].long()
        tokens = tokens.long()
        active = ((self.ids >= 0) & (self.ids < 256)).all(-1)
        inactive = (self.ids == -1).all(-1)
        if not bool((active | inactive).all()):
            raise ValueError("partial or invalid Top6 routing row")
        active_rows = torch.nonzero(active).flatten()
        num_tokens = active_rows.numel()
        if num_tokens == 0:
            self.calls += 1
            return target.zero_()
        if num_tokens != self.ids.shape[0]:
            selected = experts >= 0
            values, experts, tokens = (
                values[selected],
                experts[selected],
                tokens[selected],
            )
            dense_tokens = active.long().cumsum(0) - 1
            tokens = dense_tokens[tokens]
        canonical = (tokens * 256 + experts).argsort(stable=True)
        slots = torch.empty_like(tokens)
        slots[canonical] = torch.arange(tokens.numel(), device=tokens.device) % 6
        run_id = "vllm-native-top6-expert-ascending"
        plan = MoEOutputCombinePlan(
            6, values.shape[1], expert_weight_applied=True, output_dtype="float32"
        )
        inputs = MoEOutputCombineInputs(
            values,
            tokens,
            experts,
            slots,
            mapping_run_id=run_id,
            producer_backend="vllm.native.W4A16",
            producer_build_fingerprint="triton-kernels-3.8-unfused",
            num_tokens=num_tokens,
        )
        prepared = prepare_fixed_output_combine(
            plan, inputs, expected_mapping_run_id=run_id, num_experts=256
        )
        reduced = output_combine(prepared, values, mapping_run_id=run_id)
        self.calls += 1
        if num_tokens == self.ids.shape[0]:
            return target.copy_(reduced)
        return target.zero_().index_copy_(0, active_rows, reduced)


def bind_fixed_combine(ffn):
    """Bind the one supported native W4A16 ABI, with no numerical fallback."""
    backend = ffn.experts.routed_experts.quant_method.moe_kernel.fused_experts
    if getattr(backend, "_rlk_fixed_combine", False):
        return backend
    original = backend.apply
    function = original.__func__
    if (
        type(backend).__name__ != "UnfusedOAITritonExperts"
        or function.__globals__.get("triton_kernels_version") != "3.8"
        or "index_add_" not in function.__code__.co_names
    ):
        raise ValueError(
            "fixed combine requires pinned unfused triton-kernels 3.8 W4A16"
        )
    signature = inspect.signature(original)

    def apply(self, *args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        ids = bound.arguments["topk_ids"]
        if bound.arguments["expert_map"] is not None or self._lora_context is not None:
            raise ValueError("fixed native Top6 combine requires EP1 without LoRA")
        mode = _FixedReduce(ids)
        with mode:
            result = original(*args, **kwargs)
        if mode.calls != 1:
            raise RuntimeError(
                "native expert apply did not execute exactly one fixed combine"
            )
        return result

    backend.apply = types.MethodType(apply, backend)
    backend._rlk_fixed_combine = True
    print(
        "RLK_FIXED_ROLLOUT_COMBINE_BOUND",
        __file__,
        "FP32 expert ascending; original GEMMs",
        flush=True,
    )
    return backend
