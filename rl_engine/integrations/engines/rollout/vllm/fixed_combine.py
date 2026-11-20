# SPDX-License-Identifier: Apache-2.0
"""Use the actor's expert-ascending FP32 combine at the pinned ROCm boundary.

The upstream 3.8 unfused W4A16 path uses an atomic index_add_. Its addition
order can change with batch shape and cross a BF16 rounding midpoint. Keep
upstream routing, packed GEMMs, activation and shared-expert handling; scope
only that one reduction to RLK's existing fixed Top6 operator.
"""

import ast
import inspect
import textwrap
import types
from contextvars import ContextVar

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
        return self.reduce(*args, **kwargs)

    def reduce(self, target, dim, tokens, values, *, alpha=1):
        if (
            self.calls
            or dim != 0
            or alpha != 1
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


def _scoped_reduction(function, current_mode):
    """Rewrite only acc.index_add_; preserve every original torch operation."""
    if function.__closure__:
        raise ValueError("unsupported native expert closure ABI")
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    # This is a scoped replacement for a freshly zeroed native accumulator,
    # not a general index_add_. Reject future upstream pre-seeding at bind time.
    for parent in ast.walk(tree):
        for _, statements in ast.iter_fields(parent):
            if not isinstance(statements, list):
                continue
            for i, statement in enumerate(statements):
                if not isinstance(statement, ast.Expr):
                    continue
                call = statement.value
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "index_add_"
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "acc"
                ):
                    continue
                previous = statements[i - 1] if i else None
                if not (
                    isinstance(previous, ast.Assign)
                    and len(previous.targets) == 1
                    and isinstance(previous.targets[0], ast.Name)
                    and previous.targets[0].id == "acc"
                    and isinstance(previous.value, ast.Call)
                    and isinstance(previous.value.func, ast.Attribute)
                    and isinstance(previous.value.func.value, ast.Name)
                    and previous.value.func.value.id == "torch"
                    and previous.value.func.attr == "zeros"
                ):
                    raise ValueError(
                        "fixed combine requires a freshly zeroed native accumulator"
                    )

    class Boundary(ast.NodeTransformer):
        calls = 0

        def visit_Call(self, node):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "index_add_"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "acc"
            ):
                self.calls += 1
                node.args.insert(0, node.func.value)
                node.func = ast.Name("_rlk_fixed_index_add", ast.Load())
            return self.generic_visit(node)

    transform = Boundary()
    tree = transform.visit(tree)
    if transform.calls != 1:
        raise ValueError("unsupported native accumulator reduction boundary")
    tree.body[0].decorator_list = []

    def reduce(target, *args, **kwargs):
        return current_mode.get().reduce(target, *args, **kwargs)

    namespace = dict(function.__globals__, _rlk_fixed_index_add=reduce)
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    return namespace[function.__name__]


def bind_fixed_combine(ffn):
    """Bind the one supported native W4A16 ABI, with no numerical fallback."""
    experts = ffn.experts.routed_experts
    if experts.w13_weight.shape[0] != 256 or experts.w2_weight.shape[0] != 256:
        raise ValueError("fixed native combine supports exactly 256 local experts")
    backend = experts.quant_method.moe_kernel.fused_experts
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
    current_mode = ContextVar("rlk_fixed_combine_mode")
    scoped = _scoped_reduction(function, current_mode)

    def apply(self, *args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        ids = bound.arguments["topk_ids"]
        if bound.arguments["expert_map"] is not None or self._lora_context is not None:
            raise ValueError("fixed native Top6 combine requires EP1 without LoRA")
        mode = _FixedReduce(ids)
        token = current_mode.set(mode)
        try:
            result = scoped(self, *args, **kwargs)
        finally:
            current_mode.reset(token)
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
        "FP32 expert ascending; original GEMMs; scope=acc.index_add_; experts=256",
        flush=True,
    )
    return backend
