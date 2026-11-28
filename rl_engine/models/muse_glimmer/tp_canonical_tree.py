# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Muse Glimmer canonical TP tree (RFC #501, WS2 ``tp_canonical_tree``).

Every TP reduction in a Muse Glimmer decoder layer is evaluated on one canonical tree whose
leaves are the ``CANONICAL_TP = 8`` shards of the reduced dimension:

* row-parallel forward sums -- ``self_attn.o_proj`` over 4,096 (8 shards of 512, 4 heads
  each) and ``mlp.down_proj`` over 19,968 (8 shards of 2,496);
* column-parallel input gradients -- ``self_attn.q_proj`` / ``self_attn.gate_proj`` and
  ``mlp.gate_proj`` / ``mlp.up_proj`` reduce over their output dimension.

Each leaf is one GEMM with FP32 output. Leaves are combined in FP32 as the balanced tree
``((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))`` and rounded to the activation dtype
exactly once. A physical TP rank ``r`` owns the contiguous leaves
``[r * 8 / tp, (r + 1) * 8 / tp)``; it folds them locally into the root of its subtree and the
deterministic collective folds the rank roots in the same balanced order. TP1, TP2, TP4 and
TP8 therefore evaluate the same floating-point expression and agree byte for byte.

Every leaf GEMM has the same shape at every physical TP, so cross-TP equality only needs a
run-to-run deterministic GEMM. The default leaf GEMM is the FP32 upcast reference; the
qualified [G] kernel (``dense_gemm_qualification``) plugs in through ``gemm=``.

K/V input gradients and the KV ownership map belong to ``kv_head_replication`` and are not
covered here. Sequence-parallel reduce-scatter is not supported yet and fails closed.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from rl_engine.models.muse_glimmer.workload import (
    CANONICAL_TP,
    OFFICIAL_FINGERPRINT,
    OFFICIAL_PARALLEL_LAYOUT,
    OFFICIAL_PROJECTION_GROUPING,
)

CANONICAL_TP_ENV = "RL_KERNEL_STRICT_CANONICAL_TP"
SUPPORTED_PHYSICAL_TP = (1, 2, 4, 8)
PARTIAL_DTYPE = torch.float32

# Collectives whose reduction order is the fixed balanced rank tree. Anything else, including
# a native NCCL/RCCL all_reduce, is a reducing collective with an unspecified tree.
FIXED_TREE_COLLECTIVES = frozenset(
    {"cuda_ipc_fixed_tree", "rocm_ipc_fixed_tree", "torch_distributed_balanced_tree"}
)

ROW_PARALLEL = "row"
COLUMN_PARALLEL = "column"


class CanonicalTreeError(ValueError):
    """Raised when an input would leave the canonical TP tree (strict mode fails closed)."""


@dataclass(frozen=True)
class ProjectionSpec:
    """One TP-sharded projection: weight shape ``[out_features, in_features]``."""

    name: str
    kind: str
    in_features: int
    out_features: int

    @property
    def reduced_features(self) -> int:
        """Size of the dimension the canonical tree reduces over."""

        return self.in_features if self.kind == ROW_PARALLEL else self.out_features

    @property
    def canonical_width(self) -> int:
        return self.reduced_features // CANONICAL_TP

    def validate(self) -> None:
        if self.kind not in (ROW_PARALLEL, COLUMN_PARALLEL):
            raise CanonicalTreeError(f"{self.name}: unknown parallel kind {self.kind!r}")
        if self.in_features <= 0 or self.out_features <= 0:
            raise CanonicalTreeError(f"{self.name}: feature sizes must be positive")
        if self.reduced_features % CANONICAL_TP:
            raise CanonicalTreeError(
                f"{self.name}: reduced dimension {self.reduced_features} does not divide into "
                f"{CANONICAL_TP} canonical shards"
            )


def _official_projections() -> dict[str, ProjectionSpec]:
    h = int(OFFICIAL_FINGERPRINT["hidden_size"])
    q = int(OFFICIAL_FINGERPRINT["num_attention_heads"]) * int(OFFICIAL_FINGERPRINT["head_dim"])
    f = int(OFFICIAL_FINGERPRINT["intermediate_size"])
    specs = {
        "self_attn.o_proj": ProjectionSpec("self_attn.o_proj", ROW_PARALLEL, q, h),
        "mlp.down_proj": ProjectionSpec("mlp.down_proj", ROW_PARALLEL, f, h),
        "self_attn.q_proj": ProjectionSpec("self_attn.q_proj", COLUMN_PARALLEL, h, q),
        "self_attn.gate_proj": ProjectionSpec("self_attn.gate_proj", COLUMN_PARALLEL, h, q),
        "mlp.gate_proj": ProjectionSpec("mlp.gate_proj", COLUMN_PARALLEL, h, f),
        "mlp.up_proj": ProjectionSpec("mlp.up_proj", COLUMN_PARALLEL, h, f),
    }
    shards = OFFICIAL_PARALLEL_LAYOUT["row_parallel_shards"]
    for short, width in shards.items():
        spec = next(s for s in specs.values() if s.name.endswith("." + short))
        if spec.canonical_width != width:
            raise CanonicalTreeError(
                f"{spec.name}: canonical shard {spec.canonical_width} disagrees with the "
                f"fingerprint's row_parallel_shards[{short!r}] = {width}"
            )
    return specs


MUSE_PROJECTIONS: dict[str, ProjectionSpec] = _official_projections()

# Column-parallel projections that read the same input; their input gradients are summed in
# this declared order, in FP32, after each lane has been reduced on its own canonical tree.
# K/V join the attention group with ``kv_head_replication``.
MUSE_COLUMN_GROUPS: dict[str, tuple[str, ...]] = {
    "attention": ("self_attn.q_proj", "self_attn.gate_proj"),
    "mlp": ("mlp.gate_proj", "mlp.up_proj"),
}


def muse_projection(name: str) -> ProjectionSpec:
    """Return the frozen spec for ``name``; fused or unknown projections fail closed."""

    fused = OFFICIAL_PROJECTION_GROUPING["fail_closed"]
    short = name.rsplit(".", 1)[-1]
    if name in fused or short in fused:
        raise CanonicalTreeError(
            f"{name} is a fused projection; strict mode keeps the frozen grouping "
            f"{OFFICIAL_PROJECTION_GROUPING['attention']} / {OFFICIAL_PROJECTION_GROUPING['mlp']}"
        )
    try:
        return MUSE_PROJECTIONS[name]
    except KeyError:
        raise CanonicalTreeError(
            f"{name} is not on the Muse Glimmer canonical TP tree; known: "
            f"{sorted(MUSE_PROJECTIONS)}"
        ) from None


# --------------------------------------------------------------------------------------------
# Topology and collective checks
# --------------------------------------------------------------------------------------------


def require_canonical_tp() -> int:
    """Check ``RL_KERNEL_STRICT_CANONICAL_TP``; only the frozen value 8 is accepted."""

    raw = os.getenv(CANONICAL_TP_ENV, "").strip()
    if raw and raw != str(CANONICAL_TP):
        raise CanonicalTreeError(
            f"{CANONICAL_TP_ENV}={raw}; Muse Glimmer freezes canonical TP = {CANONICAL_TP}"
        )
    return CANONICAL_TP


def leaves_per_rank(tp_world: int) -> int:
    require_canonical_tp()
    if tp_world not in SUPPORTED_PHYSICAL_TP:
        raise CanonicalTreeError(f"physical TP {tp_world} is not one of {SUPPORTED_PHYSICAL_TP}")
    return CANONICAL_TP // tp_world


def owned_leaves(tp_rank: int, tp_world: int) -> list[int]:
    per_rank = leaves_per_rank(tp_world)
    if not 0 <= tp_rank < tp_world:
        raise CanonicalTreeError(f"TP rank {tp_rank} is outside TP world {tp_world}")
    return list(range(tp_rank * per_rank, (tp_rank + 1) * per_rank))


def require_fixed_tree_collective(collective: Any) -> int:
    """Return the physical TP size of ``collective`` (``None`` means TP1)."""

    if collective is None:
        return 1
    backend = getattr(collective, "backend_id", None)
    if backend not in FIXED_TREE_COLLECTIVES:
        raise CanonicalTreeError(
            f"collective {type(collective).__name__} (backend_id={backend!r}) is not a "
            f"fixed-tree collective; native reducing collectives are not allowed in strict "
            f"mode. Allowed: {sorted(FIXED_TREE_COLLECTIVES)}"
        )
    world = int(collective.world_size)
    leaves_per_rank(world)
    return world


def _tp_rank(collective: Any) -> int:
    return 0 if collective is None else int(collective.rank)


# --------------------------------------------------------------------------------------------
# Leaf GEMM and the FP32 tree
# --------------------------------------------------------------------------------------------

PartialGemm = Callable[[Tensor, Tensor], Tensor]
"""``gemm(a, b)`` returns ``a @ b.T`` in FP32 for ``a: [M, K]`` and ``b: [N, K]``."""


def fp32_reference_gemm(a: Tensor, b: Tensor) -> Tensor:
    """Upcast leaf GEMM: exact BF16 -> FP32 inputs, FP32 accumulation, no TF32."""

    if a.is_cuda and (
        torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest"
    ):
        raise CanonicalTreeError("TF32 is enabled; strict leaves need full FP32 accumulation")
    return torch.matmul(a.to(PARTIAL_DTYPE), b.to(PARTIAL_DTYPE).t())


fp32_reference_gemm.gemm_id = "fp32_upcast_reference"  # type: ignore[attr-defined]


def gemm_id(gemm: PartialGemm) -> str:
    return str(getattr(gemm, "gemm_id", getattr(gemm, "__qualname__", repr(gemm))))


def _leaf(gemm: PartialGemm, a: Tensor, b: Tensor) -> Tensor:
    out = gemm(a.contiguous(), b.contiguous())
    if out.dtype != PARTIAL_DTYPE:
        raise CanonicalTreeError(
            f"leaf GEMM {gemm_id(gemm)} returned {out.dtype}; partials must be FP32 so the "
            "tree rounds once"
        )
    return out


def balanced_tree_sum(partials: Sequence[Tensor]) -> Tensor:
    """``((p0 + p1) + (p2 + p3)) + ...`` over a power-of-two number of partials."""

    values = list(partials)
    if not values or len(values) & (len(values) - 1):
        raise CanonicalTreeError(f"a balanced tree needs 2^k partials, got {len(values)}")
    while len(values) > 1:
        values = [values[i] + values[i + 1] for i in range(0, len(values), 2)]
    return values[0]


def _reduce_across_ranks(local_root: Tensor, collective: Any) -> Tensor:
    if collective is None:
        return local_root
    return collective.all_reduce(local_root.contiguous())


def _check_tokens(name: str, value: Tensor) -> None:
    if value.dim() < 1 or value.numel() == 0:
        raise CanonicalTreeError(f"{name} must hold at least one token")
    if not value.is_floating_point():
        raise CanonicalTreeError(f"{name} must be floating point, got {value.dtype}")


# --------------------------------------------------------------------------------------------
# Row-parallel projection (o_proj, down_proj)
# --------------------------------------------------------------------------------------------


def _check_row_shard(spec: ProjectionSpec, x_shard: Tensor, w_shard: Tensor, tp: int) -> int:
    spec.validate()
    if spec.kind != ROW_PARALLEL:
        raise CanonicalTreeError(f"{spec.name} is not row-parallel")
    _check_tokens("input shard", x_shard)
    local_in = spec.in_features // tp
    if x_shard.size(-1) != local_in:
        raise CanonicalTreeError(
            f"{spec.name}: input shard width {x_shard.size(-1)} != {spec.in_features} / TP{tp}"
        )
    if tuple(w_shard.shape) != (spec.out_features, local_in):
        raise CanonicalTreeError(
            f"{spec.name}: weight shard {tuple(w_shard.shape)} != "
            f"{(spec.out_features, local_in)} at TP{tp}"
        )
    if w_shard.dtype != x_shard.dtype:
        raise CanonicalTreeError(f"{spec.name}: weight and input dtypes differ")
    return spec.canonical_width


def row_parallel_forward(
    x_shard: Tensor,
    w_shard: Tensor,
    spec: ProjectionSpec,
    *,
    collective: Any = None,
    gemm: PartialGemm = fp32_reference_gemm,
) -> Tensor:
    """``x @ W.T`` for a row-parallel projection, summed on the canonical tree.

    ``x_shard`` is this rank's ``[..., in_features / tp]`` slice and ``w_shard`` the matching
    ``[out_features, in_features / tp]`` weight slice. Returns the full replicated output in
    the input dtype, rounded once from FP32.
    """

    tp = require_fixed_tree_collective(collective)
    width = _check_row_shard(spec, x_shard, w_shard, tp)
    x2d = x_shard.reshape(-1, x_shard.size(-1))
    partials = [
        _leaf(gemm, x2d.narrow(1, i * width, width), w_shard.narrow(1, i * width, width))
        for i in range(leaves_per_rank(tp))
    ]
    total = _reduce_across_ranks(balanced_tree_sum(partials), collective)
    return total.to(x_shard.dtype).reshape(*x_shard.shape[:-1], spec.out_features)


def row_parallel_backward(
    grad_out: Tensor,
    x_shard: Tensor,
    w_shard: Tensor,
    spec: ProjectionSpec,
    *,
    tp: int,
    gemm: PartialGemm = fp32_reference_gemm,
) -> tuple[Tensor, Tensor]:
    """Input and weight gradients of a row-parallel shard; neither reduces across TP.

    Both are computed per canonical leaf so a leaf's GEMM shape does not depend on TP.
    """

    width = _check_row_shard(spec, x_shard, w_shard, tp)
    g2d = grad_out.reshape(-1, spec.out_features)
    x2d = x_shard.reshape(-1, x_shard.size(-1))
    g_t = g2d.t()
    grad_x, grad_w = [], []
    for i in range(leaves_per_rank(tp)):
        w_leaf = w_shard.narrow(1, i * width, width)
        x_leaf = x2d.narrow(1, i * width, width)
        grad_x.append(_leaf(gemm, g2d, w_leaf.t()))  # [T, width], reduces over out_features
        grad_w.append(_leaf(gemm, g_t, x_leaf.t()))  # [out, width], reduces over tokens
    grad_x_shard = torch.cat(grad_x, dim=1).to(x_shard.dtype).reshape(x_shard.shape)
    grad_w_shard = torch.cat(grad_w, dim=1).to(w_shard.dtype)
    return grad_x_shard, grad_w_shard


class _RowParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_shard, w_shard, spec, collective, gemm):
        out = row_parallel_forward(x_shard, w_shard, spec, collective=collective, gemm=gemm)
        ctx.save_for_backward(x_shard, w_shard)
        ctx.spec, ctx.tp, ctx.gemm = spec, require_fixed_tree_collective(collective), gemm
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x_shard, w_shard = ctx.saved_tensors
        grad_x, grad_w = row_parallel_backward(
            grad_out, x_shard, w_shard, ctx.spec, tp=ctx.tp, gemm=ctx.gemm
        )
        return grad_x, grad_w, None, None, None


def canonical_row_parallel_linear(
    x_shard: Tensor,
    w_shard: Tensor,
    spec: ProjectionSpec | str,
    *,
    collective: Any = None,
    gemm: PartialGemm = fp32_reference_gemm,
) -> Tensor:
    """Differentiable row-parallel projection on the canonical tree."""

    if isinstance(spec, str):
        spec = muse_projection(spec)
    return _RowParallelLinear.apply(x_shard, w_shard, spec, collective, gemm)


# --------------------------------------------------------------------------------------------
# Column-parallel group (q_proj + attention gate, MLP gate + up)
# --------------------------------------------------------------------------------------------


def _check_column_shard(spec: ProjectionSpec, w_shard: Tensor, x: Tensor, tp: int) -> int:
    spec.validate()
    if spec.kind != COLUMN_PARALLEL:
        raise CanonicalTreeError(f"{spec.name} is not column-parallel")
    local_out = spec.out_features // tp
    if tuple(w_shard.shape) != (local_out, spec.in_features):
        raise CanonicalTreeError(
            f"{spec.name}: weight shard {tuple(w_shard.shape)} != "
            f"{(local_out, spec.in_features)} at TP{tp}"
        )
    if x.size(-1) != spec.in_features:
        raise CanonicalTreeError(f"{spec.name}: input width {x.size(-1)} != {spec.in_features}")
    if w_shard.dtype != x.dtype:
        raise CanonicalTreeError(f"{spec.name}: weight and input dtypes differ")
    return spec.canonical_width


def _check_group(specs: Sequence[ProjectionSpec], weights: Sequence[Tensor]) -> None:
    if not specs:
        raise CanonicalTreeError("a column group needs at least one projection")
    if len(specs) != len(weights):
        raise CanonicalTreeError("one weight shard per projection is required")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise CanonicalTreeError(f"duplicate projections in a column group: {names}")
    if len({s.in_features for s in specs}) != 1:
        raise CanonicalTreeError("projections in a column group must read the same input")


def column_parallel_forward(
    x: Tensor,
    w_shards: Sequence[Tensor],
    specs: Sequence[ProjectionSpec],
    *,
    tp: int,
    gemm: PartialGemm = fp32_reference_gemm,
) -> tuple[Tensor, ...]:
    """Local output shards; each canonical leaf of output columns is its own GEMM."""

    _check_tokens("input", x)
    _check_group(specs, w_shards)
    x2d = x.reshape(-1, x.size(-1))
    outputs = []
    for spec, w_shard in zip(specs, w_shards, strict=True):
        width = _check_column_shard(spec, w_shard, x, tp)
        leaves = [
            _leaf(gemm, x2d, w_shard.narrow(0, i * width, width))
            for i in range(leaves_per_rank(tp))
        ]
        out = torch.cat(leaves, dim=1).to(x.dtype)
        outputs.append(out.reshape(*x.shape[:-1], w_shard.size(0)))
    return tuple(outputs)


def column_parallel_input_grad(
    grad_outs: Sequence[Tensor],
    w_shards: Sequence[Tensor],
    specs: Sequence[ProjectionSpec],
    *,
    collective: Any = None,
    gemm: PartialGemm = fp32_reference_gemm,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    """Input gradient of a column group: every lane on its own canonical tree, then the lanes
    summed in declared order in FP32, then one rounding.

    The lanes are packed into one collective call; an element-wise fixed-tree all_reduce keeps
    each lane's tree separate.
    """

    tp = require_fixed_tree_collective(collective)
    _check_group(specs, w_shards)
    if len(grad_outs) != len(specs):
        raise CanonicalTreeError("one output gradient per projection is required")
    roots = []
    for spec, g, w_shard in zip(specs, grad_outs, w_shards, strict=True):
        _check_tokens(f"{spec.name} gradient", g)
        width = spec.canonical_width
        local_out = spec.out_features // tp
        if tuple(w_shard.shape) != (local_out, spec.in_features) or g.size(-1) != local_out:
            raise CanonicalTreeError(f"{spec.name}: gradient or weight shard is not TP{tp}-local")
        g2d = g.reshape(-1, local_out)
        partials = [
            _leaf(gemm, g2d.narrow(1, i * width, width), w_shard.narrow(0, i * width, width).t())
            for i in range(leaves_per_rank(tp))
        ]
        roots.append(balanced_tree_sum(partials))
    if len({tuple(r.shape) for r in roots}) != 1:
        raise CanonicalTreeError("column group lanes cover different token counts")
    reduced = _reduce_across_ranks(torch.stack(roots), collective)
    total = reduced[0]
    for lane in range(1, reduced.size(0)):
        total = total + reduced[lane]
    dtype = out_dtype if out_dtype is not None else grad_outs[0].dtype
    return total.to(dtype).reshape(*grad_outs[0].shape[:-1], specs[0].in_features)


def column_parallel_weight_grads(
    grad_outs: Sequence[Tensor],
    x: Tensor,
    w_shards: Sequence[Tensor],
    specs: Sequence[ProjectionSpec],
    *,
    tp: int,
    gemm: PartialGemm = fp32_reference_gemm,
) -> tuple[Tensor, ...]:
    """Weight-gradient shards; each reduces over tokens only, per canonical leaf of rows."""

    x2d_t = x.reshape(-1, x.size(-1)).t()
    grads = []
    for spec, g, w_shard in zip(specs, grad_outs, w_shards, strict=True):
        width = _check_column_shard(spec, w_shard, x, tp)
        g2d = g.reshape(-1, w_shard.size(0))
        leaves = [
            _leaf(gemm, g2d.narrow(1, i * width, width).t(), x2d_t)
            for i in range(leaves_per_rank(tp))
        ]
        grads.append(torch.cat(leaves, dim=0).to(w_shard.dtype))
    return tuple(grads)


class _ColumnParallelGroup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, specs, collective, gemm, *w_shards):
        tp = require_fixed_tree_collective(collective)
        outputs = column_parallel_forward(x, w_shards, specs, tp=tp, gemm=gemm)
        ctx.save_for_backward(x, *w_shards)
        ctx.specs, ctx.collective, ctx.gemm, ctx.tp = specs, collective, gemm, tp
        return outputs

    @staticmethod
    def backward(ctx, *grad_outs):
        x, *w_shards = ctx.saved_tensors
        grads = [
            g
            if g is not None
            else torch.zeros(*x.shape[:-1], w.size(0), dtype=x.dtype, device=x.device)
            for g, w in zip(grad_outs, w_shards, strict=True)
        ]
        grad_x = column_parallel_input_grad(
            grads,
            w_shards,
            ctx.specs,
            collective=ctx.collective,
            gemm=ctx.gemm,
            out_dtype=x.dtype,
        )
        grad_ws = column_parallel_weight_grads(
            grads, x, w_shards, ctx.specs, tp=ctx.tp, gemm=ctx.gemm
        )
        return (grad_x, None, None, None, *grad_ws)


def canonical_column_parallel_group(
    x: Tensor,
    w_shards: Sequence[Tensor],
    specs: Sequence[ProjectionSpec | str],
    *,
    collective: Any = None,
    gemm: PartialGemm = fp32_reference_gemm,
) -> tuple[Tensor, ...]:
    """Differentiable column-parallel projections that share one input.

    Returns this rank's output shard per projection. Backward produces the input gradient on
    the canonical tree with every lane reduced separately and one final rounding.
    """

    resolved = tuple(muse_projection(s) if isinstance(s, str) else s for s in specs)
    return _ColumnParallelGroup.apply(x, resolved, collective, gemm, *w_shards)


# --------------------------------------------------------------------------------------------
# Sharding helpers and provenance
# --------------------------------------------------------------------------------------------


def shard_weight(weight: Tensor, spec: ProjectionSpec, tp_rank: int, tp_world: int) -> Tensor:
    """Megatron-style contiguous shard of a full weight for ``tp_rank``."""

    leaves_per_rank(tp_world)
    if spec.kind == ROW_PARALLEL:
        width = spec.in_features // tp_world
        return weight.narrow(1, tp_rank * width, width).contiguous()
    width = spec.out_features // tp_world
    return weight.narrow(0, tp_rank * width, width).contiguous()


def shard_activation(x: Tensor, spec: ProjectionSpec, tp_rank: int, tp_world: int) -> Tensor:
    """Row-parallel input slice (the column-parallel output layout) for ``tp_rank``."""

    width = spec.reduced_features // tp_world
    return x.narrow(-1, tp_rank * width, width).contiguous()


def tp_canonical_provenance(
    collective: Any = None,
    *,
    specs: Sequence[ProjectionSpec] | None = None,
    gemm: PartialGemm = fp32_reference_gemm,
) -> dict[str, Any]:
    """What a strict run records about its TP tree."""

    tp = require_fixed_tree_collective(collective)
    rank = _tp_rank(collective)
    specs = list(MUSE_PROJECTIONS.values()) if specs is None else list(specs)
    return {
        "row": "tp_canonical_tree",
        "physical_tp": tp,
        "canonical_tp": CANONICAL_TP,
        "tp_rank": rank,
        "owned_canonical_leaves": owned_leaves(rank, tp),
        "collective_backend": "none" if collective is None else collective.backend_id,
        "reduction_order": "balanced_tree_over_canonical_leaves",
        "partial_dtype": str(PARTIAL_DTYPE).replace("torch.", ""),
        "rounding": "once_after_tree",
        "leaf_gemm": gemm_id(gemm),
        "column_groups": {k: list(v) for k, v in MUSE_COLUMN_GROUPS.items()},
        "canonical_shards": {s.name: [s.kind, s.canonical_width] for s in specs},
    }


__all__ = [
    "CANONICAL_TP_ENV",
    "COLUMN_PARALLEL",
    "CanonicalTreeError",
    "FIXED_TREE_COLLECTIVES",
    "MUSE_COLUMN_GROUPS",
    "MUSE_PROJECTIONS",
    "ProjectionSpec",
    "ROW_PARALLEL",
    "balanced_tree_sum",
    "canonical_column_parallel_group",
    "canonical_row_parallel_linear",
    "column_parallel_forward",
    "column_parallel_input_grad",
    "column_parallel_weight_grads",
    "fp32_reference_gemm",
    "leaves_per_rank",
    "muse_projection",
    "owned_leaves",
    "require_canonical_tp",
    "require_fixed_tree_collective",
    "row_parallel_backward",
    "row_parallel_forward",
    "shard_activation",
    "shard_weight",
    "tp_canonical_provenance",
]
