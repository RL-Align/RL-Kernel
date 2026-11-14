# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 linear projection rows (RFC #420).

Four rows share one contract, each pinned to its checkpoint tensors:

=============================== ============ ============ ===== ================
row                             in -> out    dtype        bias  checkpoint tensors
=============================== ============ ============ ===== ================
``audio_input_projection_fp32``  32 -> 5376   float32      yes   ``audio_proj_in``
``audio_output_projection_fp32`` 5376 -> 32   float32      yes   ``audio_proj_out``
``text_context_projection``      5120 -> 5376 bfloat16     yes   ``context_embedder``
``h3_qkv_gemm`` (per projection) 5376 -> 7168 bfloat16     no    ``attn.to_{q,k,v}``
=============================== ============ ============ ===== ================

``audio_*`` modules are in ``_keep_in_fp32_modules``: weights, bias, input and output
are FP32 and a lower-precision argument is a contract violation (RFC probe H7), not a
variant. The text projection and Q/K/V are BF16 modules (FP32 accumulate, one rounding).

``h3_qkv_gemm`` also covers the fused layout used by serving engines (vLLM-Omni's
``QKVParallelLinear``): the rows of ``to_q``, ``to_k`` and ``to_v`` concatenated along
the output axis. ``forward_fused`` must equal the separate projections column for column.

Output width may be a shard of the full width so tensor-parallel layouts reuse the
rows: Q/K/V shards are whole heads (a multiple of 128 output columns).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

HEAD_DIM = 128


@dataclass(frozen=True)
class ProjectionSpec:
    row: str
    in_features: int
    out_features: int  # full width; shards must be whole multiples of ``shard_unit``
    dtype: torch.dtype
    has_bias: bool
    shard_unit: int  # smallest legal output width (columns)


AUDIO_INPUT = ProjectionSpec("audio_input_projection_fp32", 32, 5376, torch.float32, True, 64)
AUDIO_OUTPUT = ProjectionSpec("audio_output_projection_fp32", 5376, 32, torch.float32, True, 32)
TEXT_CONTEXT = ProjectionSpec("text_context_projection", 5120, 5376, torch.bfloat16, True, 64)
QKV = ProjectionSpec("h3_qkv_gemm", 5376, 7168, torch.bfloat16, False, HEAD_DIM)
SPECS = {spec.row: spec for spec in (AUDIO_INPUT, AUDIO_OUTPUT, TEXT_CONTEXT, QKV)}


def validate_h3_projection(
    spec: ProjectionSpec,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    allow_shard: bool = True,
) -> int:
    """Check ``x (T, K)``, ``weight (N, K)``, ``bias (N,)`` against ``spec``; return ``N``.

    Dtype is exact (no promotion), tensors share a device, ``T > 0``, ``K`` is the
    pinned input width, and ``N`` is the full width or (``allow_shard``) a whole-unit
    shard of it. A bias is required when the module has one and forbidden when it has none.
    """

    for name, tensor in (("x", x), ("weight", weight), ("bias", bias)):
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != spec.dtype:
            raise TypeError(
                f"{spec.row}: {name} must be {spec.dtype}, got {tensor.dtype} "
                "(the module's declared precision is part of the contract)"
            )
        if tensor.device != x.device:
            raise ValueError(f"{spec.row}: {name} is on {tensor.device}, x is on {x.device}")
    if x.dim() != 2 or x.shape[0] == 0:
        raise ValueError(f"{spec.row}: x must be a non-empty (T, {spec.in_features}) matrix")
    if x.shape[1] != spec.in_features:
        raise ValueError(f"{spec.row}: x width {x.shape[1]} != {spec.in_features}")
    if weight.dim() != 2 or weight.shape[1] != spec.in_features:
        raise ValueError(f"{spec.row}: weight must be (N, {spec.in_features})")
    n = weight.shape[0]
    legal = n == spec.out_features or (
        allow_shard and 0 < n < spec.out_features and n % spec.shard_unit == 0
    )
    if not legal:
        raise ValueError(
            f"{spec.row}: output width {n} must be {spec.out_features}"
            + (f" or a multiple of {spec.shard_unit} below it" if allow_shard else "")
        )
    if spec.has_bias:
        if bias is None:
            raise ValueError(f"{spec.row}: this module has a bias")
        if bias.shape != (n,):
            raise ValueError(f"{spec.row}: bias must be ({n},), got {tuple(bias.shape)}")
    elif bias is not None:
        raise ValueError(f"{spec.row}: this module has no bias")
    return n


def canonical_row_order(logical_keys: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
    """Row indices of the active rows in ascending logical-key order.

    ``None`` without keys, and also when the rows already are the active rows in canonical order
    (the reduction then runs in place and gives the same bytes).

    Active keys must be unique, otherwise the order of two equal keys would depend on the call.
    """

    if logical_keys is None:
        return None
    if not isinstance(logical_keys, torch.Tensor):
        raise TypeError("logical_keys must be a torch.Tensor")
    rows = x.shape[0]
    if logical_keys.dtype != torch.int64 or logical_keys.device != x.device:
        raise ValueError("logical_keys must be int64 on the projection device")
    if (
        logical_keys.ndim != 2
        or logical_keys.shape[0] != rows
        or logical_keys.shape[1] not in (2, 3)
    ):
        raise ValueError(f"logical_keys must have shape [{rows}, 2] or [{rows}, 3]")
    active = (logical_keys[:, 0] >= 0).nonzero().flatten()
    keys = logical_keys[active]
    if torch.unique(keys, dim=0).shape[0] != keys.shape[0]:
        raise ValueError("active logical keys must be unique")
    order = torch.arange(keys.shape[0], device=x.device)
    for column in reversed(range(keys.shape[1])):  # stable passes, least significant key first
        order = order[torch.argsort(keys[order, column], stable=True)]
    order = active[order]
    if order.numel() == rows and torch.equal(order, torch.arange(rows, device=x.device)):
        return None  # already in canonical order: reduce in place, no gather of x and g
    return order


def _linear(x, weight, bias, logical_keys):
    """``F.linear`` with the rows' ``logical_keys`` contract: padding rows give no parameter
    gradient. The keys are validated exactly as the CUDA rows validate them; the token-axis
    reduction itself stays the provider's (this is the reference, not a fixed-order kernel)."""

    canonical_row_order(logical_keys, x)  # validation only
    out = F.linear(x, weight, bias)
    if logical_keys is None or not bool((logical_keys[:, 0] < 0).any()):
        return out
    frozen = F.linear(x, weight.detach(), None if bias is None else bias.detach())
    return torch.where((logical_keys[:, 0] < 0)[:, None], frozen, out)


class _NativeProjection:
    """PyTorch reference: ``forward`` is the provider (``F.linear``); ``forward_fp32`` is golden.

    Same signature as the CUDA rows, including the optional ``logical_keys``.
    """

    spec: ProjectionSpec
    op_class = "reduction"

    def __call__(self, x, weight, bias=None, *, logical_keys=None):
        return self.forward(x, weight, bias, logical_keys=logical_keys)

    def forward(self, x, weight, bias=None, *, logical_keys=None) -> torch.Tensor:
        """Provider path (``nn.Linear``); run with TF32 disabled on CUDA."""

        validate_h3_projection(self.spec, x, weight, bias)
        return _linear(x, weight, bias, logical_keys)

    def forward_fp32(self, x, weight, bias=None, *, logical_keys=None) -> torch.Tensor:
        """Independent golden: the product in FP64, rounded once to the module dtype."""

        validate_h3_projection(self.spec, x, weight, bias)
        canonical_row_order(logical_keys, x)  # validation only
        out = F.linear(x.double(), weight.double(), None if bias is None else bias.double())
        return out.to(self.spec.dtype)


class NativeH3AudioInputProjectionOp(_NativeProjection):
    spec = AUDIO_INPUT


class NativeH3AudioOutputProjectionOp(_NativeProjection):
    spec = AUDIO_OUTPUT


class NativeH3TextContextProjectionOp(_NativeProjection):
    spec = TEXT_CONTEXT


class NativeH3QKVGemmOp:
    """Q, K and V projections of one attention block (no bias), separate or fused."""

    spec = QKV
    op_class = "reduction"

    def __call__(self, x, weight_q, weight_k, weight_v, *, logical_keys=None):
        return self.forward(x, weight_q, weight_k, weight_v, logical_keys=logical_keys)

    def forward(self, x, weight_q, weight_k, weight_v, *, logical_keys=None):
        """``(q, k, v)`` from three separate ``F.linear`` calls (the Diffusers layout)."""

        widths = {validate_h3_projection(QKV, x, w, None) for w in (weight_q, weight_k, weight_v)}
        if len(widths) != 1:
            raise ValueError(f"h3_qkv_gemm: Q/K/V widths differ: {sorted(widths)}")
        return tuple(_linear(x, w, None, logical_keys) for w in (weight_q, weight_k, weight_v))

    def project(self, x, weight, *, logical_keys=None) -> torch.Tensor:
        """One of the three projections alone (same as the CUDA row's ``project``)."""

        validate_h3_projection(QKV, x, weight, None)
        return _linear(x, weight, None, logical_keys)

    def forward_fused(self, x, weight_qkv, *, logical_keys=None):
        """One ``(T, 3 * N)`` product of the concatenated ``[Q; K; V]`` weight (serving layout)."""

        n = _fused_width(weight_qkv)
        validate_h3_projection(QKV, x, weight_qkv[:n], None)
        return _linear(x, weight_qkv, None, logical_keys)

    def forward_fp32(self, x, weight_q, weight_k, weight_v, *, logical_keys=None):
        """FP64 golden of each projection, rounded once to BF16."""

        canonical_row_order(logical_keys, x)  # validation only
        outputs = []
        for w in (weight_q, weight_k, weight_v):
            validate_h3_projection(QKV, x, w, None)
            outputs.append(F.linear(x.double(), w.double()).to(torch.bfloat16))
        return tuple(outputs)


def _fused_width(weight_qkv: torch.Tensor) -> int:
    if weight_qkv.dim() != 2 or weight_qkv.shape[0] % 3:
        raise ValueError("h3_qkv_gemm: fused weight must be (3 * N, K)")
    return weight_qkv.shape[0] // 3


def fuse_qkv_weights(weight_q, weight_k, weight_v) -> torch.Tensor:
    """Concatenate Q/K/V rows into the fused serving layout ``[Q; K; V]``."""

    return torch.cat([weight_q, weight_k, weight_v], dim=0)
