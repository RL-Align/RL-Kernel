# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Hardware-independent entry for the Nemotron grouped expert MLP.

The `moe_experts_combine` row of RFC #434: grouped up-projection GEMM with a
relu-square epilogue, grouped down-projection, a fixed-order weighted combine
over the six route slots and the optional shared expert, with one exit cast.

This module owns the only copy of the argument validation and selects a
provider through the kernel registry, so the CUDA and ROCm columns implement one
versioned contract.  The hot path never synchronizes: `permutation` and
`offsets` values are the router's responsibility (a bijection and exclusive
prefix sums), and only shape/dtype/device metadata is checked here.
"""

from __future__ import annotations

import threading

import torch

from rl_engine.runtime.policy import rl_kernel_mode, route_report_enabled
from rl_engine.runtime.registry import kernel_registry

CONTRACT_VERSION = "nemotron-experts-sm90-v1"
OP_TYPE = "moe_experts_combine"
# Bound every 32-bit index product: six packed rows per token, 2688 columns.
MAX_TOKENS = 65536
# Wide enough for the vectorized stages and for the 64-wide GEMM tiles.
_WIDTH_ALIGNMENT = 64
_VECTOR_ALIGNMENT = 8

_PROVIDER_REPORTED = False
_PROVIDER_REPORT_LOCK = threading.Lock()


def _require(name, tensor, dtype, *, dims=None, device=None, contiguous=True):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {str(dtype).replace('torch.', '')}")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must live on the payload device")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if dims is not None and tensor.dim() != dims:
        raise ValueError(f"{name} must be rank {dims}")
    return tensor


def _validate(
    packed,
    x,
    weights,
    permutation,
    offsets,
    w_up,
    w_down,
    w_sh1,
    w_sh2,
):
    _require("packed", packed, torch.bfloat16, dims=2)
    device = packed.device
    tokens, width = packed.shape
    if tokens % 6:
        raise ValueError("packed must hold six rows per token")
    tokens //= 6
    if tokens > MAX_TOKENS:
        raise ValueError(f"at most {MAX_TOKENS} tokens supported by this operator")
    if width % _WIDTH_ALIGNMENT or width % _VECTOR_ALIGNMENT:
        raise ValueError(f"hidden width must be {_WIDTH_ALIGNMENT} aligned")
    _require("weights", weights, torch.float32, dims=2, device=device)
    if weights.shape != (tokens, 6):
        raise ValueError("weights must be [tokens, 6] float32")
    _require("permutation", permutation, torch.int64, dims=1, device=device)
    if permutation.numel() != tokens * 6:
        raise ValueError("permutation must cover every packed row")
    _require("offsets", offsets, torch.int64, dims=1, device=device)
    _require("w_up", w_up, torch.bfloat16, dims=3, device=device)
    _require("w_down", w_down, torch.bfloat16, dims=3, device=device)
    experts, intermediate = w_up.shape[0], w_up.shape[1]
    if experts < 1:
        raise ValueError("at least one expert is required")
    if w_up.shape[2] != width:
        raise ValueError("w_up hidden width must match packed")
    if w_down.shape != (experts, width, intermediate):
        raise ValueError("w_down must be [experts, hidden, intermediate]")
    if intermediate % _WIDTH_ALIGNMENT:
        raise ValueError(f"intermediate width must be {_WIDTH_ALIGNMENT} aligned")
    if offsets.numel() != experts + 1:
        raise ValueError("offsets must hold experts+1 exclusive prefix sums")
    if (w_sh1 is None) != (w_sh2 is None):
        raise ValueError("shared expert weights must be supplied as a pair")
    if w_sh1 is not None:
        _require("w_sh1", w_sh1, torch.bfloat16, dims=2, device=device)
        _require("w_sh2", w_sh2, torch.bfloat16, dims=2, device=device)
        if x is None:
            raise ValueError("the shared expert requires the token payload x")
        if w_sh1.shape[1] != width or w_sh2.shape[0] != width:
            raise ValueError("shared expert hidden widths must agree")
        if w_sh1.shape[0] != w_sh2.shape[1]:
            raise ValueError("shared expert intermediate widths must agree")
        if w_sh1.shape[0] % _WIDTH_ALIGNMENT:
            raise ValueError(f"shared intermediate width must be {_WIDTH_ALIGNMENT} aligned")
    if x is not None:
        _require("x", x, torch.bfloat16, dims=2, device=device)
        if tuple(x.shape) != (tokens, width):
            raise ValueError("x must be [tokens, hidden]")


def _report_provider_once(op) -> None:
    global _PROVIDER_REPORTED
    if torch._dynamo.is_compiling():
        return
    if not route_report_enabled():
        return
    with _PROVIDER_REPORT_LOCK:
        if _PROVIDER_REPORTED:
            return
        _PROVIDER_REPORTED = True
    print(
        f"[RL-Kernel][route] mode={rl_kernel_mode().value} module=moe_experts "
        f"provider={nemotron_experts_provider_id(op)}",
        flush=True,
    )


def nemotron_experts_provider_id(op=None) -> str:
    """Return the provider id that would run `moe_experts_combine` on this device.

    Pass an already selected operator to avoid a second registry lookup.
    """
    if op is None:
        op = kernel_registry.get_op(OP_TYPE)
    return str(getattr(op, "provider_id", type(op).__name__))


def nemotron_experts(
    packed: torch.Tensor,
    x: torch.Tensor | None,
    weights: torch.Tensor,
    permutation: torch.Tensor,
    offsets: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    w_sh1: torch.Tensor | None = None,
    w_sh2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the grouped expert MLP, combine and shared expert add.

    `packed` is the router payload `[6*tokens, hidden]` and stays differentiable:
    its gradient is the route-weighted, down-projected expert gradient.  `x` is
    the original token payload used only by the shared expert, so it may be None
    when `w_sh1`/`w_sh2` are omitted.  Every input must be contiguous, on one
    CUDA device, and `permutation`/`offsets` must be the router's bijection and
    exclusive prefix sums; their values are not re-validated on the hot path.
    """
    _validate(packed, x, weights, permutation, offsets, w_up, w_down, w_sh1, w_sh2)
    op = kernel_registry.get_op(OP_TYPE, device=packed.device)
    _report_provider_once(op)
    return op(packed, x, weights, permutation, offsets, w_up, w_down, w_sh1, w_sh2)
