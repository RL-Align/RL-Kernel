# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C4 (#270): enumerable gradient adapters and status matrix.

Name-only mention in a chain report does not count. Each required
differentiable op has a registered adapter with stable logical grad names.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from rl_engine.kernels.gtest.forward_invariance import ConfigSpec, RuntimeObservation
from rl_engine.kernels.gtest.gradient_invariance import (
    GradientObservation,
    GradientTensorSpec,
    MissingBackwardError,
)
from rl_engine.kernels.gtest.operator_specs import OP_SPECS, _load_object
from rl_engine.kernels.gtest.tolerance import normalize_dtype_name
from rl_engine.kernels.ops.pytorch.mhc import MHC_PRE_HIDDEN_SIZE
from rl_engine.testing.ws1_workload import (
    PaddedBatch,
    PhysicalLayout,
    WS1Manifest,
    load_manifest,
    profile_required_nodes,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

AdapterRequirement = Literal[
    "required", "optional_fused", "layout_supported", "absent_not_required"
]


@dataclass(frozen=True)
class GradientAdapterSpec:
    """One enumerable differentiable WS1 operator adapter."""

    op_name: str
    chain_node: str
    op_class: str
    spec_name: str | None
    tensors: tuple[GradientTensorSpec, ...]
    requirement: AdapterRequirement
    source_files: tuple[str, ...]
    shape_dependent_bwd_accum: str = "forbidden"
    atomic_add: str = "forbidden"


@dataclass(frozen=True)
class AdapterStatusRow:
    """One cell of the C4 adapter status matrix."""

    op_name: str
    chain_node: str
    backend_profile: str
    requirement: AdapterRequirement
    candidate_status: str
    adapter_registered: bool
    expected_backend_id: str | None
    candidate_path: str | None
    tracked_red: bool
    untracked_red: bool
    grad_tensor_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "chain_node": self.chain_node,
            "backend_profile": self.backend_profile,
            "requirement": self.requirement,
            "candidate_status": self.candidate_status,
            "adapter_registered": self.adapter_registered,
            "expected_backend_id": self.expected_backend_id,
            "candidate_path": self.candidate_path,
            "tracked_red": self.tracked_red,
            "untracked_red": self.untracked_red,
            "grad_tensor_names": list(self.grad_tensor_names),
        }


_DX = GradientTensorSpec("dx", "token", "x")
_DWEIGHT = GradientTensorSpec("dweight", "parameter", "weight")
_DX_GEMM = GradientTensorSpec("dX", "token", "a")
_DW_GEMM = GradientTensorSpec("dW", "parameter", "b")
_DQ = GradientTensorSpec("dQ", "token", "q")
_DK = GradientTensorSpec("dK", "token", "k")
_DV = GradientTensorSpec("dV", "token", "v")
_DHIDDEN = GradientTensorSpec("dhidden", "token", "hidden")
_DLOGITS = GradientTensorSpec("dlogits", "token", "logits")
_DGATE = GradientTensorSpec("dgate", "token", "gate")
_DUP = GradientTensorSpec("dup", "token", "up")
_DW_LINEAR = GradientTensorSpec("dW", "parameter", "lm_head_weight")
_DRESIDUAL = GradientTensorSpec("dresidual", "token", "residual")
_DPRE = GradientTensorSpec("dpre", "token", "pre")


GRADIENT_ADAPTERS: dict[str, GradientAdapterSpec] = {
    "mhc_pre_h_aggregate": GradientAdapterSpec(
        op_name="mhc_pre_h_aggregate",
        chain_node="mhc_pre_h_aggregate",
        op_class="reduction",
        spec_name="mhc_pre_h_aggregate",
        tensors=(_DRESIDUAL, _DPRE),
        requirement="optional_fused",
        source_files=(
            "rl_engine/kernels/ops/cuda/mhc.py",
            "rl_engine/kernels/ops/pytorch/mhc.py",
            "csrc/cuda/mhc/mhc_pre_h_aggregate.cu",
            "csrc/cuda/mhc/mhc_pre_h_aggregate_kernel.cuh",
        ),
    ),
    "rms_norm": GradientAdapterSpec(
        op_name="rms_norm",
        chain_node="rms_norm",
        op_class="reduction",
        spec_name="rms_norm",
        tensors=(_DX, _DWEIGHT),
        requirement="required",
        source_files=(
            "rl_engine/kernels/ops/cuda/norm/rmsnorm.py",
            "rl_engine/kernels/ops/triton/rmsnorm_triton.py",
            "csrc/cuda/rmsnorm.cu",
        ),
    ),
    "qk_norm": GradientAdapterSpec(
        op_name="qk_norm",
        chain_node="qk_norm",
        op_class="reduction",
        spec_name="qk_norm",
        tensors=(_DX, _DWEIGHT),
        requirement="required",
        source_files=(
            "rl_engine/kernels/ops/cuda/norm/rmsnorm.py",
            "rl_engine/kernels/ops/triton/rmsnorm_triton.py",
            "csrc/cuda/rmsnorm.cu",
        ),
    ),
    "det_gemm": GradientAdapterSpec(
        op_name="det_gemm",
        chain_node="det_gemm",
        op_class="reduction",
        spec_name="det_gemm",
        tensors=(_DX_GEMM, _DW_GEMM),
        requirement="required",
        source_files=(
            "rl_engine/kernels/ops/cuda/matmul/det_gemm.py",
            "rl_engine/kernels/ops/triton/matmul/det_gemm.py",
            "csrc/cuda/gemm/det_gemm_kernel.cu",
        ),
    ),
    "attention": GradientAdapterSpec(
        op_name="attention",
        chain_node="attention",
        op_class="attention",
        spec_name="attention",
        tensors=(_DQ, _DK, _DV),
        requirement="required",
        source_files=(
            "rl_engine/kernels/ops/cuda/attention/deterministic_attn.py",
            "rl_engine/kernels/ops/triton/attention/standard_attn.py",
            "csrc/cuda/attention/deterministic_attention.cu",
        ),
    ),
    "embedding": GradientAdapterSpec(
        op_name="embedding",
        chain_node="embedding",
        op_class="elementwise",
        spec_name="embedding",
        tensors=(_DWEIGHT,),
        requirement="required",
        source_files=(
            "rl_engine/kernels/ops/cuda/linear/embedding.py",
            "rl_engine/kernels/ops/triton/linear/embedding.py",
            "csrc/cuda/embedding_lm_head_sm90.cu",
        ),
    ),
    "lm_head": GradientAdapterSpec(
        op_name="lm_head",
        chain_node="lm_head",
        op_class="reduction",
        spec_name="lm_head",
        tensors=(_DHIDDEN, _DWEIGHT),
        requirement="required",
        source_files=(
            "rl_engine/kernels/ops/cuda/linear/lm_head.py",
            "rl_engine/kernels/ops/triton/linear/lm_head.py",
            "csrc/cuda/embedding_lm_head_sm90.cu",
        ),
    ),
    "logp": GradientAdapterSpec(
        op_name="logp",
        chain_node="logprob",
        op_class="logprob",
        spec_name="logp",
        tensors=(_DLOGITS,),
        requirement="required",
        source_files=(
            "rl_engine/kernels/ops/cuda/loss/logp.py",
            "rl_engine/kernels/ops/triton/loss/logp.py",
            "csrc/fused_logp_kernel.cu",
            "csrc/deterministic_logp_kernel.cu",
        ),
    ),
    "batch_invariant_logp": GradientAdapterSpec(
        op_name="batch_invariant_logp",
        chain_node="batch_invariant_logp",
        op_class="logprob",
        spec_name="batch_invariant_logp",
        tensors=(_DLOGITS,),
        requirement="required",
        source_files=(
            "rl_engine/kernels/ops/cuda/loss/batch_invariant_logp.py",
            "rl_engine/kernels/ops/triton/loss/batch_invariant_logp.py",
            "csrc/cuda/batch_invariant_logp_kernel_sm90.cu",
        ),
    ),
    "linear_logp": GradientAdapterSpec(
        op_name="linear_logp",
        chain_node="linear_logp",
        op_class="logprob",
        spec_name="linear_logp",
        tensors=(_DHIDDEN, _DW_LINEAR),
        requirement="optional_fused",
        source_files=(
            "rl_engine/kernels/ops/cuda/loss/linear_logp.py",
            "rl_engine/kernels/ops/triton/loss/linear_logp.py",
            "csrc/cuda/fused_linear_logp_sm90.cu",
        ),
    ),
    "rope": GradientAdapterSpec(
        op_name="rope",
        chain_node="rope",
        op_class="elementwise",
        spec_name="rope",
        tensors=(_DX,),
        requirement="required",
        source_files=(
            "rl_engine/kernels/ops/cuda/rotary_embedding/rope.py",
            "rl_engine/kernels/ops/triton/rotary_embedding/rope.py",
            "csrc/cuda/rope_sm90.cu",
        ),
    ),
    "silu": GradientAdapterSpec(
        op_name="silu",
        chain_node="silu",
        op_class="elementwise",
        spec_name="silu",
        tensors=(_DX,),
        requirement="required",
        source_files=(
            "rl_engine/kernels/ops/cuda/activation/swiglu.py",
            "rl_engine/kernels/ops/triton/activation/swiglu.py",
            "csrc/cuda/activation.cu",
        ),
    ),
    "swiglu": GradientAdapterSpec(
        op_name="swiglu",
        chain_node="swiglu",
        op_class="elementwise",
        spec_name="swiglu",
        tensors=(_DGATE, _DUP),
        requirement="required",
        source_files=(
            "rl_engine/kernels/ops/cuda/activation/swiglu.py",
            "rl_engine/kernels/ops/triton/activation/swiglu.py",
            "csrc/cuda/activation.cu",
        ),
    ),
    "pack": GradientAdapterSpec(
        op_name="pack",
        chain_node="pack",
        op_class="elementwise",
        spec_name=None,
        tensors=(_DX,),
        requirement="layout_supported",
        source_files=("rl_engine/kernels/ops/pytorch/packing/pack.py",),
    ),
    "kv_cache_attention": GradientAdapterSpec(
        op_name="kv_cache_attention",
        chain_node="kv_cache_attention",
        op_class="attention",
        spec_name=None,
        tensors=(),
        requirement="absent_not_required",
        source_files=(),
    ),
}


def adapter_names() -> tuple[str, ...]:
    return tuple(GRADIENT_ADAPTERS)


def get_adapter(op_name: str) -> GradientAdapterSpec:
    try:
        return GRADIENT_ADAPTERS[op_name]
    except KeyError as exc:
        raise KeyError(f"unknown gradient adapter {op_name!r}") from exc


def required_gradient_adapters() -> tuple[GradientAdapterSpec, ...]:
    return tuple(
        spec
        for spec in GRADIENT_ADAPTERS.values()
        if spec.requirement in ("required", "layout_supported")
    )


def required_forward_adapters() -> tuple[GradientAdapterSpec, ...]:
    """Same enumerable WS1 ops as C4; C3 reuses the registry, not a second list."""

    return required_gradient_adapters()


@dataclass(frozen=True)
class _PhysicalPlan:
    """How one C2 config actually presents its tokens to the operator.

    ``row_keys`` is the physical row order the operator sees; ``None`` marks a
    pad row. ``call_spans`` splits those rows into the calls the layout implies
    (one call for a packed batch, one per chunk for chunked-prefill), so the
    operator's reduction shape genuinely changes across the matrix.
    """

    kind: str
    row_keys: tuple[tuple[str, int] | None, ...]
    call_spans: tuple[tuple[int, int], ...]
    batch: int
    padded_len: int


def _physical_plan(config: ConfigSpec) -> _PhysicalPlan:
    layout = config.physical_layout
    if isinstance(layout, PaddedBatch):
        rows: list[tuple[str, int] | None] = []
        for row_map in layout.restore_map:
            rows.extend(row_map)
        return _PhysicalPlan(
            kind="padded",
            row_keys=tuple(rows),
            call_spans=((0, len(rows)),),
            batch=len(layout.restore_map),
            padded_len=layout.padded_len,
        )
    if not isinstance(layout, PhysicalLayout):
        raise TypeError(f"unsupported physical layout {type(layout)!r}")
    rows = list(layout.restore_map)
    if layout.layout_kind == "chunked":
        spans = tuple(
            (int(offset), int(length))
            for offset, length in zip(layout.segment_offsets, layout.segment_lengths, strict=True)
        )
    else:
        spans = ((0, len(rows)),)
    return _PhysicalPlan(
        kind=layout.layout_kind,
        row_keys=tuple(rows),
        call_spans=spans,
        batch=1,
        padded_len=0,
    )


def _token_lookup(config: ConfigSpec) -> dict[tuple[str, int], Any]:
    return {
        (token.sample_id, token.token_position): token
        for sample in config.logical_batch.samples
        for token in sample.tokens()
    }


def _logical_fill(
    key: tuple[str, int] | None,
    tail: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    offset: int = 0,
) -> torch.Tensor:
    n = 1
    for dim in tail:
        n *= int(dim)
    if key is None:
        return torch.zeros((n,), device=device, dtype=dtype).reshape(tail)
    sample_ord = sum(ord(ch) for ch in key[0])
    position = key[1]
    axis = torch.arange(n, device=device, dtype=torch.int64)
    values = ((axis + sample_ord * 17 + position * 13 + offset * 11) % 257) - 128
    return (values.to(torch.float32) / 1024.0).to(dtype).reshape(tail)


def _shared_parameter(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    offset: int = 0,
) -> torch.Tensor:
    n = 1
    for dim in shape:
        n *= int(dim)
    axis = torch.arange(n, device=device, dtype=torch.int64)
    values = ((axis * 17 + offset * 13) % 257) - 128
    return (values.to(torch.float32) / 1024.0).to(dtype).reshape(shape)


def _stack_rows(
    keys: Sequence[tuple[str, int] | None],
    leading: tuple[int, ...],
    tail: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    offset: int = 0,
) -> torch.Tensor:
    rows = [_logical_fill(key, tail, device=device, dtype=dtype, offset=offset) for key in keys]
    return torch.stack(rows).reshape(leading + tail)


def _row_token_ids(
    keys: Sequence[tuple[str, int] | None],
    tokens: Mapping[tuple[str, int], Any],
    *,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    ids = [0 if key is None else int(tokens[key].token_id) % vocab_size for key in keys]
    return torch.tensor(ids, device=device, dtype=torch.long)


def _row_positions(keys: Sequence[tuple[str, int] | None], *, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [0 if key is None else int(key[1]) for key in keys], device=device, dtype=torch.long
    )


def _scaled_upstream(
    keys: Sequence[tuple[str, int] | None],
    tokens: Mapping[tuple[str, int], Any],
    tail: tuple[int, ...],
    *,
    active_token_denominator: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Upstream VJP seed: a pure function of logical identity, layout-independent.

    Seeding ``autograd.grad`` directly (instead of summing a scalar loss) keeps
    the comparison free of physical summation order, so a failure means the
    operator's own backward moved, not our harness reduction.
    """
    rows = []
    for key in keys:
        row = _logical_fill(key, tail, device=device, dtype=torch.float32, offset=3)
        scale = 0.0 if key is None or not tokens[key].is_active else 1.0
        rows.append(row * (scale / float(active_token_denominator)))
    stacked = torch.stack(rows) if rows else torch.zeros((0, *tail), device=device)
    return stacked.to(dtype)


def _call_operator(operator: Any, inputs: Mapping[str, Any]) -> Any:
    kwargs = dict(inputs)
    if hasattr(operator, "forward") and callable(operator.forward):
        return operator.forward(**kwargs)
    return operator(**kwargs)


def _requires_grad_inputs(inputs: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    named = set(names)
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor) and name in named:
            tensor = value.detach().clone()
            if not tensor.is_floating_point():
                raise TypeError(f"gradient input {name!r} must be floating point")
            tensor.requires_grad_(True)
            cloned[name] = tensor
        elif isinstance(value, torch.Tensor):
            cloned[name] = value.detach().clone()
        else:
            cloned[name] = value
    return cloned


def _first_output(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"operator output must be a Tensor or Tensor tuple, got {type(value)!r}")


def _require_differentiable(op_name: str, output: torch.Tensor) -> torch.Tensor:
    """Turn a non-differentiable candidate into a categorised red, not a traceback.

    An op wired straight to a C++ entry point (no ``torch.autograd.Function``)
    returns a tensor with no ``grad_fn`` even though its inputs require grad.
    """
    if output.grad_fn is None and not output.requires_grad:
        raise MissingBackwardError(
            op_name,
            "candidate is not wired through torch.autograd (no torch.autograd.Function)",
        )
    return output


def make_gradient_runner(
    op_name: str,
    operator: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    reference: bool,
    hidden: int = 64,
    vocab_size: int = 256,
    n_heads: int = 4,
    n_kv_heads: int = 1,
    head_dim: int = 16,
    backend_family: str | None = None,
    kernel_id: str | None = None,
) -> Callable[..., Any]:
    """Build a C2-config runner that returns named training-style gradients."""

    adapter = get_adapter(op_name)
    if adapter.requirement == "absent_not_required":
        raise RuntimeError(f"adapter {op_name!r} is not declared supported+differentiable")

    def run(config: ConfigSpec, **kwargs: Any) -> dict[str, torch.Tensor] | GradientObservation:
        denom = int(kwargs["active_token_denominator"])
        exec_dtype = (
            torch.bfloat16
            if op_name == "mhc_pre_h_aggregate"
            else torch.float32 if reference else dtype
        )
        grads = _run_adapter(
            adapter,
            operator,
            config,
            device=device,
            dtype=exec_dtype,
            hidden=hidden,
            vocab_size=vocab_size,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            active_token_denominator=denom,
        )
        if reference:
            return grads
        if backend_family is None or kernel_id is None:
            raise RuntimeError("candidate telemetry must declare backend_family and kernel_id")
        return GradientObservation(
            grads=grads,
            actual_backend=backend_family,
            kernel_id=kernel_id,
            # Parameter grads accumulate in FP32, so report the execution dtype
            # rather than whichever grad happens to come first.
            output_dtype=normalize_dtype_name(exec_dtype),
            device=str(device),
        )

    return run


def make_forward_runner(
    op_name: str,
    operator: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    reference: bool,
    hidden: int = 64,
    vocab_size: int = 256,
    n_heads: int = 4,
    n_kv_heads: int = 1,
    head_dim: int = 16,
    backend_family: str | None = None,
    kernel_id: str | None = None,
) -> Callable[..., Any]:
    """Build a C2-config runner that returns per-token forward outputs.

    Token maps are keyed by C2 ``(sample_id, token_position)`` so C3 can compare
    vector-valued ops (RMSNorm, GEMM, attention, …) without assuming logprob
    scalars. Inputs follow the same physical layout as ``make_gradient_runner``.
    """

    adapter = get_adapter(op_name)
    if adapter.requirement == "absent_not_required":
        raise RuntimeError(f"adapter {op_name!r} is not declared supported+differentiable")

    def run(
        config: ConfigSpec, **kwargs: Any
    ) -> dict[tuple[str, int], torch.Tensor] | RuntimeObservation:
        del kwargs
        exec_dtype = (
            torch.bfloat16
            if op_name == "mhc_pre_h_aggregate"
            else torch.float32 if reference else dtype
        )
        outputs = _run_forward(
            adapter,
            operator,
            config,
            device=device,
            dtype=exec_dtype,
            hidden=hidden,
            vocab_size=vocab_size,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        if reference:
            return outputs
        if backend_family is None or kernel_id is None:
            raise RuntimeError("candidate telemetry must declare backend_family and kernel_id")
        sample = next(iter(outputs.values()))
        return RuntimeObservation(
            output=outputs,
            actual_backend=backend_family,
            kernel_id=kernel_id,
            output_dtype=normalize_dtype_name(sample.dtype),
            device=str(device),
        )

    return run


def _row_parameters(
    op_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden: int,
    vocab_size: int,
    head_dim: int = 16,
) -> dict[str, torch.Tensor]:
    """Config-independent trainable parameters, built in the execution dtype."""
    if op_name == "rms_norm":
        return {"weight": _shared_parameter((hidden,), device=device, dtype=dtype, offset=1)}
    if op_name == "qk_norm":
        return {"weight": _shared_parameter((head_dim,), device=device, dtype=dtype, offset=1)}
    if op_name == "det_gemm":
        return {"b": _shared_parameter((hidden, hidden), device=device, dtype=dtype, offset=2)}
    if op_name == "linear_logp":
        return {
            "lm_head_weight": _shared_parameter(
                (vocab_size, hidden), device=device, dtype=dtype, offset=4
            )
        }
    if op_name == "embedding":
        return {
            "weight": _shared_parameter((vocab_size, hidden), device=device, dtype=dtype, offset=5)
        }
    if op_name == "lm_head":
        return {
            "weight": _shared_parameter((vocab_size, hidden), device=device, dtype=dtype, offset=6)
        }
    return {}


def _row_inputs(
    op_name: str,
    keys: Sequence[tuple[str, int] | None],
    tokens: Mapping[tuple[str, int], Any],
    params: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden: int,
    vocab_size: int,
    n_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Operator kwargs for one physical call span.

    Rows follow the layout's physical order, so packing, chunking, padding and
    permutation each hand the operator a genuinely different reduction shape.
    """
    n = len(keys)
    leading = (n,)
    if op_name == "rms_norm":
        return {
            "x": _stack_rows(keys, leading, (hidden,), device=device, dtype=dtype),
            "weight": params["weight"],
            "eps": 1.0e-6,
        }
    if op_name == "qk_norm":
        return {
            "x": _stack_rows(keys, leading, (head_dim,), device=device, dtype=dtype),
            "weight": params["weight"],
            "eps": 1.0e-6,
        }
    if op_name == "silu":
        return {"x": _stack_rows(keys, leading, (hidden,), device=device, dtype=dtype)}
    if op_name == "swiglu":
        return {
            "gate": _stack_rows(keys, leading, (hidden,), device=device, dtype=dtype, offset=0),
            "up": _stack_rows(keys, leading, (hidden,), device=device, dtype=dtype, offset=1),
        }
    if op_name == "mhc_pre_h_aggregate":
        return {
            "residual": _stack_rows(
                keys, leading, (4, MHC_PRE_HIDDEN_SIZE), device=device, dtype=dtype
            ),
            "pre": _stack_rows(
                keys,
                leading,
                (4,),
                device=device,
                dtype=torch.float32,
                offset=1,
            ),
        }
    if op_name == "det_gemm":
        return {
            "a": _stack_rows(keys, leading, (hidden,), device=device, dtype=dtype),
            "b": params["b"],
        }
    if op_name in {"logp", "batch_invariant_logp"}:
        ids = _row_token_ids(keys, tokens, vocab_size=vocab_size, device=device)
        target_key = "token_ids" if op_name == "logp" else "target_ids"
        return {
            "logits": _stack_rows(keys, leading, (vocab_size,), device=device, dtype=dtype),
            target_key: ids,
        }
    if op_name == "linear_logp":
        return {
            "hidden": _stack_rows(keys, leading, (hidden,), device=device, dtype=dtype),
            "lm_head_weight": params["lm_head_weight"],
            "target_ids": _row_token_ids(keys, tokens, vocab_size=vocab_size, device=device),
            "bias": None,
        }
    if op_name == "embedding":
        return {
            "token_ids": _row_token_ids(keys, tokens, vocab_size=vocab_size, device=device),
            "weight": params["weight"],
        }
    if op_name == "lm_head":
        return {
            "hidden": _stack_rows(keys, leading, (hidden,), device=device, dtype=dtype),
            "weight": params["weight"],
            "bias": None,
        }
    if op_name == "rope":
        rows = _stack_rows(keys, leading, (n_heads, head_dim), device=device, dtype=dtype)
        return {
            "x": rows.unsqueeze(0).permute(0, 2, 1, 3).contiguous(),
            "positions": _row_positions(keys, device=device),
            "theta": 1.0e6,
        }
    raise RuntimeError(f"no runnable gradient adapter for {op_name!r}")


def _to_rows(op_name: str, value: torch.Tensor, n_rows: int) -> torch.Tensor:
    """Normalize an operator output / input-grad back to (n_rows, *tail)."""
    if op_name == "rope":
        # RoPE runs as (1, heads, tokens, head_dim); tokens is the row axis.
        permuted = value.permute(0, 2, 1, 3)
        if permuted.shape[1] != n_rows:
            raise ValueError(f"{op_name} produced {permuted.shape[1]} rows, expected {n_rows}")
        return permuted.reshape(n_rows, permuted.shape[2], permuted.shape[3])
    if value.shape[0] != n_rows:
        raise ValueError(f"{op_name} produced {value.shape[0]} rows, expected {n_rows}")
    return value


def _assemble_token_grad(rows: Sequence[torch.Tensor], plan: _PhysicalPlan) -> torch.Tensor:
    """Stack physical rows into the tensor shape C2's restore helpers expect."""
    stacked = torch.stack(list(rows))
    if plan.kind == "padded":
        return stacked.reshape(plan.batch, plan.padded_len, *stacked.shape[1:])
    return stacked


def _run_row_stream(
    adapter: GradientAdapterSpec,
    operator: Any,
    config: ConfigSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden: int,
    vocab_size: int,
    n_heads: int,
    head_dim: int,
    active_token_denominator: int,
) -> dict[str, Any]:
    """Run a row-wise operator over the config's physical layout.

    Token gradients come back as physical tensors so the harness restores them
    through the C2 restore map, and parameter gradients accumulate in FP32
    across the layout's call spans.
    """
    plan = _physical_plan(config)
    tokens = _token_lookup(config)
    specs = adapter.tensors
    params = _row_parameters(
        adapter.op_name,
        device=device,
        dtype=dtype,
        hidden=hidden,
        vocab_size=vocab_size,
        head_dim=head_dim,
    )

    token_rows: dict[str, list[torch.Tensor | None]] = {
        spec.name: [None] * len(plan.row_keys) for spec in specs if spec.kind == "token"
    }
    param_totals: dict[str, torch.Tensor | None] = {
        spec.name: None for spec in specs if spec.kind == "parameter"
    }
    param_contributions: dict[str, dict[tuple[str, int], torch.Tensor]] = {
        spec.name: {} for spec in specs if spec.kind == "parameter"
    }

    for start, length in plan.call_spans:
        keys = plan.row_keys[start : start + length]
        inputs = _row_inputs(
            adapter.op_name,
            keys,
            tokens,
            params,
            device=device,
            dtype=dtype,
            hidden=hidden,
            vocab_size=vocab_size,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        if adapter.op_name == "mhc_pre_h_aggregate":
            prepared = inputs
            raw = _first_output(_call_operator(operator, prepared))
            out_rows = _to_rows(adapter.op_name, raw, length)
            upstream = _scaled_upstream(
                keys,
                tokens,
                tuple(out_rows.shape[1:]),
                active_token_denominator=active_token_denominator,
                device=device,
                dtype=out_rows.dtype,
            )
            backward_fp32 = getattr(operator, "backward_fp32", None)
            if not callable(backward_fp32):
                raise MissingBackwardError(
                    adapter.op_name, "candidate has no explicit FP32 backward"
                )
            grads = backward_fp32(
                upstream,
                prepared["residual"],
                prepared["pre"],
            )
        else:
            prepared = _requires_grad_inputs(inputs, [spec.source_input for spec in specs])
            raw = _require_differentiable(
                adapter.op_name, _first_output(_call_operator(operator, prepared))
            )
            out_rows = _to_rows(adapter.op_name, raw, length)
            upstream = _scaled_upstream(
                keys,
                tokens,
                tuple(out_rows.shape[1:]),
                active_token_denominator=active_token_denominator,
                device=device,
                dtype=out_rows.dtype,
            )
            grads = torch.autograd.grad(
                out_rows,
                [prepared[spec.source_input] for spec in specs],
                grad_outputs=upstream,
                allow_unused=True,
            )
        contribution_fn = getattr(operator, "parameter_vjp_contributions_fp32", None)
        contributions = (
            contribution_fn(**prepared, grad_output=upstream) if callable(contribution_fn) else None
        )
        for spec, grad in zip(specs, grads, strict=True):
            if grad is None:
                raise RuntimeError(
                    f"{adapter.op_name} produced no gradient for {spec.source_input!r}"
                )
            if spec.kind == "parameter":
                if contributions is not None:
                    per_row = contributions[spec.source_input]
                    if per_row.shape[0] != len(keys):
                        raise RuntimeError(
                            f"{adapter.op_name} {spec.name} VJP returned "
                            f"{per_row.shape[0]} rows, expected {len(keys)}"
                        )
                    for key, row in zip(keys, per_row, strict=True):
                        if key is not None:
                            param_contributions[spec.name][key] = row
                else:
                    total = param_totals[spec.name]
                    param_totals[spec.name] = (
                        grad.float() if total is None else total + grad.float()
                    )
            else:
                rows = _to_rows(adapter.op_name, grad, length)
                for index in range(length):
                    token_rows[spec.name][start + index] = rows[index]

    result: dict[str, Any] = {}
    for spec in specs:
        if spec.kind == "parameter":
            keyed = param_contributions[spec.name]
            total = param_totals[spec.name]
            if keyed:
                ordered = [keyed[key] for key in sorted(keyed)]
                total = torch.zeros_like(ordered[0], dtype=torch.float32)
                for contribution in ordered:
                    total = total + contribution.float()
            if total is None:
                raise RuntimeError(f"{adapter.op_name} produced no {spec.name}")
            result[spec.name] = total
        else:
            filled = token_rows[spec.name]
            if any(row is None for row in filled):
                raise RuntimeError(f"{adapter.op_name} left physical rows unfilled for {spec.name}")
            result[spec.name] = _assemble_token_grad(
                [row for row in filled if row is not None], plan
            )
    if any(param_contributions.values()):
        result["__parameter_contributions__"] = param_contributions
    return result


def _grid_keys(
    config: ConfigSpec, plan: _PhysicalPlan
) -> tuple[tuple[tuple[str, int] | None, ...], int, int]:
    """A (batch, length) token grid for operators that need whole sequences.

    Padded configs use their real pad grid; packed/chunked configs pad to the
    longest sample *in that config*, so B=1 and B=N differ genuinely.
    """
    if plan.kind == "padded":
        return plan.row_keys, plan.batch, plan.padded_len
    samples = config.logical_batch.samples
    length = max(sample.seq_len for sample in samples)
    keys: list[tuple[str, int] | None] = []
    for sample in samples:
        row = [(token.sample_id, token.token_position) for token in sample.tokens()]
        keys.extend(row)
        keys.extend([None] * (length - len(row)))
    return tuple(keys), len(samples), length


def _run_attention(
    operator: Any,
    config: ConfigSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    active_token_denominator: int,
) -> dict[str, Any]:
    plan = _physical_plan(config)
    grid, batch, length = _grid_keys(config, plan)
    tokens = _token_lookup(config)

    def _grid_tensor(heads: int, offset: int) -> torch.Tensor:
        rows = _stack_rows(
            grid, (batch * length,), (heads, head_dim), device=device, dtype=dtype, offset=offset
        )
        return rows.reshape(batch, length, heads, head_dim).permute(0, 2, 1, 3).contiguous()

    key_padding_mask = torch.tensor(
        [key is not None for key in grid], device=device, dtype=torch.bool
    ).reshape(batch, length)
    prepared = _requires_grad_inputs(
        {
            "q": _grid_tensor(n_heads, 0),
            "k": _grid_tensor(n_kv_heads, 1),
            "v": _grid_tensor(n_kv_heads, 2),
            "causal": True,
            "key_padding_mask": key_padding_mask,
        },
        ("q", "k", "v"),
    )
    output = _require_differentiable("attention", _first_output(_call_operator(operator, prepared)))
    upstream = _scaled_upstream(
        grid,
        tokens,
        (n_heads, head_dim),
        active_token_denominator=active_token_denominator,
        device=device,
        dtype=output.dtype,
    ).reshape(batch, length, n_heads, head_dim)
    grads = torch.autograd.grad(
        output,
        [prepared["q"], prepared["k"], prepared["v"]],
        grad_outputs=upstream.permute(0, 2, 1, 3).contiguous(),
    )

    result: dict[str, Any] = {}
    for name, grad in zip(("dQ", "dK", "dV"), grads, strict=True):
        physical = grad.permute(0, 2, 1, 3).contiguous()
        if plan.kind == "padded":
            result[name] = physical
        else:
            result[name] = {
                key: physical[index // length, index % length]
                for index, key in enumerate(grid)
                if key is not None
            }
    return result


def _run_pack(
    operator: Any,
    config: ConfigSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden: int,
    active_token_denominator: int,
) -> dict[str, Any]:
    plan = _physical_plan(config)
    grid, batch, length = _grid_keys(config, plan)
    tokens = _token_lookup(config)

    x = _stack_rows(grid, (batch * length,), (hidden,), device=device, dtype=dtype).reshape(
        batch, length, hidden
    )
    mask = torch.tensor([key is not None for key in grid], device=device, dtype=torch.bool).reshape(
        batch, length
    )
    prepared = _requires_grad_inputs({"x": x, "mask": mask}, ("x",))
    packed = _require_differentiable("pack", _first_output(_call_operator(operator, prepared)))
    # Packing keeps mask-true rows in row-major order; inactive tokens are
    # carried but must contribute zero, exactly like every other adapter.
    packed_keys = [key for key in grid if key is not None]
    upstream = _scaled_upstream(
        packed_keys,
        tokens,
        (hidden,),
        active_token_denominator=active_token_denominator,
        device=device,
        dtype=packed.dtype,
    )
    (grad,) = torch.autograd.grad(packed, [prepared["x"]], grad_outputs=upstream)
    if plan.kind == "padded":
        return {"dx": grad}
    return {
        "dx": {
            key: grad[index // length, index % length]
            for index, key in enumerate(grid)
            if key is not None
        }
    }


def _token_output_map(
    rows: Sequence[torch.Tensor],
    keys: Sequence[tuple[str, int] | None],
) -> dict[tuple[str, int], torch.Tensor]:
    result: dict[tuple[str, int], torch.Tensor] = {}
    for key, row in zip(keys, rows, strict=True):
        if key is not None:
            result[key] = row
    return result


def _run_row_stream_forward(
    adapter: GradientAdapterSpec,
    operator: Any,
    config: ConfigSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden: int,
    vocab_size: int,
    n_heads: int,
    head_dim: int,
) -> dict[tuple[str, int], torch.Tensor]:
    """Forward-only counterpart of ``_run_row_stream``."""

    plan = _physical_plan(config)
    tokens = _token_lookup(config)
    params = _row_parameters(
        adapter.op_name,
        device=device,
        dtype=dtype,
        hidden=hidden,
        vocab_size=vocab_size,
        head_dim=head_dim,
    )
    out_rows: list[torch.Tensor | None] = [None] * len(plan.row_keys)
    for start, length in plan.call_spans:
        keys = plan.row_keys[start : start + length]
        inputs = _row_inputs(
            adapter.op_name,
            keys,
            tokens,
            params,
            device=device,
            dtype=dtype,
            hidden=hidden,
            vocab_size=vocab_size,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        raw = _first_output(_call_operator(operator, inputs))
        rows = _to_rows(adapter.op_name, raw, length)
        for index in range(length):
            out_rows[start + index] = rows[index]
    filled = [row for row in out_rows if row is not None]
    if len(filled) != len(plan.row_keys):
        raise RuntimeError(f"{adapter.op_name} left physical rows unfilled")
    return _token_output_map(filled, plan.row_keys)


def _run_attention_forward(
    operator: Any,
    config: ConfigSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> dict[tuple[str, int], torch.Tensor]:
    plan = _physical_plan(config)
    grid, batch, length = _grid_keys(config, plan)

    def _grid_tensor(heads: int, offset: int) -> torch.Tensor:
        rows = _stack_rows(
            grid, (batch * length,), (heads, head_dim), device=device, dtype=dtype, offset=offset
        )
        return rows.reshape(batch, length, heads, head_dim).permute(0, 2, 1, 3).contiguous()

    key_padding_mask = torch.tensor(
        [key is not None for key in grid], device=device, dtype=torch.bool
    ).reshape(batch, length)
    output = _first_output(
        _call_operator(
            operator,
            {
                "q": _grid_tensor(n_heads, 0),
                "k": _grid_tensor(n_kv_heads, 1),
                "v": _grid_tensor(n_kv_heads, 2),
                "causal": True,
                "key_padding_mask": key_padding_mask,
            },
        )
    )
    physical = output.permute(0, 2, 1, 3).contiguous()
    return {
        key: physical[index // length, index % length]
        for index, key in enumerate(grid)
        if key is not None
    }


def _run_pack_forward(
    operator: Any,
    config: ConfigSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden: int,
) -> dict[tuple[str, int], torch.Tensor]:
    plan = _physical_plan(config)
    grid, batch, length = _grid_keys(config, plan)
    x = _stack_rows(grid, (batch * length,), (hidden,), device=device, dtype=dtype).reshape(
        batch, length, hidden
    )
    mask = torch.tensor([key is not None for key in grid], device=device, dtype=torch.bool).reshape(
        batch, length
    )
    packed = _first_output(_call_operator(operator, {"x": x, "mask": mask}))
    packed_keys = [key for key in grid if key is not None]
    if packed.shape[0] != len(packed_keys):
        raise ValueError(
            f"pack produced {packed.shape[0]} rows, expected {len(packed_keys)} active keys"
        )
    return {key: packed[index] for index, key in enumerate(packed_keys)}


def _run_forward(
    adapter: GradientAdapterSpec,
    operator: Any,
    config: ConfigSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden: int,
    vocab_size: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> dict[tuple[str, int], torch.Tensor]:
    if adapter.op_name == "attention":
        return _run_attention_forward(
            operator,
            config,
            device=device,
            dtype=dtype,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
    if adapter.op_name == "pack":
        return _run_pack_forward(
            operator,
            config,
            device=device,
            dtype=dtype,
            hidden=hidden,
        )
    return _run_row_stream_forward(
        adapter,
        operator,
        config,
        device=device,
        dtype=dtype,
        hidden=hidden,
        vocab_size=vocab_size,
        n_heads=n_heads,
        head_dim=head_dim,
    )


def _run_adapter(
    adapter: GradientAdapterSpec,
    operator: Any,
    config: ConfigSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden: int,
    vocab_size: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    active_token_denominator: int,
) -> dict[str, Any]:
    if adapter.op_name == "attention":
        return _run_attention(
            operator,
            config,
            device=device,
            dtype=dtype,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            active_token_denominator=active_token_denominator,
        )
    if adapter.op_name == "pack":
        return _run_pack(
            operator,
            config,
            device=device,
            dtype=dtype,
            hidden=hidden,
            active_token_denominator=active_token_denominator,
        )
    return _run_row_stream(
        adapter,
        operator,
        config,
        device=device,
        dtype=dtype,
        hidden=hidden,
        vocab_size=vocab_size,
        n_heads=n_heads,
        head_dim=head_dim,
        active_token_denominator=active_token_denominator,
    )


def _candidate_family(candidate: str) -> str:
    if candidate.startswith("cuda"):
        return "cuda"
    if candidate == "triton":
        return "triton"
    return candidate


def resolve_profile_candidate(
    adapter: GradientAdapterSpec,
    profile: str,
    manifest: WS1Manifest | None = None,
) -> dict[str, Any]:
    m = manifest if manifest is not None else load_manifest()
    if adapter.requirement == "absent_not_required":
        return {
            "status": "absent_not_required",
            "expected_backend_id": None,
            "candidate_path": None,
        }
    if adapter.requirement == "layout_supported":
        return {
            "status": "declared",
            "expected_backend_id": "pytorch",
            "candidate_path": "rl_engine.kernels.ops.pytorch.packing.pack.NativePackOp",
        }
    nodes = {item["node"]: item for item in profile_required_nodes(m, profile)}
    node = nodes.get(adapter.chain_node)
    if node is None and adapter.requirement == "optional_fused":
        return {
            "status": "optional",
            "expected_backend_id": None,
            "candidate_path": None,
        }
    if node is None:
        return {
            "status": "untracked_missing_node",
            "expected_backend_id": None,
            "candidate_path": None,
        }
    status = str(node.get("status", "declared"))
    expected = node.get("expected_backend_id")
    path = None
    if adapter.spec_name and expected:
        spec = OP_SPECS[adapter.spec_name]
        path = spec.candidate_paths.get(str(expected))
    return {
        "status": status,
        "expected_backend_id": expected,
        "candidate_path": path,
    }


def gradient_adapter_status_matrix(
    manifest: WS1Manifest | None = None,
    profiles: Sequence[str] = ("cuda_bf16", "triton_cuda_bf16"),
) -> tuple[AdapterStatusRow, ...]:
    m = manifest if manifest is not None else load_manifest()
    rows: list[AdapterStatusRow] = []
    for profile in profiles:
        expected_family = m.backend_profiles[profile]["backend_family"]
        for adapter in GRADIENT_ADAPTERS.values():
            resolved = resolve_profile_candidate(adapter, profile, m)
            status = str(resolved["status"])
            expected = resolved["expected_backend_id"]
            path = resolved["candidate_path"]
            tracked_red = status == "missing_required"
            untracked_red = False
            if adapter.requirement in ("required", "layout_supported"):
                if status == "untracked_missing_node":
                    untracked_red = True
                if status == "declared" and adapter.requirement == "required":
                    if not expected or not path:
                        untracked_red = True
                    elif _candidate_family(str(expected)) != expected_family:
                        untracked_red = True
            rows.append(
                AdapterStatusRow(
                    op_name=adapter.op_name,
                    chain_node=adapter.chain_node,
                    backend_profile=profile,
                    requirement=adapter.requirement,
                    candidate_status=status,
                    adapter_registered=True,
                    expected_backend_id=None if expected is None else str(expected),
                    candidate_path=None if path is None else str(path),
                    tracked_red=tracked_red,
                    untracked_red=untracked_red,
                    grad_tensor_names=tuple(tensor.name for tensor in adapter.tensors),
                )
            )
    return tuple(rows)


def load_adapter_operator(op_name: str, candidate: str) -> Any:
    adapter = get_adapter(op_name)
    if adapter.requirement == "layout_supported":
        return _load_object("rl_engine.kernels.ops.pytorch.packing.pack.NativePackOp")()
    if adapter.spec_name is None:
        raise RuntimeError(f"adapter {op_name!r} has no OP_SPECS entry")
    spec = OP_SPECS[adapter.spec_name]
    if candidate not in spec.candidate_paths:
        raise RuntimeError(f"operator {adapter.spec_name!r} has no candidate {candidate!r}")
    return _load_object(spec.candidate_paths[candidate])()


def load_adapter_gold(op_name: str) -> Any:
    adapter = get_adapter(op_name)
    if adapter.requirement == "layout_supported":
        gold = _load_object("rl_engine.kernels.ops.pytorch.packing.pack.NativePackOp")()
        return gold
    if adapter.spec_name is None:
        raise RuntimeError(f"adapter {op_name!r} has no gold path")
    spec = OP_SPECS[adapter.spec_name]
    gold_op = _load_object(spec.gold_path)()
    return getattr(gold_op, spec.gold_method)


def listed_source_paths(adapter: GradientAdapterSpec) -> list[Path]:
    return [REPO_ROOT / relative for relative in adapter.source_files]


__all__ = [
    "GRADIENT_ADAPTERS",
    "AdapterStatusRow",
    "GradientAdapterSpec",
    "adapter_names",
    "get_adapter",
    "gradient_adapter_status_matrix",
    "listed_source_paths",
    "load_adapter_gold",
    "load_adapter_operator",
    "make_forward_runner",
    "make_gradient_runner",
    "required_forward_adapters",
    "required_gradient_adapters",
    "resolve_profile_candidate",
]
