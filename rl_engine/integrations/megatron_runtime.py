# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Megatron runtime hooks installed without editing Megatron source files."""

from __future__ import annotations

import importlib
import io
import os
import pathlib
from collections.abc import Callable, Iterable
from types import MethodType
from typing import Any

import torch

from rl_engine.integrations.ablation import Implementation, IntegrationPlan
from rl_engine.integrations.framework_operators import (
    _MEGATRON_TP_OUTPUT_PROJECTION_COLLECTIVE_ATTR,
    _MEGATRON_TP_QKV_DGRAD_COLLECTIVE_ATTR,
    MegatronAttentionOperator,
    MegatronFFNOperator,
    MegatronLogpOperator,
    _fused_rms_norm_input,
    _strict_attention_projection_op,
)
from rl_engine.integrations.linear_logp import LinearLogpWrapper
from rl_engine.integrations.megatron import MegatronIntegration
from rl_engine.integrations.state import get_active_integration, set_active_integration
from rl_engine.integrations.vime.linear_logp_provider import _provider_impl

_PATCH_MARKER = "__rl_kernel_original_forward__"
_STRICT_ATTENTION_PATCH_MARKER = "__rl_kernel_original_strict_attention_init__"
_STRICT_ATTENTION_PROJECTION_MARKER = "__rl_kernel_strict_attention_projection__"
_STRICT_ATTENTION_CORE_MARKER = "__rl_kernel_strict_attention_core__"
_STRICT_TE_RMS_NORM_PATCH_MARKER = "__rl_kernel_original_strict_rms_norm_forward__"
_STRICT_LOGP_OUTPUT_PATCH_MARKER = "__rl_kernel_original_strict_logp_output_forward__"
_STRICT_LOGP_REUSABLE_MARKER = "__rl_kernel_reusable_local_logits__"
_STRICT_ROCM_ROPE_PATCH_MARKER = "__rl_kernel_original_rocm_rope_apply__"
_STRICT_LAYER_DIAGNOSTIC_PATCH_MARKER = "__rl_kernel_original_layer_diagnostic_forward__"


def _alignment_diagnostics_enabled() -> bool:
    value = os.getenv(
        "RL_KERNEL_LAYER_ALIGNMENT_DIAGNOSTICS",
        os.getenv("RL_KERNEL_ALIGNMENT_DIAGNOSTICS", ""),
    )
    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _diagnostic_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.getenv("RANK", "0"))


def _diagnostic_rows(row_count: int) -> torch.Tensor:
    requested = os.getenv("RL_KERNEL_ALIGNMENT_ROWS", "").strip()
    rows: list[int] = []
    for part in requested.split(",") if requested else ():
        start_text, separator, end_text = part.strip().partition("-")
        try:
            start = int(start_text)
            end = int(end_text) if separator else start
        except ValueError:
            continue
        rows.extend(range(max(start, 0), min(end + 1, row_count)))
    if not rows:
        rows = list(range(max(0, row_count - min(row_count, 64)), row_count))
    return torch.tensor(sorted(set(rows)), dtype=torch.long)


def _save_megatron_layer_diagnostic(
    instance: Any,
    input_value: torch.Tensor,
    output_value: torch.Tensor,
    intermediates: dict[str, torch.Tensor] | None = None,
) -> None:
    if not _alignment_diagnostics_enabled():
        return
    root = os.getenv("RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR", "").strip()
    if not root:
        return
    layer = int(getattr(instance, "layer_number", 0)) - 1
    requested_layers = os.getenv("RL_KERNEL_ALIGNMENT_LAYERS", "").strip()
    if requested_layers:
        try:
            enabled_layers = {int(value.strip()) for value in requested_layers.split(",")}
        except ValueError as exc:
            raise RuntimeError(
                "RL_KERNEL_ALIGNMENT_LAYERS must be comma-separated integers"
            ) from exc
        if layer not in enabled_layers:
            return
    if input_value.ndim == 3:
        if input_value.size(1) != 1 or output_value.size(1) != 1:
            raise RuntimeError("Megatron alignment diagnostics require batch size one")
        input_rows = input_value[:, 0]
        output_rows = output_value[:, 0]
    elif input_value.ndim == 2:
        input_rows = input_value
        output_rows = output_value
    else:
        raise RuntimeError("Megatron layer diagnostics require [S,B,H] or [T,H]")
    call_index = int(getattr(instance, "__rl_kernel_layer_diagnostic_call__", 0))
    setattr(instance, "__rl_kernel_layer_diagnostic_call__", call_index + 1)
    indices_cpu = _diagnostic_rows(int(input_rows.size(0)))
    indices = indices_cpu.to(device=input_rows.device)
    rank = _diagnostic_rank()
    payload = {
        "schema_version": "rlkernel.layer_alignment_diagnostic.v1",
        "framework": "megatron",
        "rank": rank,
        "layer": layer,
        "call_index": call_index,
        "row_indices": indices_cpu,
        "input": input_rows.detach().index_select(0, indices).cpu(),
        "output": output_rows.detach().index_select(0, indices).cpu(),
    }
    for name, value in (intermediates or {}).items():
        rows = value.detach()
        if rows.ndim >= 3 and rows.size(1) == 1:
            rows = rows[:, 0]
        if rows.ndim < 2 or rows.size(0) != input_rows.size(0):
            raise RuntimeError(
                f"Megatron layer diagnostic {name} does not expose the token axis first"
            )
        payload[name] = rows.index_select(0, indices).cpu()
    output_dir = pathlib.Path(root) / "layers"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        payload,
        output_dir
        / (
            f"megatron-pid{os.getpid()}-rank{rank:05d}-layer{layer:02d}-" f"call{call_index:08d}.pt"
        ),
    )


def _patch_layer_alignment_diagnostics() -> None:
    if not _alignment_diagnostics_enabled():
        return
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if hasattr(TransformerLayer, _STRICT_LAYER_DIAGNOSTIC_PATCH_MARKER):
        return
    original = TransformerLayer.forward

    def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
        input_value = args[0] if args else kwargs.get("hidden_states")
        intermediates: dict[str, torch.Tensor] = {}
        handles: list[Any] = []

        def tensor_result(value: Any) -> torch.Tensor | None:
            candidate = value[0] if isinstance(value, tuple) else value
            return candidate if isinstance(candidate, torch.Tensor) else None

        if int(getattr(instance, "layer_number", 0)) == 1:
            attention = instance.self_attention
            qkv_projection = attention.linear_qkv
            core_attention = attention.core_attention
            mlp = instance.mlp

            def qkv_hook(_module: Any, hook_args: tuple[Any, ...], hook_result: Any) -> None:
                qkv = tensor_result(hook_result)
                if qkv is not None and hook_args and isinstance(hook_args[0], torch.Tensor):
                    intermediates["attention_norm"] = _fused_rms_norm_input(
                        qkv_projection, hook_args[0], "linear_qkv"
                    )
                    intermediates["qkv"] = qkv

            def core_hook(
                _module: Any,
                hook_args: tuple[Any, ...],
                hook_kwargs: dict[str, Any],
                hook_result: Any,
            ) -> None:
                names = ("query", "key", "value")
                for index, name in enumerate(names):
                    value = hook_args[index] if len(hook_args) > index else hook_kwargs.get(name)
                    if isinstance(value, torch.Tensor):
                        intermediates[name] = value
                output = tensor_result(hook_result)
                if output is not None:
                    intermediates["attention_core"] = output

            def attention_hook(_module: Any, _args: tuple[Any, ...], hook_result: Any) -> None:
                output = tensor_result(hook_result)
                if output is not None:
                    intermediates["attention_output"] = output

            def mlp_hook(_module: Any, hook_args: tuple[Any, ...], hook_result: Any) -> None:
                output = tensor_result(hook_result)
                if output is not None and hook_args and isinstance(hook_args[0], torch.Tensor):
                    intermediates["mlp_norm"] = _fused_rms_norm_input(
                        mlp.linear_fc1, hook_args[0], "linear_fc1"
                    )
                    intermediates["mlp_output"] = output

            handles.extend(
                (
                    qkv_projection.register_forward_hook(qkv_hook),
                    core_attention.register_forward_hook(core_hook, with_kwargs=True),
                    attention.register_forward_hook(attention_hook),
                    mlp.register_forward_hook(mlp_hook),
                )
            )
        try:
            result = original(instance, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        output_value = result[0] if isinstance(result, tuple) else result
        if isinstance(input_value, torch.Tensor) and isinstance(output_value, torch.Tensor):
            attention_output = intermediates.get("attention_output")
            if attention_output is not None:
                intermediates["attention_residual"] = input_value + attention_output
            _save_megatron_layer_diagnostic(
                instance,
                input_value,
                output_value,
                intermediates,
            )
        return result

    setattr(TransformerLayer, _STRICT_LAYER_DIAGNOSTIC_PATCH_MARKER, original)
    TransformerLayer.forward = wrapped


def _strict_rocm_rope_positions(
    tensor: torch.Tensor,
    freqs: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cp_group: Any,
) -> torch.Tensor:
    """Recover Megatron's logical RoPE positions without a per-head loop.

    Megatron's packed THD input is stored as two CP-owned sequence chunks.  The
    deterministic ROCm kernel indexes a head-major view with ``row % S``;
    returning one position per token lets every head share one table and one
    launch while preserving the framework's zigzag ownership.
    """

    cp_size = 1 if cp_group is None else int(cp_group.size())
    cp_rank = 0 if cp_group is None else int(cp_group.rank())
    if tensor.ndim == 3:
        if cu_seqlens is None:
            raise RuntimeError("strict ROCm THD RoPE requires cu_seqlens")
        values = tuple(
            int(value) for value in cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
        )
        if len(values) < 2 or values[0] != 0:
            raise RuntimeError("strict ROCm THD RoPE received invalid cu_seqlens")
        exact_global_freqs = int(freqs.size(0)) == values[-1]
        positions: list[int] = []
        for index, (start, end) in enumerate(zip(values[:-1], values[1:], strict=True)):
            length = end - start
            if length <= 0 or length % cp_size:
                raise RuntimeError("strict ROCm THD RoPE sequence length is not CP divisible")
            local = length // cp_size
            if local % 2:
                raise RuntimeError("strict ROCm THD RoPE local length must be even")
            half = local // 2
            base = start if exact_global_freqs else 0
            positions.extend(range(base + cp_rank * half, base + (cp_rank + 1) * half))
            second = 2 * cp_size - cp_rank - 1
            positions.extend(range(base + second * half, base + (second + 1) * half))
        if len(positions) != tensor.size(0):
            raise RuntimeError(
                "strict ROCm THD RoPE position count does not match local token rows: "
                f"{len(positions)} != {tensor.size(0)}"
            )
        return torch.tensor(positions, dtype=torch.int64, device=tensor.device)

    if tensor.ndim != 4:
        raise RuntimeError(f"strict ROCm RoPE expects SBHD or THD tensors, got {tensor.shape}")
    sequence = int(tensor.size(0))
    if sequence <= 0 or (cp_size > 1 and sequence % 2):
        raise RuntimeError("strict ROCm SBHD RoPE received an invalid sequence length")
    if cp_size == 1:
        local_positions = list(range(sequence))
    else:
        half = sequence // 2
        second = 2 * cp_size - cp_rank - 1
        local_positions = list(range(cp_rank * half, (cp_rank + 1) * half))
        local_positions.extend(range(second * half, (second + 1) * half))
    positions = torch.tensor(local_positions, dtype=torch.int64, device=tensor.device)
    return positions.unsqueeze(0).expand(tensor.size(1), -1).contiguous()


def _apply_strict_rocm_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    operator: Any,
) -> torch.Tensor:
    """Apply the shared ROCm RoPE kernel in its head-major indexing layout."""

    if tensor.size(-1) % 2 or tensor.size(-1) <= 0:
        raise RuntimeError("strict ROCm RoPE requires an even head dimension")
    if tensor.ndim == 3:
        # Megatron THD is token-major; the kernel's table index is shared by
        # all heads, so transpose once and launch the operator over [H,T,D].
        head_major = tensor.transpose(0, 1).contiguous()
        return operator(head_major, positions).transpose(0, 1).contiguous()
    if tensor.ndim == 4:
        # SelfAttention uses [S,B,H,D].  The operator accepts [B,H,S,D] with
        # one position table per batch row.
        batch_head_major = tensor.permute(1, 2, 0, 3).contiguous()
        return operator(batch_head_major, positions).permute(2, 0, 1, 3).contiguous()
    raise RuntimeError(f"strict ROCm RoPE expects 3-D or 4-D tensors, got {tensor.shape}")


def _patch_strict_rocm_rope() -> None:
    """Use one deterministic RoPE implementation for Megatron and vLLM."""

    if torch.version.hip is None:
        return
    attention_module = importlib.import_module("megatron.core.transformer.attention")
    if hasattr(attention_module, _STRICT_ROCM_ROPE_PATCH_MARKER):
        return
    from rl_engine.kernels.ops.rocm.rotary_embedding.rope import RocmDeterministicRoPEOp

    operator = RocmDeterministicRoPEOp()
    original = attention_module.apply_rotary_pos_emb

    def strict_apply_rotary_pos_emb(
        tensor: torch.Tensor,
        freqs: torch.Tensor,
        config: Any,
        cu_seqlens: torch.Tensor | None = None,
        mscale: float = 1.0,
        cp_group: Any = None,
    ) -> torch.Tensor:
        if torch.version.hip is None:
            return original(tensor, freqs, config, cu_seqlens, mscale, cp_group)
        if bool(getattr(config, "rotary_interleaved", False)):
            raise RuntimeError("strict ROCm RoPE requires non-interleaved Qwen3 layout")
        if bool(getattr(config, "multi_latent_attention", False)) or float(mscale) != 1.0:
            raise RuntimeError("strict ROCm RoPE does not support MLA or mscale != 1")
        if freqs.ndim < 1 or int(freqs.shape[-1]) != int(tensor.shape[-1]):
            raise RuntimeError(
                "strict ROCm RoPE requires full-dimension frequency tensors: "
                f"freqs={tuple(freqs.shape)}, tensor={tuple(tensor.shape)}"
            )
        positions = _strict_rocm_rope_positions(tensor, freqs, cu_seqlens, cp_group)
        return _apply_strict_rocm_rope(tensor, positions, operator)

    strict_apply_rotary_pos_emb.__name__ = getattr(original, "__name__", "apply_rotary_pos_emb")
    setattr(attention_module, _STRICT_ROCM_ROPE_PATCH_MARKER, original)
    attention_module.apply_rotary_pos_emb = strict_apply_rotary_pos_emb


def _install_torch_dist_object_compatibility() -> None:
    """Normalize the PyTorch DCP object shape expected by this Megatron revision."""

    strategy = importlib.import_module("megatron.core.dist_checkpointing.strategies.torch")
    original = strategy._replace_sharded_keys_with_state_dict_keys
    if getattr(original, "__rl_kernel_dcp_object_compatibility__", False):
        return

    def wrapped(state_dict: dict[str, Any], flat_mapping: Any, rename_mapping: Any):
        normalized = {}
        for key, value in state_dict.items():
            if isinstance(value, io.BytesIO):
                value.seek(0)
                value = torch.load(value, weights_only=False)
            normalized[key] = value
        return original(normalized, flat_mapping, rename_mapping)

    wrapped.__rl_kernel_dcp_object_compatibility__ = True
    strategy._replace_sharded_keys_with_state_dict_keys = wrapped


class _DeterministicTensorParallelReduce(torch.autograd.Function):
    """Megatron reduce-from-TP semantics backed by the shared fixed tree."""

    @staticmethod
    def forward(ctx: Any, input_value: torch.Tensor, collective: Any | None) -> torch.Tensor:
        del ctx
        if collective is None:
            return input_value
        return collective.all_reduce(input_value.contiguous())

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        del ctx
        # ``reduce_from_tensor_model_parallel_region`` is an all-reduce in the
        # forward pass and identity in the backward pass.
        return grad_output, None


class _DeterministicCopyToTensorParallelRegion(torch.autograd.Function):
    """Megatron copy-to-TP semantics with a fixed-tree dgrad reduction."""

    @staticmethod
    def forward(ctx: Any, input_value: torch.Tensor, collective: Any | None) -> torch.Tensor:
        ctx.collective = collective
        return input_value

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        collective = ctx.collective
        if collective is None:
            return grad_output, None
        return collective.all_reduce(grad_output.contiguous()), None


def _deterministic_reduce_from_tensor_model_parallel_region(
    input_value: torch.Tensor,
    collective: Any | None,
) -> torch.Tensor:
    resolved = collective
    if resolved is not None and not callable(getattr(resolved, "all_reduce", None)):
        from rl_engine.distributed.collectives import collective_for_group

        resolved = collective_for_group(
            resolved,
            min_size_bytes=input_value.numel() * input_value.element_size(),
        )
    return _DeterministicTensorParallelReduce.apply(input_value, resolved)


def _deterministic_copy_to_tensor_model_parallel_region(
    input_value: torch.Tensor,
    collective: Any | None,
) -> torch.Tensor:
    return _DeterministicCopyToTensorParallelRegion.apply(input_value, collective)


def _module_tp_group(module: Any) -> Any | None:
    for name in ("tp_group", "_tp_group"):
        group = getattr(module, name, None)
        if group is not None:
            return group
    try:
        from megatron.core import parallel_state

        return parallel_state.get_tensor_model_parallel_group()
    except (ImportError, AssertionError, RuntimeError):
        return None


def _tp_world_size(group: Any | None) -> int:
    if (
        group is None
        or not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        return 1
    return int(torch.distributed.get_world_size(group=group))


def _fixed_tree_collective(
    module: Any,
    input_value: torch.Tensor | None = None,
) -> Any | None:
    group = _module_tp_group(module)
    if _tp_world_size(group) == 1:
        return None
    from rl_engine.distributed.collectives import collective_for_group

    min_size_bytes = 0 if input_value is None else input_value.numel() * input_value.element_size()
    collective = collective_for_group(group, min_size_bytes=min_size_bytes)
    if collective is None:
        raise RuntimeError("strict Attention TP collective is not initialized")
    backend_id = getattr(collective, "backend_id", None)
    if not isinstance(backend_id, str) or not backend_id.strip():
        raise RuntimeError("strict Attention TP collective has no backend identity")
    return collective


def _collective_backend_id(collective: Any | None) -> str:
    if collective is None:
        return "none"
    backend_id = getattr(collective, "backend_id", None)
    if not isinstance(backend_id, str) or not backend_id.strip():
        raise RuntimeError("strict Attention TP collective has no backend identity")
    return backend_id.strip()


class _DeterministicTPOutputProjection(torch.autograd.Function):
    """Materialize the strict TP LM head once and preserve its dgrad contract."""

    @staticmethod
    def forward(
        ctx: Any,
        input_value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        tp_group: Any,
    ) -> torch.Tensor:
        from rl_engine.kernels.ops.matmul.det_gemm import det_gemm_linear

        if input_value.ndim == 3:
            input_2d = input_value.transpose(0, 1).contiguous().reshape(-1, input_value.shape[-1])
            ctx.batch_major = True
        else:
            input_2d = input_value.reshape(-1, input_value.shape[-1]).contiguous()
            ctx.batch_major = False
        weight_2d = weight.contiguous()
        output_2d = det_gemm_linear(input_2d, weight_2d)
        if bias is not None:
            output_2d = (output_2d.float() + bias.float().reshape(1, -1)).to(torch.bfloat16)
        ctx.save_for_backward(input_2d, weight_2d)
        ctx.input_shape = input_value.shape
        ctx.input_dtype = input_value.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.has_bias = bias is not None
        ctx.tp_group = tp_group
        if ctx.batch_major:
            return (
                output_2d.reshape(input_value.shape[1], input_value.shape[0], weight.size(0))
                .transpose(0, 1)
                .contiguous()
            )
        return output_2d.reshape(*input_value.shape[:-1], weight.size(0))

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        from rl_engine.kernels.ops.cuda.loss.linear_logp import _deterministic_tp_all_reduce_
        from rl_engine.kernels.ops.matmul.det_gemm import (
            det_gemm_linear_input_gradient,
            det_gemm_linear_weight_gradient,
        )

        input_2d, weight = ctx.saved_tensors
        if ctx.batch_major:
            dlogits = grad_output.transpose(0, 1).contiguous().reshape(-1, grad_output.shape[-1])
        else:
            dlogits = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        if dlogits.dtype != torch.bfloat16:
            raise TypeError("strict TP LM-head backward requires BF16 dlogits")
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = det_gemm_linear_input_gradient(dlogits, weight)
            _deterministic_tp_all_reduce_(grad_input, ctx.tp_group)
            if ctx.batch_major:
                grad_input = (
                    grad_input.reshape(ctx.input_shape[1], ctx.input_shape[0], ctx.input_shape[2])
                    .transpose(0, 1)
                    .contiguous()
                )
            else:
                grad_input = grad_input.reshape(ctx.input_shape)
            grad_input = grad_input.to(ctx.input_dtype)
        if ctx.needs_input_grad[1]:
            grad_weight = det_gemm_linear_weight_gradient(input_2d, dlogits).to(ctx.weight_dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = dlogits.float().sum(dim=0).to(ctx.bias_dtype)
        return grad_input, grad_weight, grad_bias, None


def _optional_class(path: str) -> type[Any] | None:
    module_name, _, class_name = path.rpartition(".")
    try:
        value = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError):
        return None
    return value if isinstance(value, type) else None


def _discover_attention_classes() -> tuple[type[Any], ...]:
    paths = (
        "megatron.core.transformer.dot_product_attention.DotProductAttention",
        "megatron.core.extensions.transformer_engine.TEDotProductAttention",
        "megatron.core.extensions.transformer_engine.TEDotProductAttentionWithCP",
    )
    return tuple(value for path in paths if (value := _optional_class(path)) is not None)


def _discover_ffn_classes() -> tuple[type[Any], ...]:
    paths = ("megatron.core.transformer.mlp.MLP",)
    return tuple(value for path in paths if (value := _optional_class(path)) is not None)


def _unique_classes(values: Iterable[type[Any]]) -> tuple[type[Any], ...]:
    return tuple(dict.fromkeys(values))


def _patch_forward(
    cls: type[Any],
    *,
    integration: MegatronIntegration,
    module: str,
) -> None:
    if hasattr(cls, _PATCH_MARKER):
        raise RuntimeError(f"{cls.__module__}.{cls.__name__} is already RL-Kernel patched")
    original = cls.forward

    def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
        def native(_module: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            return original(instance, *call_args, **call_kwargs)

        return integration.execute(module, native, instance, *args, **kwargs)

    wrapped.__name__ = getattr(original, "__name__", "forward")
    wrapped.__doc__ = getattr(original, "__doc__", None)
    setattr(cls, _PATCH_MARKER, original)
    cls.forward = wrapped


def _patch_strict_attention_projections(
    *,
    self_attention_cls: type[Any] | None = None,
    column_linear_cls: type[Any] | None = None,
    row_linear_cls: type[Any] | None = None,
    det_gemm: Any | None = None,
    copy_to_tp: Callable[[torch.Tensor], torch.Tensor] | None = None,
    reduce_from_tp: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    """Use the shared deterministic GEMM for Megatron Attention projections."""

    if self_attention_cls is None or column_linear_cls is None or row_linear_cls is None:
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
        from megatron.core.transformer.attention import SelfAttention

        self_attention_cls = SelfAttention
        column_linear_cls = ColumnParallelLinear
        row_linear_cls = RowParallelLinear
    if hasattr(self_attention_cls, _STRICT_ATTENTION_PATCH_MARKER):
        return
    if det_gemm is None:
        det_gemm = _strict_attention_projection_op()

    attention_init = self_attention_cls.__init__
    column_forward_impl = column_linear_cls._forward_impl
    row_forward_impl = row_linear_cls._forward_impl

    def deterministic_projection(
        input_value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        input_2d = input_value.reshape(-1, input_value.shape[-1])
        linear = getattr(det_gemm, "linear", None)
        output_2d = (
            linear(input_2d, weight)
            if linear is not None
            else det_gemm(input_2d, weight.t().contiguous())
        )
        output = output_2d.reshape(*input_value.shape[:-1], weight.shape[0])
        return output if bias is None else output + bias

    def record_collective_backend(core_attention: Any, attribute: str, backend: str) -> None:
        if core_attention is not None:
            setattr(core_attention, attribute, backend)

    def callback_backend(callback: Callable[[torch.Tensor], torch.Tensor]) -> str:
        backend_id = getattr(callback, "backend_id", None)
        return (
            backend_id.strip()
            if isinstance(backend_id, str) and backend_id.strip()
            else "test_override"
        )

    def strict_tp_copy(
        module: Any,
        core_attention: Any,
        input_value: torch.Tensor,
    ) -> torch.Tensor:
        if copy_to_tp is not None:
            record_collective_backend(
                core_attention,
                _MEGATRON_TP_QKV_DGRAD_COLLECTIVE_ATTR,
                callback_backend(copy_to_tp),
            )
            return copy_to_tp(input_value)
        collective = _fixed_tree_collective(module, input_value)
        record_collective_backend(
            core_attention,
            _MEGATRON_TP_QKV_DGRAD_COLLECTIVE_ATTR,
            _collective_backend_id(collective),
        )
        return _deterministic_copy_to_tensor_model_parallel_region(input_value, collective)

    def strict_tp_reduce(
        module: Any,
        core_attention: Any,
        input_value: torch.Tensor,
    ) -> torch.Tensor:
        if reduce_from_tp is not None:
            record_collective_backend(
                core_attention,
                _MEGATRON_TP_OUTPUT_PROJECTION_COLLECTIVE_ATTR,
                callback_backend(reduce_from_tp),
            )
            return reduce_from_tp(input_value)
        collective = _fixed_tree_collective(module, input_value)
        record_collective_backend(
            core_attention,
            _MEGATRON_TP_OUTPUT_PROJECTION_COLLECTIVE_ATTR,
            _collective_backend_id(collective),
        )
        return _deterministic_reduce_from_tensor_model_parallel_region(input_value, collective)

    def bind_collective_identity(module: Any, core_attention: Any, attribute: str) -> None:
        collective = _fixed_tree_collective(module)
        record_collective_backend(
            core_attention,
            attribute,
            _collective_backend_id(collective),
        )

    def sequence_parallel_enabled(module: Any) -> bool:
        config = getattr(module, "config", None)
        return bool(getattr(module, "sequence_parallel", False)) or bool(
            getattr(config, "sequence_parallel", False)
        )

    def local_qkv_forward(
        module: Any,
        input_value: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        gather_output = bool(getattr(module, "gather_output", False))
        if runtime_gather_output is not None:
            gather_output = bool(runtime_gather_output)
        if gather_output:
            raise RuntimeError("strict Attention QKV does not support gathered column output")
        selected_weight = getattr(module, "weight", None) if weight is None else weight
        if not isinstance(selected_weight, torch.Tensor):
            raise RuntimeError("strict Attention QKV requires an allocated weight")
        core_attention = getattr(module, _STRICT_ATTENTION_CORE_MARKER, None)
        copied = strict_tp_copy(module, core_attention, input_value)
        skip_bias_add = bool(getattr(module, "skip_bias_add", False))
        bias = None if skip_bias_add else getattr(module, "bias", None)
        output = deterministic_projection(copied, selected_weight, bias)
        return output, getattr(module, "bias", None) if skip_bias_add else None

    def local_projection_forward(
        module: Any,
        input_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not bool(getattr(module, "input_is_parallel", False)):
            raise RuntimeError("strict Attention output projection requires TP-sharded input")
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError("strict Attention output projection requires an allocated weight")
        core_attention = getattr(module, _STRICT_ATTENTION_CORE_MARKER, None)
        output = deterministic_projection(input_value, weight, None)
        output = strict_tp_reduce(module, core_attention, output)
        skip_bias_add = bool(getattr(module, "skip_bias_add", False))
        bias = getattr(module, "bias", None)
        if not skip_bias_add and bias is not None:
            output = output + bias
        return output, bias if skip_bias_add else None

    def attention_init_wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
        attention_init(instance, *args, **kwargs)
        qkv = instance.linear_qkv
        projection = instance.linear_proj
        core_attention = getattr(instance, "core_attention", None)
        if sequence_parallel_enabled(qkv) or sequence_parallel_enabled(projection):
            raise RuntimeError(
                "strict Attention projection collectives do not support sequence parallelism"
            )
        setattr(qkv, _STRICT_ATTENTION_PROJECTION_MARKER, "qkv")
        setattr(projection, _STRICT_ATTENTION_PROJECTION_MARKER, "o_proj")
        object.__setattr__(qkv, _STRICT_ATTENTION_CORE_MARKER, core_attention)
        object.__setattr__(projection, _STRICT_ATTENTION_CORE_MARKER, core_attention)
        if copy_to_tp is None:
            bind_collective_identity(
                qkv,
                core_attention,
                _MEGATRON_TP_QKV_DGRAD_COLLECTIVE_ATTR,
            )
        else:
            record_collective_backend(
                core_attention,
                _MEGATRON_TP_QKV_DGRAD_COLLECTIVE_ATTR,
                callback_backend(copy_to_tp),
            )
        if reduce_from_tp is None:
            bind_collective_identity(
                projection,
                core_attention,
                _MEGATRON_TP_OUTPUT_PROJECTION_COLLECTIVE_ATTR,
            )
        else:
            record_collective_backend(
                core_attention,
                _MEGATRON_TP_OUTPUT_PROJECTION_COLLECTIVE_ATTR,
                callback_backend(reduce_from_tp),
            )
        if hasattr(qkv, "layer_norm_weight"):

            def te_qkv_forward(module: Any, input_value: torch.Tensor) -> Any:
                normalized = _fused_rms_norm_input(module, input_value, "linear_qkv")
                normalized = strict_tp_copy(module, core_attention, normalized)
                return deterministic_projection(normalized, module.weight, None), None

            def te_projection_forward(module: Any, input_value: torch.Tensor) -> Any:
                output = deterministic_projection(input_value, module.weight, None)
                return strict_tp_reduce(module, core_attention, output), None

            qkv.forward = MethodType(te_qkv_forward, qkv)
            projection.forward = MethodType(te_projection_forward, projection)
        else:
            # Per-instance forwards bypass Megatron's arithmetic collectives.
            qkv.allreduce_dgrad = False
            if callable(getattr(qkv, "forward", None)):
                qkv.forward = MethodType(local_qkv_forward, qkv)
            if callable(getattr(projection, "forward", None)):
                projection.forward = MethodType(local_projection_forward, projection)

    def column_forward_impl_wrapped(
        instance: Any,
        input: torch.Tensor,
        weight: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if getattr(instance, _STRICT_ATTENTION_PROJECTION_MARKER, None) == "qkv":
            core_attention = getattr(instance, _STRICT_ATTENTION_CORE_MARKER, None)
            input = strict_tp_copy(instance, core_attention, input)
            return deterministic_projection(input, weight, kwargs.get("bias"))
        return column_forward_impl(instance, input, weight, *args, **kwargs)

    def row_forward_impl_wrapped(
        instance: Any,
        input: torch.Tensor,
        weight: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if hasattr(instance, _STRICT_ATTENTION_PROJECTION_MARKER):
            return deterministic_projection(input, weight, kwargs.get("bias"))
        return row_forward_impl(instance, input, weight, *args, **kwargs)

    setattr(self_attention_cls, _STRICT_ATTENTION_PATCH_MARKER, attention_init)
    self_attention_cls.__init__ = attention_init_wrapped
    column_linear_cls._forward_impl = column_forward_impl_wrapped
    row_linear_cls._forward_impl = row_forward_impl_wrapped


def _patch_strict_te_rms_norm(rms_norm_cls: type[Any] | None = None) -> None:
    """Match standalone TE RMSNorm modules to the strict rollout arithmetic."""

    if rms_norm_cls is None:
        try:
            from transformer_engine.pytorch import RMSNorm
        except ImportError:
            return

        rms_norm_cls = RMSNorm
    if hasattr(rms_norm_cls, _STRICT_TE_RMS_NORM_PATCH_MARKER):
        return
    original = rms_norm_cls.forward

    def wrapped(instance: Any, input_value: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            raise RuntimeError("strict TE RMSNorm does not accept extra forward arguments")
        if bool(getattr(instance, "zero_centered_gamma", False)):
            raise RuntimeError("strict TE RMSNorm does not support zero-centered gamma")
        return torch.nn.functional.rms_norm(
            input_value,
            (input_value.shape[-1],),
            instance.weight,
            float(instance.eps),
        )

    setattr(rms_norm_cls, _STRICT_TE_RMS_NORM_PATCH_MARKER, original)
    rms_norm_cls.forward = wrapped


def _patch_strict_logp_output_layer(
    linear_cross_entropy_cls: type[Any] | None = None,
) -> None:
    """Replace Megatron's duplicate LM-head path with one reusable strict result."""

    if linear_cross_entropy_cls is None:
        from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule

        linear_cross_entropy_cls = LinearCrossEntropyModule
    if hasattr(linear_cross_entropy_cls, _STRICT_LOGP_OUTPUT_PATCH_MARKER):
        return
    original = linear_cross_entropy_cls.forward

    def wrapped(
        instance: Any,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
        output_cross_entropy_loss: bool = False,
        labels: torch.Tensor | None = None,
        reduction: str = "none",
        ignore_index: int = -100,
    ) -> Any:
        if output_cross_entropy_loss:
            return original(
                instance,
                input_,
                weight,
                runtime_gather_output,
                output_cross_entropy_loss,
                labels,
                reduction,
                ignore_index,
            )
        output_weight = instance.weight if weight is None else weight
        if output_weight is None:
            raise RuntimeError("strict TP LM head requires an explicit weight")
        gather_output = (
            instance.gather_output if runtime_gather_output is None else bool(runtime_gather_output)
        )
        if gather_output:
            raise RuntimeError("strict reusable TP LM head does not support gathered logits")
        if bool(getattr(instance, "sequence_parallel", False)):
            raise RuntimeError("strict reusable TP LM head does not support sequence parallelism")
        if bool(getattr(instance, "explicit_expert_comm", False)) or bool(
            getattr(instance, "disable_grad_reduce", False)
        ):
            raise RuntimeError("strict reusable TP LM head requires ordinary TP dgrad reduction")
        bias = instance.bias if not instance.skip_bias_add else None
        output = _DeterministicTPOutputProjection.apply(
            input_, output_weight, bias, instance.tp_group
        )
        instance._rl_kernel_local_logits = output
        output_bias = instance.bias if instance.skip_bias_add else None
        return output, output_bias

    wrapped.__name__ = getattr(original, "__name__", "forward")
    wrapped.__doc__ = getattr(original, "__doc__", None)
    setattr(linear_cross_entropy_cls, _STRICT_LOGP_OUTPUT_PATCH_MARKER, original)
    setattr(linear_cross_entropy_cls, _STRICT_LOGP_REUSABLE_MARKER, True)
    linear_cross_entropy_cls.forward = wrapped


def install_megatron_integration(
    plan: IntegrationPlan,
    *,
    attention_classes: Iterable[type[Any]] | None = None,
    ffn_classes: Iterable[type[Any]] | None = None,
) -> MegatronIntegration:
    """Install Attention, dense FFN and structural Logp routes in one actor."""

    existing = get_active_integration("megatron")
    if existing is not None:
        if not isinstance(existing, MegatronIntegration):
            raise RuntimeError("active Megatron integration has an unexpected type")
        if existing.plan != plan:
            raise RuntimeError("Megatron integration is already installed with another plan")
        return existing

    rl_kernel_operators: dict[str, Any] = {}
    if plan.implementation_for("attention", "training") is Implementation.RL_KERNEL:
        rl_kernel_operators["attention"] = MegatronAttentionOperator()
    if plan.implementation_for("ffn", "training") is Implementation.RL_KERNEL:
        rl_kernel_operators["ffn"] = MegatronFFNOperator()
    if plan.implementation_for("logp", "training") is Implementation.RL_KERNEL:
        rl_kernel_operators["logp"] = MegatronLogpOperator(
            _provider_impl,
            linear_logp=LinearLogpWrapper(),
        )
    integration = MegatronIntegration(plan, rl_kernel_operators=rl_kernel_operators)
    resolved_attention = _unique_classes(
        _discover_attention_classes() if attention_classes is None else attention_classes
    )
    resolved_ffn = _unique_classes(_discover_ffn_classes() if ffn_classes is None else ffn_classes)
    if (
        plan.implementation_for("attention", "training") is Implementation.RL_KERNEL
        and not resolved_attention
    ):
        raise RuntimeError("R/R Megatron Attention selected but no supported class was found")
    if plan.implementation_for("ffn", "training") is Implementation.RL_KERNEL and not resolved_ffn:
        raise RuntimeError("R/R Megatron FFN selected but no supported class was found")

    set_active_integration("megatron", integration)
    _patch_layer_alignment_diagnostics()
    if plan.implementation_for("logp", "training") is Implementation.RL_KERNEL:
        _patch_strict_logp_output_layer()
    if plan.implementation_for("attention", "training") is Implementation.RL_KERNEL:
        _patch_strict_rocm_rope()
        _patch_strict_attention_projections()
        if plan.implementation_for("ffn", "training") is Implementation.RL_KERNEL:
            _patch_strict_te_rms_norm()
    for cls in resolved_attention:
        _patch_forward(cls, integration=integration, module="attention")
    if resolved_attention:
        integration.record_installed_hook(
            "attention",
            ",".join(f"{cls.__module__}.{cls.__name__}.forward" for cls in resolved_attention),
        )
    for cls in resolved_ffn:
        _patch_forward(cls, integration=integration, module="ffn")
    if resolved_ffn:
        integration.record_installed_hook(
            "ffn",
            ",".join(f"{cls.__module__}.{cls.__name__}.forward" for cls in resolved_ffn),
        )
    if plan.implementation_for("logp", "training") is Implementation.RL_KERNEL:
        integration.record_installed_hook(
            "logp",
            "rl_engine.integrations.vime.linear_logp_provider.provider,"
            "megatron.core.transformer.linear_cross_entropy.LinearCrossEntropyModule.forward",
        )
    return integration


def initialize_from_environment(_args: Any = None) -> MegatronIntegration:
    """Vime-compatible custom-init entry point backed by the shared plan env."""

    from rl_engine.integrations.ablation import integration_plan_from_environment

    _install_torch_dist_object_compatibility()
    plan = integration_plan_from_environment()
    integration = install_megatron_integration(plan)
    if plan.implementation_for("attention", "training") is Implementation.RL_KERNEL:
        _precompile_strict_attention_training(_args)
    return integration


def _precompile_strict_attention_training(args: Any) -> None:
    """Warm FA4 CuTe fwd/bwd JIT outside Vime's actor_train timer."""

    if args is None or torch.version.hip is not None or not torch.cuda.is_available():
        return
    if os.getenv("RL_KERNEL_PRECOMPILE_FA4", "1") == "0":
        return

    from rl_engine.kernels.ops.cuda.attention.flash_attn import StrictFlashAttention4Core

    attention_heads = int(getattr(args, "num_attention_heads", 0) or 0)
    query_groups = int(getattr(args, "num_query_groups", 0) or attention_heads)
    tp_size = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    head_dim = int(
        getattr(args, "kv_channels", 0)
        or (int(getattr(args, "hidden_size", 0) or 0) // attention_heads)
    )
    if (
        attention_heads <= 0
        or query_groups <= 0
        or tp_size <= 0
        or attention_heads % tp_size
        or query_groups % tp_size
        or head_dim <= 0
    ):
        return

    params_dtype = getattr(args, "params_dtype", None)
    dtype = params_dtype if params_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    StrictFlashAttention4Core.precompile_training(
        q_heads=attention_heads // tp_size,
        kv_heads=query_groups // tp_size,
        head_dim=head_dim,
        dtype=dtype,
    )


__all__ = ["initialize_from_environment", "install_megatron_integration"]
