# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tensor-layout adapters from framework boundaries to semantic operators.

This module owns no numerical kernel selection. Attention and FFN instances
are resolved by :class:`OperatorBridge`; Logp uses the existing contract-aware
``KernelRegistry`` dispatch shared with the Vime provider.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, cast

import torch

from rl_engine.contracts.operators.attention import (
    STRICT_ATTENTION_FA4_SCHEDULE_ID,
    STRICT_ATTENTION_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
    AttentionRole,
)
from rl_engine.integrations.common.linear_logp import LinearLogpWrapper
from rl_engine.integrations.common.operators import (
    ATTENTION_BACKEND_ID,
    FFN_BACKEND_ID,
    SemanticOperatorHandle,
    _compact_attention_provenance,
    _dense_attention_contract,
    _device_name,
    _require_attention_accelerator,
    _require_nvidia_cuda,
    _split_gate_up,
    _strict_attention_projection_provenance,
    _tensor_cache_token,
    _weight,
)
from rl_engine.ops.gemm.det_gemm import det_gemm_backend_id
from rl_engine.reference.attention.ablation import AttentionAblationConfig
from rl_engine.reference.logprob.vocab_parallel_logp import BACKEND_ID as LOGP_BACKEND_ID
from rl_engine.runtime.policy import strict_contract_enabled

_MEGATRON_TP_QKV_DGRAD_COLLECTIVE_ATTR = "__rl_kernel_tp_qkv_dgrad_collective_backend__"


_MEGATRON_TP_OUTPUT_PROJECTION_COLLECTIVE_ATTR = (
    "__rl_kernel_tp_output_projection_collective_backend__"
)


def _strict_attention_platform_contract(platform: str) -> tuple[str, str, str]:
    if platform == "rocm":
        return (
            STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
            STRICT_ATTENTION_ROCM_SCHEDULE_ID,
            "rccl_ag_rs",
        )
    if platform == "cuda":
        return (
            STRICT_ATTENTION_PRODUCTION_CORE_ID,
            STRICT_ATTENTION_FA4_SCHEDULE_ID,
            "cuda_ag_rs",
        )
    raise RuntimeError(f"unsupported strict Attention platform {platform!r}")


def _fused_rms_norm_input(
    projection: Any,
    hidden_states: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Recover the RMSNorm hidden by TE's LayerNormLinear wrapper."""

    weight = getattr(projection, "layer_norm_weight", None)
    if weight is None:
        return hidden_states
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError(f"{name}.layer_norm_weight must be a tensor")
    if getattr(projection, "normalization", None) != "RMSNorm":
        raise RuntimeError(f"strict {name} requires fused RMSNorm")
    if getattr(projection, "layer_norm_bias", None) is not None:
        raise RuntimeError(f"strict {name} RMSNorm must be bias-free")
    if bool(getattr(projection, "zero_centered_gamma", False)):
        raise RuntimeError(f"strict {name} does not support zero-centered gamma")
    eps = float(getattr(projection, "eps"))
    from rl_engine.distributed.algorithms.canonical_cp import rms_norm

    return rms_norm(hidden_states, weight, eps)


def _megatron_parallel_state() -> Any:
    try:
        from megatron.core import parallel_state
    except ImportError as exc:  # pragma: no cover - exercised in framework environment
        raise RuntimeError("Megatron parallel_state is unavailable") from exc
    return parallel_state


@lru_cache(maxsize=512)
def _megatron_zigzag_layout(
    local_tokens: int,
    *,
    cp_rank: int,
    cp_world_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if cp_world_size == 1:
        positions = tuple(range(local_tokens))
        return positions, (0,), (0,), (0, local_tokens)
    if local_tokens % 2:
        raise RuntimeError("Megatron zigzag CP requires an even local sequence length")
    chunk_size = local_tokens // 2
    second_index = 2 * cp_world_size - cp_rank - 1
    starts = (cp_rank * chunk_size, second_index * chunk_size)
    positions = tuple(range(starts[0], starts[0] + chunk_size)) + tuple(
        range(starts[1], starts[1] + chunk_size)
    )
    return positions, (cp_rank, second_index), starts, (0, chunk_size, local_tokens)


def _packed_local_sequence_layout(
    packed_seq_params: Any,
    *,
    cp_world_size: int,
    local_query_tokens: int,
    local_kv_tokens: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Recover local THD sequence offsets from Megatron's global cu_seqlens."""

    if str(getattr(packed_seq_params, "qkv_format", "")).lower() != "thd":
        raise RuntimeError("strict RL-Kernel packed Attention requires qkv_format='thd'")
    query_cu = getattr(packed_seq_params, "cu_seqlens_q", None)
    kv_cu = getattr(packed_seq_params, "cu_seqlens_kv", None)
    if not isinstance(query_cu, torch.Tensor) or not isinstance(kv_cu, torch.Tensor):
        raise RuntimeError("packed Attention requires tensor cu_seqlens_q/cu_seqlens_kv")
    query_offsets = tuple(
        int(value) for value in query_cu.detach().to(device="cpu", dtype=torch.int64).tolist()
    )
    kv_offsets = tuple(
        int(value) for value in kv_cu.detach().to(device="cpu", dtype=torch.int64).tolist()
    )
    if query_offsets != kv_offsets:
        raise RuntimeError("strict self-Attention requires identical Q and KV cu_seqlens")
    if len(query_offsets) < 2 or query_offsets[0] != 0:
        raise RuntimeError("packed Attention cu_seqlens must start at zero")
    global_lengths = tuple(
        right - left for left, right in zip(query_offsets[:-1], query_offsets[1:], strict=True)
    )
    if any(length <= 0 for length in global_lengths):
        raise RuntimeError("packed Attention cu_seqlens must be strictly increasing")
    if any(length % cp_world_size for length in global_lengths):
        raise RuntimeError("packed Attention sequence lengths must be divisible by CP size")
    local_lengths = tuple(length // cp_world_size for length in global_lengths)
    local_offsets = [0]
    for length in local_lengths:
        local_offsets.append(local_offsets[-1] + length)
    if local_offsets[-1] != local_query_tokens or local_offsets[-1] != local_kv_tokens:
        raise RuntimeError("packed Attention cu_seqlens do not cover the local Q/KV token rows")
    return tuple(local_offsets), global_lengths


class MegatronAttentionOperator:
    """Materialize Megatron layout, then call the registered Attention wrapper."""

    backend_id = ATTENTION_BACKEND_ID

    def __init__(self, handle: SemanticOperatorHandle | None = None) -> None:
        self._handle = handle or SemanticOperatorHandle(
            target="training", semantic_op="attention", backend_id=self.backend_id
        )
        self._last_provenance: dict[str, Any] = {}
        self._packed_layout_owner: Any | None = None
        self._packed_layout_key: tuple[Any, ...] | None = None
        self._packed_layout_value: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self._position_ids_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    @staticmethod
    def _tp_collective_backend(module: Any, attribute: str, tp_world: int) -> str:
        value = getattr(module, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "none" if tp_world == 1 else "unbound"

    def _position_ids(
        self,
        positions: tuple[int, ...],
        *,
        batch_size: int,
        cp_rank: int,
        cp_world_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (
            len(positions),
            int(batch_size),
            int(cp_rank),
            int(cp_world_size),
            device.type,
            device.index,
        )
        cached = self._position_ids_cache.get(key)
        if cached is not None:
            return cached
        if len(self._position_ids_cache) >= 128:
            self._position_ids_cache.pop(next(iter(self._position_ids_cache)))
        value = torch.tensor(positions, dtype=torch.int64, device=device).repeat(batch_size, 1)
        self._position_ids_cache[key] = value
        return value

    def _packed_layout(
        self,
        packed_seq_params: Any,
        *,
        cp_world_size: int,
        local_query_tokens: int,
        local_kv_tokens: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        query_cu = getattr(packed_seq_params, "cu_seqlens_q", None)
        kv_cu = getattr(packed_seq_params, "cu_seqlens_kv", None)
        if not isinstance(query_cu, torch.Tensor) or not isinstance(kv_cu, torch.Tensor):
            raise RuntimeError("packed Attention requires tensor cu_seqlens_q/cu_seqlens_kv")
        key = (
            _tensor_cache_token(query_cu),
            _tensor_cache_token(kv_cu),
            cp_world_size,
            local_query_tokens,
            local_kv_tokens,
        )
        if self._packed_layout_owner is packed_seq_params and self._packed_layout_key == key:
            if self._packed_layout_value is None:
                raise RuntimeError("packed Attention layout cache is empty")
            return self._packed_layout_value
        value = _packed_local_sequence_layout(
            packed_seq_params,
            cp_world_size=cp_world_size,
            local_query_tokens=local_query_tokens,
            local_kv_tokens=local_kv_tokens,
        )
        self._packed_layout_owner = packed_seq_params
        self._packed_layout_key = key
        self._packed_layout_value = value
        return value

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "interface": "megatron.attention.forward",
            "operator": self.backend_id,
            "fallback": False,
            "semantic_instance": self._handle.provenance,
            "execution": dict(self._last_provenance),
        }

    def __call__(
        self,
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        attn_mask_type: Any = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: Any = None,
        num_splits: int | None = None,
    ) -> torch.Tensor:
        del attn_mask_type
        if attention_bias is not None:
            raise RuntimeError("strict RL-Kernel Attention does not accept bias")
        if num_splits not in (None, 1):
            raise RuntimeError("strict RL-Kernel Attention requires num_splits=1")
        if attention_mask is not None and attention_mask.numel() > 1:
            raise RuntimeError("strict RL-Kernel Attention supports only its causal contract")
        expected_ndim = 3 if packed_seq_params is not None else 4
        if query.ndim != expected_ndim or key.ndim != expected_ndim or value.ndim != expected_ndim:
            layout = "[T, H, D]" if packed_seq_params is not None else "[S, B, H, D]"
            raise RuntimeError(f"Megatron Attention Q/K/V must use {layout}")
        runtime_platform = _require_attention_accelerator(query)
        strict_core_id, strict_schedule, communication_backend = (
            _strict_attention_platform_contract(runtime_platform)
        )

        parallel_state = _megatron_parallel_state()
        cp_world = int(parallel_state.get_context_parallel_world_size())
        cp_rank = int(parallel_state.get_context_parallel_rank())
        tp_world = int(parallel_state.get_tensor_model_parallel_world_size())
        tp_rank = int(parallel_state.get_tensor_model_parallel_rank())
        cp_group = parallel_state.get_context_parallel_group() if cp_world > 1 else None
        operator = self._handle.get(
            query,
            topology={
                "world_size": tp_world * cp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": cp_world,
            },
        )
        operator.bind_accelerator_runtime(query, process_group=cp_group)
        scale = float(getattr(module, "softmax_scale", query.size(-1) ** -0.5))

        def execute_sequence(
            q_ready: torch.Tensor,
            k_ready: torch.Tensor,
            v_ready: torch.Tensor,
            *,
            global_sequence_length: int,
        ) -> Any:
            positions, block_indices, block_starts, block_offsets = _megatron_zigzag_layout(
                q_ready.size(2),
                cp_rank=cp_rank,
                cp_world_size=cp_world,
            )
            position_ids = self._position_ids(
                positions,
                batch_size=q_ready.size(0),
                cp_rank=cp_rank,
                cp_world_size=cp_world,
                device=q_ready.device,
            )
            return operator(
                q_ready,
                k_ready,
                v_ready,
                contract=_dense_attention_contract(
                    q_ready,
                    k_ready,
                    role=AttentionRole.TRAIN,
                    causal=True,
                    tp_rank=tp_rank,
                    tp_world_size=tp_world,
                    cp_rank=cp_rank,
                    cp_world_size=cp_world,
                    global_sequence_length=global_sequence_length,
                    global_block_indices=block_indices,
                    global_block_token_starts=block_starts,
                    local_block_offsets=block_offsets,
                ),
                config=AttentionAblationConfig(
                    strict_core_id=strict_core_id,
                    strict_schedule=strict_schedule,
                ),
                return_lse=True,
                communication_backend=communication_backend if cp_world > 1 else "none",
                query_position_ids=position_ids,
                key_position_ids=position_ids,
                scale=scale,
            )

        if packed_seq_params is None:
            q_ready = query.permute(1, 2, 0, 3).contiguous()
            k_ready = key.permute(1, 2, 0, 3).contiguous()
            v_ready = value.permute(1, 2, 0, 3).contiguous()
            result = execute_sequence(
                q_ready,
                k_ready,
                v_ready,
                global_sequence_length=q_ready.size(2) * cp_world,
            )
            context = result.out.permute(2, 0, 1, 3).contiguous()
            output = context.flatten(start_dim=2)
            execution_provenance: dict[str, Any] = {
                "packed_sequence_count": 0,
                "operator": _compact_attention_provenance(result.provenance),
            }
        else:
            local_offsets, global_lengths = self._packed_layout(
                packed_seq_params,
                cp_world_size=cp_world,
                local_query_tokens=query.size(0),
                local_kv_tokens=key.size(0),
            )
            grouped_sequences: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
            for sequence_index, (start, end, global_length) in enumerate(
                zip(
                    local_offsets[:-1],
                    local_offsets[1:],
                    global_lengths,
                    strict=True,
                )
            ):
                grouped_sequences.setdefault((end - start, global_length), []).append(
                    (sequence_index, start, end)
                )

            outputs: list[torch.Tensor | None] = [None] * len(global_lengths)
            sequence_provenance: list[dict[str, Any] | None] = [None] * len(global_lengths)
            launch_group_count = 0
            for (local_length, global_length), sequences in grouped_sequences.items():
                # Stack the cheap [H, T, D] views directly into FA4's
                # [B, H, T, D] layout.  Stacking before permuting would first
                # materialize [B, T, H, D] and then copy the entire tensor a
                # second time for every layer and microbatch.
                q_ready = torch.stack(
                    [query[start:end].permute(1, 0, 2) for _index, start, end in sequences],
                    dim=0,
                )
                k_ready = torch.stack(
                    [key[start:end].permute(1, 0, 2) for _index, start, end in sequences],
                    dim=0,
                )
                v_ready = torch.stack(
                    [value[start:end].permute(1, 0, 2) for _index, start, end in sequences],
                    dim=0,
                )
                result = execute_sequence(
                    q_ready,
                    k_ready,
                    v_ready,
                    global_sequence_length=global_length,
                )
                group_output = result.out.permute(0, 2, 1, 3).contiguous().flatten(start_dim=2)
                operator_provenance = _compact_attention_provenance(result.provenance)
                for batch_index, (sequence_index, _start, _end) in enumerate(sequences):
                    outputs[sequence_index] = group_output[batch_index]
                    sequence_provenance[sequence_index] = {
                        "sequence_index": sequence_index,
                        "local_tokens": local_length,
                        "global_tokens": global_length,
                        "operator": operator_provenance,
                    }
                launch_group_count += 1
            if any(item is None for item in outputs) or any(
                item is None for item in sequence_provenance
            ):
                raise RuntimeError("packed Attention failed to materialize every sequence")
            output = torch.cat(cast(list[torch.Tensor], outputs), dim=0)
            execution_provenance = {
                "packed_sequence_count": len(global_lengths),
                "launch_group_count": launch_group_count,
                "sequence_batching": "equal_length_rows",
                "sequences": cast(list[dict[str, Any]], sequence_provenance),
            }
        self._last_provenance = {
            "framework_layout": (
                "megatron_thd_packed_zigzag_cp"
                if packed_seq_params is not None
                else "megatron_sbh_zigzag_cp"
            ),
            "materialization": f"owner_local_zigzag_{communication_backend}",
            "cp_world_size": cp_world,
            "tp_world_size": tp_world,
            "runtime_platform": runtime_platform,
            "triton_used": runtime_platform == "rocm",
            "deterministic_projection": _strict_attention_projection_provenance(runtime_platform),
            "tp_qkv_dgrad_collective": self._tp_collective_backend(
                module,
                _MEGATRON_TP_QKV_DGRAD_COLLECTIVE_ATTR,
                tp_world,
            ),
            "tp_output_projection_collective": self._tp_collective_backend(
                module,
                _MEGATRON_TP_OUTPUT_PROJECTION_COLLECTIVE_ATTR,
                tp_world,
            ),
            **execution_provenance,
        }
        return output


class _MegatronCPWeightGradient(torch.autograd.Function):
    """Cancel the extra CP replica count before Megatron reduces parameter grads.

    The strict FFN gathers all CP tokens for its weight-gradient GEMMs, so
    each CP rank already has the complete gradient. Megatron still performs
    its normal DP/CP reduction; its loss scaling assumes rank-local grads.
    """

    @staticmethod
    def forward(ctx, weight, cp_world):
        from rl_engine.distributed.algorithms.canonical_cp import current_layout

        ctx.cp_world = cp_world
        ctx.cp_layout = current_layout()
        return weight

    @staticmethod
    def backward(ctx, grad):
        if ctx.cp_layout is not None:
            from rl_engine.distributed.algorithms.canonical_cp import replica_parameter_gradient

            grad = replica_parameter_gradient(grad, ctx.cp_world, ctx.cp_layout.cp_rank)
        else:
            grad = grad / ctx.cp_world
        return grad, None


class MegatronFFNOperator:
    backend_id = FFN_BACKEND_ID

    def __init__(self, handle: SemanticOperatorHandle | None = None) -> None:
        self._handle = handle or SemanticOperatorHandle(
            target="training", semantic_op="ffn", backend_id=self.backend_id
        )
        self._last_provenance: dict[str, Any] = {}

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "interface": "megatron.mlp.forward",
            "operator": self.backend_id,
            "fallback": False,
            "semantic_instance": self._handle.provenance,
            "execution": dict(self._last_provenance),
        }

    def __call__(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        per_token_scale: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        # Megatron's dense MLP path forwards padding_mask=None even when
        # no token padding is active. It is a routing-only argument and must
        # not be treated as expert-token scaling in the strict dense wrapper.
        if kwargs:
            unexpected = {
                name: value
                for name, value in kwargs.items()
                if name != "padding_mask" or value is not None
            }
            if unexpected:
                raise RuntimeError("strict dense Qwen3 FFN does not accept expert token scaling")
        if per_token_scale is not None:
            raise RuntimeError("strict dense Qwen3 FFN does not accept expert token scaling")
        _require_nvidia_cuda(hidden_states, "FFN")
        config = module.config
        if bool(getattr(config, "add_bias_linear", False)):
            raise RuntimeError("strict Qwen3 FFN requires bias-free projections")
        if not bool(getattr(config, "gated_linear_unit", False)):
            raise RuntimeError("strict Qwen3 FFN requires a gated linear unit")
        hidden_states = _fused_rms_norm_input(
            module.linear_fc1,
            hidden_states,
            "linear_fc1",
        )
        fused_gate_up = _weight(module.linear_fc1, "linear_fc1").contiguous()
        down = _weight(module.linear_fc2, "linear_fc2").contiguous()
        parallel_state = _megatron_parallel_state()
        cp_world = int(parallel_state.get_context_parallel_world_size())
        if cp_world > 1 and torch.is_grad_enabled():
            fused_gate_up = _MegatronCPWeightGradient.apply(fused_gate_up, cp_world)
            down = _MegatronCPWeightGradient.apply(down, cp_world)
        gate, up = _split_gate_up(fused_gate_up, "linear_fc1")
        tp_world = int(parallel_state.get_tensor_model_parallel_world_size())
        cp_group = parallel_state.get_context_parallel_group() if cp_world > 1 else None
        operator = self._handle.get(
            hidden_states,
            topology={
                "world_size": tp_world * cp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": cp_world,
            },
        )
        use_rocm_training_ffn = (
            torch.version.hip is not None
            and getattr(operator, "backend_id", None) == FFN_BACKEND_ID
        )
        ffn = operator
        if use_rocm_training_ffn:
            from rl_engine.backends.rocm.ffn import qwen3_ffn_training

            ffn = qwen3_ffn_training
        output = ffn(
            hidden_states,
            gate,
            up,
            down,
            fused_gate_up_weight=fused_gate_up,
            tp_group=getattr(module, "tp_group", None),
            cp_group=cp_group,
            sequence_parallel=bool(getattr(config, "sequence_parallel", False)),
            deterministic=True,
        )
        self._last_provenance = {
            "framework_layout": "megatron_sequence_parallel",
            "cp_world_size": cp_world,
            "tp_world_size": tp_world,
            "runtime_platform": _device_name(hidden_states),
            "actual_backend": (
                "rlkernel.rocm.det_gemm_swiglu"
                if torch.version.hip is not None
                else "rlkernel.cuda.det_gemm_swiglu"
            ),
            "gemm_backend": det_gemm_backend_id(),
            "fallback": False,
            "gate_up_projection": "packed_single_launch",
            "deterministic_all_reduce_backend": (
                "none"
                if tp_world == 1
                else (
                    "rocm_ipc_fixed_tree"
                    if torch.version.hip is not None
                    else "deterministic_all_reduce.ipc_localized_fixed_tree.v1"
                )
            ),
            "triton_used": torch.version.hip is not None,
        }
        return output, None


class MegatronLogpOperator:
    """Route structural Vime logp requests through the strict GPU backend."""

    def __init__(
        self,
        provider: Any,
        *,
        linear_logp: LinearLogpWrapper | None = None,
    ) -> None:
        self._provider = provider
        self._linear_logp = linear_logp
        self._last_provenance: dict[str, Any] = {}

    @property
    def backend_id(self) -> str:
        if self._linear_logp is not None:
            return self._linear_logp.backend_id
        return LOGP_BACKEND_ID

    @property
    def provenance(self) -> Mapping[str, Any]:
        return dict(self._last_provenance)

    def __call__(self, request: Any) -> Any:
        logits = getattr(request, "logits", None)
        context = getattr(request, "context", None)
        hidden = getattr(request, "hidden", None)
        if not isinstance(hidden, torch.Tensor):
            hidden = getattr(context, "hidden", None)
        strict = strict_contract_enabled()
        if isinstance(hidden, torch.Tensor):
            if self._linear_logp is None:
                raise RuntimeError("Megatron linear_logp route is not installed")
            _require_nvidia_cuda(hidden, "linear_logp")
            result = self._provider(request, linear_logp=self._linear_logp)
            self._last_provenance = {
                "interface": "vime.selected_logprob_provider",
                "operator": self.backend_id,
                "actual_backend": self.backend_id,
                "fallback": False,
                "runtime_platform": _device_name(hidden),
                "triton_used": torch.version.hip is not None,
                "provider": dict(getattr(result, "provenance", {})),
                "linear_logp": dict(self._linear_logp.provenance),
                "logits_materialized": False,
            }
            return result
        if strict:
            raise RuntimeError(
                "strict Megatron linear_logp request is missing hidden/LM-head structural inputs"
            )
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Megatron Logp request must expose logits or hidden")
        _require_nvidia_cuda(logits, "Logp")
        result = self._provider(request)
        self._last_provenance = {
            "interface": "vime.selected_logprob_provider",
            "operator": self.backend_id,
            "actual_backend": self.backend_id,
            "fallback": False,
            "runtime_platform": _device_name(logits),
            "triton_used": torch.version.hip is not None,
            "provider": dict(getattr(result, "provenance", {})),
        }
        return result
