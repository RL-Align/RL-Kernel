# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tensor-layout adapters from framework boundaries to semantic operators.

This module owns no numerical kernel selection. Attention and FFN instances
are resolved by :class:`OperatorBridge`; Logp uses the existing contract-aware
``KernelRegistry`` dispatch shared with the Vime provider.
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import lru_cache
from threading import Lock
from typing import Any, Callable, Mapping, cast

import torch

from rl_engine.alignment.cross_config.operators import OperatorBridge, OperatorOverride
from rl_engine.integrations.linear_logp import LinearLogpWrapper, take_rollout_linear_logp_context
from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_FA4_SCHEDULE_ID,
    STRICT_ATTENTION_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
    AttentionContract,
    AttentionDType,
    AttentionMode,
    AttentionRole,
)
from rl_engine.kernels.attention_contract import ReductionSpec as AttentionReductionSpec
from rl_engine.kernels.attention_contract import ShardingSpec as AttentionShardingSpec
from rl_engine.kernels.attention_contract import SplitKVSpec
from rl_engine.kernels.attention_projection import ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID
from rl_engine.kernels.logprob_contract import LogprobContract, LogprobDType, LogprobRole, MaskSpec
from rl_engine.kernels.logprob_contract import ReductionSpec as LogprobReductionSpec
from rl_engine.kernels.logprob_contract import ShardingSpec as LogprobShardingSpec
from rl_engine.kernels.ops.matmul.det_gemm import DetGemmOp, det_gemm_backend_id
from rl_engine.kernels.ops.pytorch.attention.ablation import AttentionAblationConfig
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import BACKEND_ID as LOGP_BACKEND_ID
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import DEFAULT_NUM_VOCAB_TILES
from rl_engine.kernels.semantic_registry import OperatorRequirements
from rl_engine.runtime_mode import strict_contract_enabled

ATTENTION_BACKEND_ID = "rlkernel.attention.deterministic.v1"
FFN_BACKEND_ID = "rlkernel.ffn.qwen3.deterministic.v1"
_MEGATRON_TP_QKV_DGRAD_COLLECTIVE_ATTR = "__rl_kernel_tp_qkv_dgrad_collective_backend__"
_MEGATRON_TP_OUTPUT_PROJECTION_COLLECTIVE_ATTR = (
    "__rl_kernel_tp_output_projection_collective_backend__"
)


def _device_name(tensor: torch.Tensor) -> str:
    if tensor.device.type == "cuda" and torch.version.hip is not None:
        return "rocm"
    return tensor.device.type


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _optional_env_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    """Describe a hot-path tensor without reading CUDA values on the host."""

    return {
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }


def _tensor_debug_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    stats = _tensor_metadata(detached)
    if detached.numel() == 0:
        return stats
    values = detached.float()
    stats.update(
        {
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
        }
    )
    return stats


def _diff_debug_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "shape_mismatch": True,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    diff = (left.detach().float() - right.detach().float()).abs()
    stats = _tensor_debug_stats(diff)
    stats["mismatch_count"] = int(torch.ne(left.detach(), right.detach()).sum().item())
    return stats


def _attention_dtype(tensor: torch.Tensor) -> AttentionDType:
    try:
        return {
            torch.bfloat16: AttentionDType.BF16,
            torch.float16: AttentionDType.FP16,
            torch.float32: AttentionDType.FP32,
        }[tensor.dtype]
    except KeyError as exc:
        raise RuntimeError(f"unsupported Attention dtype {tensor.dtype}") from exc


def _require_nvidia_cuda(tensor: torch.Tensor, module: str) -> None:
    if tensor.device.type != "cuda":
        raise RuntimeError(f"strict {module} R/R requires CUDA/ROCm GPU tensors")


def _require_attention_accelerator(tensor: torch.Tensor) -> str:
    """Return the real Attention platform behind PyTorch's CUDA device API."""

    if tensor.device.type != "cuda":
        raise RuntimeError("strict Attention R/R requires CUDA or ROCm GPU tensors")
    return "rocm" if torch.version.hip is not None else "cuda"


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


def _strict_attention_projection_backend_id(platform: str) -> str:
    if platform == "rocm":
        return ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID
    if platform == "cuda":
        return det_gemm_backend_id()
    raise RuntimeError(f"unsupported Attention projection platform {platform!r}")


def _strict_attention_projection_provenance(platform: str) -> dict[str, Any]:
    return {
        "backend_id": _strict_attention_projection_backend_id(platform),
        "deterministic": True,
        "accumulation_dtype": "fp32",
        "reduction_order": "k_ascending",
        "split_k": False,
        "roles": ["qkv", "o_proj"],
        "triton_used": platform == "rocm",
    }


def _strict_attention_projection_op() -> Any:
    """Construct the deterministic projection selected for this PyTorch build."""

    return DetGemmOp()


class SemanticOperatorHandle:
    """Resolve one exact semantic backend once for one framework process."""

    def __init__(self, *, target: str, semantic_op: str, backend_id: str) -> None:
        if target not in {"training", "rollout"}:
            raise ValueError("target must be 'training' or 'rollout'")
        self.target = target
        self.semantic_op = semantic_op
        self.backend_id = backend_id
        self._bridge = OperatorBridge()
        self._instance: Any | None = None
        self._provenance: dict[str, Any] | None = None
        self._runtime_device: torch.device | None = None
        self._runtime_dtype: torch.dtype | None = None
        self._lock = Lock()

    def get(
        self,
        tensor: torch.Tensor,
        *,
        topology: Mapping[str, Any],
        factory_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        # This method is called from vLLM model forwards that may be captured
        # by torch.compile.  A Lock context manager is unsupported in a
        # Dynamo fullgraph, so keep the hot-path lookup lock-free.  Handles
        # are constructed per framework worker and resolution is idempotent;
        # duplicate first-call resolution is harmless and the bridge caches
        # the resulting semantic instance.
        if self._instance is not None:
            # vLLM constructs the plugin before its worker TP group exists, so
            # the eager prime may observe TP=1 while the first model call has
            # TP=2. The operator receives the live group at invocation time;
            # keep device and dtype strict, but do not reject this topology
            # transition after the semantic instance is resolved.
            if tensor.device != self._runtime_device or tensor.dtype != self._runtime_dtype:
                raise RuntimeError(
                    f"{self.semantic_op} runtime device/dtype changed after resolution"
                )
            return self._instance
        requirements = OperatorRequirements(
            device=_device_name(tensor),
            dtype=_dtype_name(tensor),
            topology=topology,
            alignment_properties={"deterministic": True},
        )
        target = cast(Any, self.target)
        resolved = self._bridge.resolve_override(
            OperatorOverride.for_target(
                semantic_op=self.semantic_op,
                backend_id=self.backend_id,
                target=target,
            ),
            requirements={self.target: requirements},
            strict=True,
        )
        instance = self._bridge.instantiate(
            resolved,
            target=target,
            factory_kwargs=factory_kwargs,
            cache=True,
        )
        provenance = self._bridge.instance_provenance(
            resolved,
            target=target,
            instance=instance,
        )
        actual_backend = getattr(instance, "backend_id", None)
        if actual_backend != self.backend_id:
            raise RuntimeError(
                f"semantic registry resolved {self.backend_id!r} but instantiated "
                f"{actual_backend!r}"
            )
        self._instance = instance
        self._provenance = provenance.to_dict()
        self._runtime_device = tensor.device
        self._runtime_dtype = tensor.dtype
        return instance

    @property
    def provenance(self) -> Mapping[str, Any] | None:
        return None if self._provenance is None else dict(self._provenance)


def _weight(module: Any, name: str) -> torch.Tensor:
    value = getattr(module, "weight", None)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{name} must expose an unquantized torch.Tensor weight")
    if value.ndim != 2:
        raise RuntimeError(f"{name}.weight must be two-dimensional")
    return value


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
    return torch.nn.functional.rms_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight,
        eps,
    )


def _split_gate_up(weight: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.size(0) % 2:
        raise RuntimeError(f"{name}.weight first dimension must contain equal gate/up shards")
    gate, up = weight.chunk(2, dim=0)
    return gate.contiguous(), up.contiguous()


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


def _tensor_cache_token(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Identify one tensor value while it is reused across framework layers."""

    try:
        version: int | None = int(tensor._version)
    except RuntimeError:
        # vLLM warmup runs under inference mode.  Inference tensors have no
        # version counter, but their storage address and metadata remain valid
        # cache identity for the lifetime of that model forward.
        version = None
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
        version,
    )


@lru_cache(maxsize=1)
def _rocm_paged_kv_max_tokens() -> int | None:
    """Return an optional workload bound for AITER paged scheduling."""

    value = os.environ.get("RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS", "").strip()
    if not value:
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise RuntimeError("RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS must be an integer") from exc
    if limit <= 0:
        raise RuntimeError("RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS must be positive")
    return limit


def _compact_attention_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep strict backend identity without retaining one record per token."""

    compact = dict(value)
    rows = compact.pop("core_rows", None)
    if isinstance(rows, (list, tuple)):
        compact["core_row_count"] = len(rows)
        backends = sorted(
            {
                str(row["actual_backend"])
                for row in rows
                if isinstance(row, Mapping) and row.get("actual_backend")
            }
        )
        compact["core_actual_backends"] = backends
    return compact


def _dense_attention_contract(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    role: AttentionRole,
    causal: bool,
    tp_rank: int,
    tp_world_size: int,
    cp_rank: int = 0,
    cp_world_size: int = 1,
    mode: AttentionMode | None = None,
    global_sequence_length: int | None = None,
    global_block_indices: tuple[int, ...] = (0,),
    global_block_token_starts: tuple[int, ...] = (0,),
    local_block_offsets: tuple[int, ...] | None = None,
) -> AttentionContract:
    batch, q_heads, query_tokens, head_dim = query.shape
    kv_heads = key.size(1)
    return AttentionContract(
        role=role,
        mode=mode or AttentionMode.PREFILL,
        dtype=_attention_dtype(query),
        batch_size=batch,
        query_sequence_length=query_tokens,
        head_dim=head_dim,
        causal=causal,
        causal_offsets=(0,) * batch if causal else None,
        sharding=AttentionShardingSpec(
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            global_q_heads=q_heads * tp_world_size,
            global_kv_heads=kv_heads * tp_world_size,
            local_q_head_start=tp_rank * q_heads,
            local_q_heads=q_heads,
            local_kv_head_start=tp_rank * kv_heads,
            local_kv_heads=kv_heads,
            global_sequence_length=global_sequence_length or query_tokens,
            local_sequence_length=query_tokens,
            global_block_indices=global_block_indices,
            global_block_token_starts=global_block_token_starts,
            local_block_offsets=local_block_offsets or (0, query_tokens),
        ),
        reduction=AttentionReductionSpec(),
        split_kv=SplitKVSpec.disabled(),
        export_lse=True,
    )


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
        gate, up = _split_gate_up(fused_gate_up, "linear_fc1")
        down = _weight(module.linear_fc2, "linear_fc2").contiguous()
        parallel_state = _megatron_parallel_state()
        cp_world = int(parallel_state.get_context_parallel_world_size())
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
        output = operator(
            hidden_states,
            gate,
            up,
            down,
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
            "gate_up_projection": "separate_strict_launches",
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


def _vllm_tp_coordinates() -> tuple[int, int, Any]:
    try:
        from vllm.distributed.parallel_state import get_tp_group

        coordinator = get_tp_group()
        group = getattr(coordinator, "device_group", coordinator)
        rank = int(getattr(coordinator, "rank_in_group", 0))
        world = int(getattr(coordinator, "world_size", 1))
        return world, rank, group
    except (ImportError, AssertionError):
        return 1, 0, None


def _vllm_kv_cache_views(
    kv_cache: torch.Tensor,
    *,
    head_size: int,
    num_kv_heads: int | None = None,
    platform: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CUDA and ROCm vLLM caches as [blocks, block, kv_heads, head]."""

    def normalize(plane: torch.Tensor) -> torch.Tensor:
        if plane.ndim != 4 or plane.size(-1) != head_size:
            raise RuntimeError("vLLM K/V cache planes must be 4-D with a head-size tail")
        if num_kv_heads is None or plane.size(2) == num_kv_heads:
            # [blocks, block, heads, head] (LBHNC after selecting K/V).
            return plane
        if plane.size(1) == num_kv_heads:
            # [blocks, heads, block, head].
            return plane.permute(0, 2, 1, 3)
        if plane.size(0) == num_kv_heads:
            # [heads, blocks, block, head] (LHBNC after selecting K/V).
            return plane.permute(1, 2, 0, 3)
        raise RuntimeError("vLLM K/V cache layout does not expose the declared number of KV heads")

    # RL-Kernel's ROCm backend allocates token-major pages so AITER CK can read
    # the paged cache directly.  Splitting the packed tail is then a zero-copy
    # [blocks, block, heads, head] view with page-major strides.  Native AITER
    # keeps the head dimension before the block dimension and still needs the
    # transpose below.
    if kv_cache.ndim == 4 and kv_cache.size(-1) == 2 * head_size:
        if (
            platform == "rocm"
            and num_kv_heads is not None
            and kv_cache.size(2) == num_kv_heads
            and kv_cache.size(1) != num_kv_heads
        ):
            return kv_cache.split(head_size, dim=-1)
        return kv_cache.transpose(1, 2).split(head_size, dim=-1)
    if (
        kv_cache.ndim == 4
        and platform == "rocm"
        and kv_cache.size(1) == 2
        and num_kv_heads is not None
        and kv_cache.size(-1) == num_kv_heads * head_size
    ):
        key_cache, value_cache = kv_cache.unbind(1)
        blocks, block_size, _packed_heads = key_cache.shape
        return (
            key_cache.view(blocks, block_size, num_kv_heads, head_size),
            value_cache.view(blocks, block_size, num_kv_heads, head_size),
        )
    # The K/V axis is leading in CUDA FlashAttention caches and follows the
    # block axis in the ROCm AITER layouts.  Prefer the platform convention in
    # the ambiguous two-block case where both dimensions happen to equal two.
    if kv_cache.ndim == 5 and platform == "rocm" and kv_cache.size(1) == 2:
        key_cache, value_cache = kv_cache.unbind(1)
        return normalize(key_cache), normalize(value_cache)
    if kv_cache.ndim == 5 and kv_cache.size(0) == 2:
        key_cache, value_cache = kv_cache.unbind(0)
        return normalize(key_cache), normalize(value_cache)
    if kv_cache.ndim == 5 and kv_cache.size(1) == 2:
        key_cache, value_cache = kv_cache.unbind(1)
        return normalize(key_cache), normalize(value_cache)
    if (
        kv_cache.ndim == 4
        and kv_cache.size(1) == 2
        and num_kv_heads is not None
        and kv_cache.size(-1) == num_kv_heads * head_size
    ):
        key_cache, value_cache = kv_cache.unbind(1)
        blocks, block_size, _packed_heads = key_cache.shape
        return (
            key_cache.view(blocks, block_size, num_kv_heads, head_size),
            value_cache.view(blocks, block_size, num_kv_heads, head_size),
        )
    raise RuntimeError("vLLM Attention KV cache does not match a supported CUDA/ROCm paged layout")


class VllmAttentionOperator:
    """Materialize logical rows from vLLM paged KV, then call the wrapper."""

    backend_id = ATTENTION_BACKEND_ID

    def __init__(
        self,
        handle: SemanticOperatorHandle | None = None,
        *,
        projection_collective_backend: Callable[[], str | None] | None = None,
    ) -> None:
        self._handle = handle or SemanticOperatorHandle(
            target="rollout", semantic_op="attention", backend_id=self.backend_id
        )
        self._projection_collective_backend = projection_collective_backend
        self._last_provenance: dict[str, Any] = {}
        self._tp_coordinates: tuple[int, int, Any] | None = None
        self._metadata_cache_key: tuple[Any, ...] | None = None
        self._metadata_cache_owners: set[int] = set()
        self._metadata_cache_value: tuple[list[dict[str, Any]], dict[str, Any]] | None = None
        self._dense_prefill_layout_key: tuple[Any, ...] | None = None
        self._dense_prefill_layout_value: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self._dense_prefill_position_ids: dict[tuple[Any, ...], torch.Tensor] = {}
        self._rocm_decode_warmup_key: tuple[Any, ...] | None = None
        self._rocm_paged_metadata_key: tuple[Any, ...] | None = None
        self._rocm_paged_metadata_owners: set[int] = set()
        self._rocm_paged_metadata_value: dict[str, Any] | None = None
        self._rocm_kv_indptr_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._phase_provenance: dict[str, dict[str, Any]] = {}

    def bind_inference(self) -> None:
        """Resolve the backend after vLLM has selected the worker CUDA device."""

        if self._tp_coordinates is not None:
            return
        tp_world, tp_rank, tp_group = _vllm_tp_coordinates()
        device = torch.device("cuda", torch.cuda.current_device())
        self._handle.get(
            torch.empty((1,), device=device, dtype=torch.bfloat16),
            topology={
                "world_size": tp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": 1,
            },
        )
        self._tp_coordinates = (tp_world, tp_rank, tp_group)

    def warmup_rocm_decode(self, impl: Any, *, dtype: torch.dtype) -> None:
        """Warm the exact strict decode schedule before rollout timing."""

        if torch.version.hip is None or dtype not in (torch.float16, torch.bfloat16):
            return
        device = torch.device("cuda", torch.cuda.current_device())
        key = (
            device.index,
            dtype,
            int(impl.num_heads),
            int(impl.num_kv_heads),
            int(impl.head_size),
        )
        if self._rocm_decode_warmup_key == key:
            return
        if self._tp_coordinates is None:
            self.bind_inference()
        assert self._tp_coordinates is not None
        tp_world, _tp_rank, _tp_group = self._tp_coordinates
        bound = self._handle.get(
            torch.empty((1,), device=device, dtype=dtype),
            topology={
                "world_size": tp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": 1,
            },
        )
        runtime = bound.bind_accelerator_runtime(torch.empty((1,), device=device, dtype=dtype))
        page_size = 16
        core = getattr(runtime, "_core", None)
        if getattr(core, "attention_backend", "ck") == "triton":
            from rl_engine.kernels.ops.triton.attention import chunked_flash_attn

            chunked_flash_attn.warmup(
                device,
                num_q_heads=int(impl.num_heads),
                num_kv_heads=int(impl.num_kv_heads),
                dtype=dtype,
            )
            from rl_engine.kernels.ops.triton.matmul import mfma_gemm

            mfma_gemm.warmup(device)
        q = torch.empty(
            (1, int(impl.num_heads), 1, int(impl.head_size)),
            device=device,
            dtype=dtype,
        )
        cache_shape = (
            1,
            page_size,
            int(impl.num_kv_heads),
            int(impl.head_size),
        )
        k_cache = torch.empty(cache_shape, device=device, dtype=dtype)
        v_cache = torch.empty_like(k_cache)
        page_table = torch.zeros((1, 1), device=device, dtype=torch.int32)
        seqused_k = torch.full((1,), page_size, device=device, dtype=torch.int32)
        cu_seqlens_q = torch.tensor((0, 1), device=device, dtype=torch.int32)
        kv_indptr = torch.tensor((0, 1), device=device, dtype=torch.int32)
        out = torch.empty_like(q)
        with torch.inference_mode():
            runtime.forward_paged_with_lse(
                q,
                k_cache,
                v_cache,
                page_table=page_table,
                seqused_k=seqused_k,
                max_seqlen_k=page_size,
                scale=float(impl.scale),
                out=out,
                return_lse=False,
                page_table_validated=True,
                cu_seqlens_q=cu_seqlens_q,
                kv_indptr=kv_indptr,
            )
            dense_q = torch.empty(
                (1, int(impl.num_heads), page_size, int(impl.head_size)),
                device=device,
                dtype=dtype,
            )
            dense_k = k_cache.permute(0, 2, 1, 3).contiguous()
            dense_v = v_cache.permute(0, 2, 1, 3).contiguous()
            positions = torch.arange(page_size, device=device, dtype=torch.int64).unsqueeze(0)
            runtime._core.forward_with_lse(
                dense_q,
                dense_k,
                dense_v,
                causal=True,
                scale=float(impl.scale),
                query_position_ids=positions,
                key_position_ids=positions,
            )
        torch.cuda.synchronize(device)
        self._rocm_decode_warmup_key = key

    @property
    def provenance(self) -> Mapping[str, Any]:
        execution = self._phase_provenance.get("decode", self._last_provenance)
        return {
            "interface": "vllm.attention.forward",
            "operator": self.backend_id,
            "fallback": False,
            "semantic_instance": self._handle.provenance,
            "execution": {
                **dict(execution),
                "captured_attention_phases": sorted(self._phase_provenance),
            },
        }

    def _record_phase_provenance(self, phase: str, provenance: dict[str, Any]) -> None:
        self._last_provenance = provenance
        self._phase_provenance[phase] = provenance

    @staticmethod
    def _metadata_tensor(metadata: Any, *names: str) -> torch.Tensor:
        for name in names:
            value = getattr(metadata, name, None)
            if isinstance(value, torch.Tensor):
                return value
        raise RuntimeError(f"vLLM Attention metadata is missing {'/'.join(names)}")

    def _dense_prefill_layout(
        self,
        starts_cpu: torch.Tensor,
        lengths_cpu: torch.Tensor,
        *,
        num_prefills: int,
        num_actual: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Cache the host-only request layout across the decoder layers."""

        key = (
            _tensor_cache_token(starts_cpu),
            _tensor_cache_token(lengths_cpu),
            num_prefills,
            num_actual,
        )
        if self._dense_prefill_layout_key == key:
            if self._dense_prefill_layout_value is None:
                raise RuntimeError("dense prefill layout cache is empty")
            return self._dense_prefill_layout_value
        starts = tuple(int(v) for v in starts_cpu[: num_prefills + 1].tolist())
        lengths = tuple(int(v) for v in lengths_cpu[:num_prefills].tolist())
        value = (starts, lengths)
        self._dense_prefill_layout_key = key
        self._dense_prefill_layout_value = value
        return value

    def _dense_prefill_positions(
        self,
        length: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        key = (int(length), device.type, device.index)
        cached = self._dense_prefill_position_ids.get(key)
        if cached is not None:
            return cached
        if len(self._dense_prefill_position_ids) >= 128:
            self._dense_prefill_position_ids.pop(next(iter(self._dense_prefill_position_ids)))
        value = torch.arange(length, dtype=torch.int64, device=device).unsqueeze(0)
        self._dense_prefill_position_ids[key] = value
        return value

    def _rocm_dense_prefill(
        self,
        impl: Any,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        attn_metadata: Any,
        runtime: Any,
        *,
        tp_rank: int,
        tp_world: int,
        num_actual: int,
    ) -> torch.Tensor | None:
        """Schedule pure prefill as dense per-request AITER launches.

        vLLM hands prefill Q/K/V in logical token order. Reusing those tensors
        avoids turning every token into a paged decode row (thousands of
        launches for one prompt) while keeping the strict runtime's one KV
        group reduction order identical to training.
        """

        if key is None or value is None:
            return None
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            return None
        if int(getattr(attn_metadata, "num_decodes", 0)) != 0:
            return None
        num_prefills = int(getattr(attn_metadata, "num_prefills", 0))
        if num_prefills <= 0 or int(getattr(attn_metadata, "num_extends", 0)) != 0:
            return None
        starts_cpu = getattr(attn_metadata, "_rlk_query_start_loc_cpu", None)
        lengths_cpu = getattr(attn_metadata, "_rlk_seq_lens_cpu", None)
        if not isinstance(starts_cpu, torch.Tensor) or starts_cpu.device.type != "cpu":
            return None
        if not isinstance(lengths_cpu, torch.Tensor) or lengths_cpu.device.type != "cpu":
            return None
        if starts_cpu.numel() < num_prefills + 1 or lengths_cpu.numel() < num_prefills:
            return None
        starts, lengths = self._dense_prefill_layout(
            starts_cpu,
            lengths_cpu,
            num_prefills=num_prefills,
            num_actual=num_actual,
        )
        if starts[0] != 0 or starts[-1] != num_actual:
            return None
        if any(
            end <= start or end - start != length
            for start, end, length in zip(starts[:-1], starts[1:], lengths, strict=True)
        ):
            return None

        output_heads = output.view(output.size(0), impl.num_heads, impl.head_size)
        for start, end in zip(starts[:-1], starts[1:], strict=True):
            query_row = query[start:end].permute(1, 0, 2).unsqueeze(0).contiguous()
            key_row = key[start:end].permute(1, 0, 2).unsqueeze(0).contiguous()
            value_row = value[start:end].permute(1, 0, 2).unsqueeze(0).contiguous()
            position_ids = self._dense_prefill_positions(
                end - start,
                device=query.device,
            )
            result = runtime.forward_with_lse(
                query_row,
                key_row,
                value_row,
                contract=_dense_attention_contract(
                    query_row,
                    key_row,
                    role=AttentionRole.INFER,
                    causal=True,
                    tp_rank=tp_rank,
                    tp_world_size=tp_world,
                    mode=AttentionMode.PREFILL,
                    global_sequence_length=end - start,
                ),
                causal=True,
                scale=float(impl.scale),
                cp_world_size=1,
                query_position_ids=position_ids,
                key_position_ids=position_ids,
                positions_are_sorted=True,
            )
            # vLLM uses a rank-3 output buffer at this boundary
            # ([tokens, heads, head_dim]); older integrations may provide the
            # equivalent flattened rank-2 view. Write through output_heads so
            # the adapter preserves the framework-owned layout in both cases.
            result_output = result.out.permute(0, 2, 1, 3).squeeze(0)
            output_group = output_heads.narrow(0, start, end - start)
            if result_output.shape != output_group.shape:
                raise RuntimeError(
                    "strict ROCm dense prefill output shape does not match vLLM: "
                    f"result={tuple(result_output.shape)}, "
                    f"output={tuple(output_group.shape)}"
                )
            output_group.copy_(result_output)
        if num_actual < output.size(0):
            output[num_actual:].zero_()
        self._record_phase_provenance(
            "prefill",
            {
                "framework_layout": "vllm_dense_qkv_prefill",
                "materialization": "direct_dense_qkv_to_aiter_ck",
                "tp_world_size": tp_world,
                "runtime_platform": "rocm",
                "triton_used": True,
                "prefill_request_count": num_prefills,
                "prefill_token_count": num_actual,
                "core_launch_count": num_prefills,
                "deterministic_projection": _strict_attention_projection_provenance("rocm"),
                "deterministic_all_reduce_backend": "unbound" if tp_world > 1 else "none",
                "direct_output_buffer": True,
            },
        )
        return output

    def _rocm_direct_paged_metadata(
        self,
        attn_metadata: Any,
        *,
        block_table: torch.Tensor,
        block_size: int,
        num_actual: int,
        cache_owner: Any,
        runtime_core: Any = None,
    ) -> tuple[dict[str, Any], bool] | None:
        """Prepare graph-safe paged metadata once and share it across layers."""

        num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
        num_prefills = int(getattr(attn_metadata, "num_prefills", 0))
        num_extends = int(getattr(attn_metadata, "num_extends", 0))
        unified = bool(getattr(runtime_core, "supports_mixed_paged_batches", False))
        if unified:
            # The chunked Triton contract serves every request kind from one
            # causal launch, so mixed decode/extend/prefill batches need no
            # per-kind expansion.
            if num_decodes + num_extends + num_prefills <= 0 or num_actual <= 0:
                return None
            if num_extends or (num_decodes > 0 and num_prefills > 0):
                mode = "mixed"
            elif num_prefills > 0:
                mode = "prefill"
            else:
                mode = "decode"
            sequence_count = num_decodes + num_extends + num_prefills
            query_start_loc = self._metadata_tensor(attn_metadata, "query_start_loc")
            max_seqlen_q = int(getattr(attn_metadata, "max_query_len", 0) or 0)
            if max_seqlen_q <= 0:
                lengths = [
                    int(getattr(meta, "max_query_len", 0) or 0)
                    for meta in (
                        getattr(attn_metadata, "decode_metadata", None),
                        getattr(attn_metadata, "extend_metadata", None),
                        getattr(attn_metadata, "prefill_metadata", None),
                    )
                    if meta is not None
                ]
                max_seqlen_q = max(lengths) if lengths else 0
            causal = True
        elif num_extends or (num_decodes > 0) == (num_prefills > 0):
            return None
        else:
            mode = "prefill" if num_prefills > 0 else "decode"
            sequence_count = num_prefills if num_prefills > 0 else num_decodes
        if sequence_count <= 0 or num_actual <= 0:
            return None
        if unified:
            pass
        elif mode == "prefill":
            prefill = getattr(attn_metadata, "prefill_metadata", None)
            if prefill is None:
                return None
            query_start_loc = getattr(prefill, "query_start_loc", None)
            max_seqlen_q = int(getattr(prefill, "max_query_len", 0))
            causal = bool(getattr(attn_metadata, "causal", True))
        else:
            decode = getattr(attn_metadata, "decode_metadata", None)
            if decode is None:
                return None
            query_start_loc = self._metadata_tensor(attn_metadata, "query_start_loc")
            max_seqlen_q = int(getattr(decode, "max_query_len", 0))
            causal = False
        if not isinstance(query_start_loc, torch.Tensor) or max_seqlen_q <= 0:
            return None
        query_starts_source = query_start_loc
        seq_lens_source = self._metadata_tensor(attn_metadata, "seq_lens")
        max_seq_len = int(getattr(attn_metadata, "max_seq_len", block_table.size(1) * block_size))
        configured_kv_limit = _rocm_paged_kv_max_tokens()
        kernel_max_seqlen_k = (
            max_seq_len if configured_kv_limit is None else min(max_seq_len, configured_kv_limit)
        )
        page_count = min(
            block_table.size(1),
            (max_seq_len + block_size - 1) // block_size,
        )
        key = (
            mode,
            unified,
            _tensor_cache_token(query_starts_source),
            _tensor_cache_token(seq_lens_source),
            _tensor_cache_token(block_table),
            sequence_count,
            num_actual,
            max_seqlen_q,
            max_seq_len,
            kernel_max_seqlen_k,
            page_count,
        )
        owner_id = id(cache_owner)
        if (
            self._rocm_paged_metadata_key == key
            and owner_id not in self._rocm_paged_metadata_owners
            and self._rocm_paged_metadata_value is not None
        ):
            self._rocm_paged_metadata_owners.add(owner_id)
            return self._rocm_paged_metadata_value, True

        query_start_loc = query_starts_source.to(device=block_table.device, dtype=torch.int32)
        if not query_start_loc.is_contiguous():
            query_start_loc = query_start_loc.contiguous()
        seq_lens = seq_lens_source.to(device=block_table.device, dtype=torch.int32)
        if not seq_lens.is_contiguous():
            seq_lens = seq_lens.contiguous()
        # Graph metadata is request-level while packed Q is query-token-level.
        # Expand decode page rows to query rows without materializing KV.
        seq_of_token = None
        if unified:
            query_start_loc = query_start_loc[: sequence_count + 1]
            seq_lens = seq_lens[:sequence_count]
            active_rows = seq_lens > 0
            seqused_k = seq_lens
            pages = block_table[:sequence_count, :page_count]
            if mode == "decode" and num_actual == sequence_count:
                seq_of_token = torch.arange(
                    num_actual, dtype=torch.int32, device=block_table.device
                )
            else:
                tokens = torch.arange(num_actual, dtype=torch.int32, device=block_table.device)
                seq_of_token = torch.searchsorted(query_start_loc[1:], tokens, right=True).to(
                    torch.int32
                )
        elif mode == "decode":
            query_starts = query_start_loc[: sequence_count + 1]
            query_ends = query_starts[1:]
            query_indices = torch.arange(num_actual, dtype=torch.int32, device=block_table.device)
            request_indices = torch.searchsorted(query_ends, query_indices, right=True).to(
                dtype=torch.long
            )
            request_indices = request_indices.clamp_max(sequence_count - 1)
            request_query_ends = query_ends.index_select(0, request_indices)
            request_seq_lens = seq_lens.index_select(0, request_indices)
            active_queries = query_indices < query_starts[-1]
            seqused_k = request_seq_lens - (request_query_ends - query_indices) + 1
            seqused_k = torch.where(active_queries, seqused_k, torch.ones_like(seqused_k))
            pages = block_table.index_select(0, request_indices)[:, :page_count]
            query_start_loc = torch.arange(
                num_actual + 1, dtype=torch.int32, device=block_table.device
            )
            sequence_count = num_actual
            max_seqlen_q = 1
            active_rows = active_queries & (request_seq_lens > 0)
        else:
            query_start_loc = query_start_loc[: sequence_count + 1]
            seq_lens = seq_lens[:sequence_count]
            active_rows = seq_lens > 0
            seqused_k = seq_lens
            pages = block_table[:sequence_count, :page_count]
        if not pages.is_contiguous():
            pages = pages.contiguous()
        if mode != "decode" and not unified:
            active_rows = seqused_k > 0
        if configured_kv_limit is not None:
            # Zero-length rows are legal vLLM graph padding.  They must not
            # trip the bound assertion when a dynamic decode batch shrinks.
            torch._assert_async(
                torch.all((seq_lens >= 0) & (seq_lens <= kernel_max_seqlen_k)),
                "vLLM sequence length exceeds RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS",
            )
        pages = pages.to(dtype=torch.int32)
        # vLLM CUDA/HIP graphs retain padded request rows after a request
        # finishes.  Their sequence length is zero and their page-table row
        # is not guaranteed to contain a valid physical page.  AITER CK does
        # not accept either state; route inactive rows to page 0 and clamp
        # their logical length. vLLM ignores outputs for graph-padding rows.
        # CK looks up complete 128-token KV tiles before applying its logical
        # mask. Unused columns must be valid even for active request rows.
        # Reuse the row's first live page so masked loads see initialized KV.
        safe_page = torch.where(active_rows, pages[:, 0], torch.zeros_like(seqused_k))
        columns = torch.arange(page_count, dtype=torch.int32, device=pages.device)
        live_columns = active_rows[:, None] & (columns[None, :] * block_size < seqused_k[:, None])
        pages = torch.where(live_columns, pages, safe_page[:, None])
        tile_pages = max(1, 128 // block_size)
        guard_columns = (-page_count) % tile_pages
        if guard_columns:
            pages = torch.cat((pages, safe_page[:, None].expand(-1, guard_columns)), dim=1)
            page_count += guard_columns
        seq_lens_for_kernel = seqused_k.clamp_min(1)
        indptr_key = (
            block_table.device.type,
            block_table.device.index,
            sequence_count,
            page_count,
        )
        kv_indptr = self._rocm_kv_indptr_cache.get(indptr_key)
        if kv_indptr is None:
            kv_indptr = (
                torch.arange(
                    sequence_count + 1,
                    dtype=torch.int32,
                    device=block_table.device,
                )
                * page_count
            )
            self._rocm_kv_indptr_cache[indptr_key] = kv_indptr
        value = {
            "mode": mode,
            "sequence_count": sequence_count,
            "page_count": page_count,
            "pages": pages,
            "seqused_k": seq_lens_for_kernel,
            "cu_seqlens_q": query_start_loc,
            "kv_indptr": kv_indptr,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": kernel_max_seqlen_k,
            "configured_kv_limit": configured_kv_limit,
            "causal": causal,
            "seq_of_token": seq_of_token,
        }
        self._rocm_paged_metadata_key = key
        self._rocm_paged_metadata_owners = {owner_id}
        self._rocm_paged_metadata_value = value
        return value, False

    def _rocm_direct_paged(
        self,
        impl: Any,
        layer: Any,
        query: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: Any,
        runtime: Any,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        *,
        tp_world: int,
        num_actual: int,
    ) -> torch.Tensor | None:
        direct = getattr(runtime, "forward_paged_varlen_with_lse", None)
        if not callable(direct):
            return None
        metadata_result = self._rocm_direct_paged_metadata(
            attn_metadata,
            block_table=block_table,
            block_size=key_cache.size(1),
            num_actual=num_actual,
            cache_owner=layer,
            runtime_core=getattr(runtime, "_core", None),
        )
        if metadata_result is None:
            return None
        metadata, reused = metadata_result
        if not reused:
            self._validate_page_bounds_once(
                [{"pages": metadata["pages"]}],
                num_cache_blocks=key_cache.size(0),
            )
        query_ready = query.narrow(0, 0, num_actual)
        if not query_ready.is_contiguous():
            query_ready = query_ready.contiguous()
        output_heads = output.view(output.size(0), impl.num_heads, impl.head_size)
        output_ready = output_heads.narrow(0, 0, num_actual)
        result = direct(
            query_ready,
            key_cache,
            value_cache,
            page_table=metadata["pages"],
            seqused_k=metadata["seqused_k"],
            cu_seqlens_q=metadata["cu_seqlens_q"],
            kv_indptr=metadata["kv_indptr"],
            max_seqlen_q=metadata["max_seqlen_q"],
            max_seqlen_k=metadata["max_seqlen_k"],
            causal=metadata["causal"],
            scale=float(impl.scale),
            out=output_ready,
            return_lse=False,
            page_table_validated=True,
        )
        if result.out.data_ptr() != output_ready.data_ptr():
            output_ready.copy_(result.out)
        if num_actual < output.size(0):
            output[num_actual:].zero_()
        operator_provenance = _compact_attention_provenance(result.provenance)
        projection_collective_backend = "none"
        if tp_world > 1:
            projection_collective_backend = "unbound"
            if self._projection_collective_backend is not None:
                projection_collective_backend = self._projection_collective_backend() or "unbound"
        self._record_phase_provenance(
            metadata["mode"],
            {
                "framework_layout": "vllm_paged_kv",
                "materialization": "direct_vllm_paged_kv_to_aiter_batch_prefill_ck",
                "dense_kv_materialized": False,
                "tp_world_size": tp_world,
                "runtime_platform": "rocm",
                "triton_used": True,
                "attention_phase": metadata["mode"],
                "sequence_count": metadata["sequence_count"],
                "query_token_count": num_actual,
                "max_seqlen_k": metadata["max_seqlen_k"],
                "configured_kv_limit": metadata["configured_kv_limit"],
                "launch_group_count": 1,
                "metadata_source": "vllm_gpu_sequence_level",
                "metadata_reused_across_layers": reused,
                "deterministic_projection": _strict_attention_projection_provenance("rocm"),
                "deterministic_all_reduce_backend": projection_collective_backend,
                "direct_output_buffer": True,
                "operator": operator_provenance,
            },
        )
        return output

    def _materialization_groups(
        self,
        attn_metadata: Any,
        *,
        query: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        num_actual: int,
        cache_owner: Any | None = None,
        include_host_lengths: bool = False,
        page_bounds_epoch_factory: Callable[[], object] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build one paged launch from vLLM's graph-replayable GPU metadata."""

        query_starts_source = self._metadata_tensor(
            attn_metadata, "query_start_loc", "query_start_loc_cpu"
        )
        seq_lens_source = self._metadata_tensor(attn_metadata, "seq_lens", "seq_lens_cpu")
        if num_actual == 0:
            return [], {
                "row_count": 0,
                "query_position_range": [None, None],
                "kv_token_range": [None, None],
                "launch_group_count": 0,
                "metadata_source": "vllm_gpu",
            }

        max_seq_len = int(getattr(attn_metadata, "max_seq_len", block_table.size(1) * block_size))
        page_count = min(block_table.size(1), (max_seq_len + block_size - 1) // block_size)
        cache_key = (
            _tensor_cache_token(query_starts_source),
            _tensor_cache_token(seq_lens_source),
            _tensor_cache_token(block_table),
            query.device.type,
            query.device.index,
            num_actual,
            block_size,
            page_count,
            include_host_lengths,
            (
                None
                if page_bounds_epoch_factory is None
                else id(getattr(page_bounds_epoch_factory, "__self__", page_bounds_epoch_factory))
            ),
        )
        owner_id = id(cache_owner) if cache_owner is not None else None
        # Resolve the cross-layer cache before scheduling any GPU metadata work.
        # Repeating an owner still forces the first layer of the next forward to miss.
        if (
            owner_id is not None
            and self._metadata_cache_key == cache_key
            and owner_id not in self._metadata_cache_owners
            and self._metadata_cache_value is not None
        ):
            self._metadata_cache_owners.add(owner_id)
            groups, cached_summary = self._metadata_cache_value
            return groups, {**cached_summary, "metadata_reused_across_layers": True}

        query_starts = query_starts_source.to(device=query.device, dtype=torch.int32)
        seq_lens = seq_lens_source.to(device=query.device, dtype=torch.int32)
        query_indices = torch.arange(num_actual, dtype=torch.int32, device=query.device)
        query_ends = query_starts[1:]
        request_indices = torch.searchsorted(query_ends, query_indices, right=True).to(
            dtype=torch.long
        )
        active_queries = query_indices < query_starts[-1]
        request_indices = request_indices.clamp_max(seq_lens.numel() - 1)
        request_query_ends = query_ends.index_select(0, request_indices)
        request_seq_lens = seq_lens.index_select(0, request_indices)
        seqused_k = request_seq_lens - (request_query_ends - query_indices) + 1
        seqused_k = torch.where(
            active_queries,
            seqused_k,
            torch.ones_like(seqused_k),
        )
        cached_lengths = (
            tuple(int(value) for value in seqused_k.tolist()) if include_host_lengths else None
        )

        pages = (
            block_table.index_select(0, request_indices)[:, :page_count]
            .to(dtype=torch.int32)
            .contiguous()
        )
        cu_seqlens_q = torch.arange(num_actual + 1, dtype=torch.int32, device=query.device)
        kv_indptr = cu_seqlens_q * page_count
        groups = [
            {
                "page_count": page_count,
                "pages": pages,
                "query_indices": query_indices,
                "query_start": 0,
                "query_count": num_actual,
                "query_contiguous": True,
                "seqused_k": seqused_k,
                "cached_lengths": cached_lengths,
                "page_bounds_epoch": (
                    None if page_bounds_epoch_factory is None else page_bounds_epoch_factory()
                ),
                "cu_seqlens_q": cu_seqlens_q,
                "kv_indptr": kv_indptr,
            }
        ]
        summary = {
            "row_count": num_actual,
            "query_position_range": "device_dynamic",
            "kv_token_range": "device_dynamic",
            "launch_group_count": 1,
            "metadata_source": "vllm_gpu",
            "metadata_reused_across_layers": False,
        }
        if owner_id is not None:
            # The first repeated layer marks the next model forward (or graph
            # capture), so dynamic metadata is re-derived instead of frozen.
            self._metadata_cache_key = cache_key
            self._metadata_cache_owners = {owner_id}
            self._metadata_cache_value = (groups, summary)
        return groups, summary

    @staticmethod
    def _validate_page_bounds_once(
        groups: list[dict[str, Any]],
        *,
        num_cache_blocks: int,
    ) -> None:
        for group in groups:
            pages = group["pages"]
            bounds_ok = torch.all((pages >= 0) & (pages < num_cache_blocks))
            if pages.is_cuda:
                torch._assert_async(bounds_ok, "page_table entries are outside the KV cache")
            elif not bool(bounds_ok.item()):
                raise ValueError("page_table entries are outside the KV cache")

    def __call__(
        self,
        impl: Any,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise RuntimeError("strict vLLM Attention does not support quantized output")
        if attn_metadata is None:
            if output is None:
                raise RuntimeError("vLLM profiling Attention requires an output buffer")
            return output.zero_()
        if query.ndim != 3:
            raise RuntimeError("vLLM query must use [tokens, heads, head_dim]")
        runtime_platform = _require_attention_accelerator(query)

        num_actual = int(getattr(attn_metadata, "num_actual_tokens", query.size(0)))
        if num_actual < 0 or num_actual > query.size(0):
            raise RuntimeError("vLLM num_actual_tokens is outside the query buffer")
        if output is None:
            output = torch.empty(
                (query.size(0), impl.num_heads * impl.head_size),
                dtype=query.dtype,
                device=query.device,
            )
        if output.size(0) < num_actual:
            raise RuntimeError("vLLM output buffer is smaller than num_actual_tokens")
        output_heads = output.view(output.size(0), impl.num_heads, impl.head_size)
        if self._tp_coordinates is None:
            self._tp_coordinates = _vllm_tp_coordinates()
        tp_world, tp_rank, tp_group = self._tp_coordinates
        operator = self._handle.get(
            query,
            topology={
                "world_size": tp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": 1,
            },
        )
        runtime = operator.bind_accelerator_runtime(query)
        page_bounds_epoch_factory: Callable[[], object] | None = None
        if runtime_platform == "rocm":
            candidate_factory = getattr(runtime, "new_page_bounds_epoch", None)
            if callable(candidate_factory):
                page_bounds_epoch_factory = candidate_factory
        block_table = self._metadata_tensor(attn_metadata, "block_table", "block_table_tensor")
        key_cache, value_cache = _vllm_kv_cache_views(
            kv_cache,
            head_size=int(impl.head_size),
            num_kv_heads=int(impl.num_kv_heads),
            platform=runtime_platform,
        )
        if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
            raise RuntimeError("strict vLLM Attention requires an unquantized KV cache")
        if runtime_platform == "rocm":
            direct_output = self._rocm_direct_paged(
                impl,
                layer,
                query,
                output,
                attn_metadata,
                runtime,
                key_cache,
                value_cache,
                block_table,
                tp_world=tp_world,
                num_actual=num_actual,
            )
            if direct_output is not None:
                return direct_output
            prefill_output = self._rocm_dense_prefill(
                impl,
                query,
                key,
                value,
                output,
                attn_metadata,
                runtime,
                tp_rank=tp_rank,
                tp_world=tp_world,
                num_actual=num_actual,
            )
            if prefill_output is not None:
                return prefill_output
        block_size = key_cache.size(1)
        groups, metadata_summary = self._materialization_groups(
            attn_metadata,
            query=query,
            block_table=block_table,
            block_size=block_size,
            num_actual=num_actual,
            cache_owner=layer,
            include_host_lengths=runtime_platform == "rocm",
            page_bounds_epoch_factory=page_bounds_epoch_factory,
        )
        page_table_validated = False
        if runtime_platform == "rocm":
            if not metadata_summary["metadata_reused_across_layers"]:
                self._validate_page_bounds_once(
                    groups,
                    num_cache_blocks=key_cache.size(0),
                )
            page_table_validated = True
        last_operator_provenance: dict[str, Any] = {}
        next_query_row = 0
        for group in groups:
            if not group["query_contiguous"] or group["query_start"] != next_query_row:
                break
            next_query_row += int(group["query_count"])
        direct_output_buffer = bool(groups) and next_query_row == num_actual
        if not direct_output_buffer:
            output.zero_()
        elif num_actual < output.size(0):
            output[num_actual:].zero_()
        for group in groups:
            page_count = int(group["page_count"])
            pages = group["pages"]
            query_indices = group["query_indices"]
            runtime_out = None
            output_group = None
            if group["query_contiguous"]:
                query_start = int(group["query_start"])
                query_count = int(group["query_count"])
                q_ready = query.narrow(0, query_start, query_count).unsqueeze(2)
                output_group = output_heads.narrow(0, query_start, query_count)
                runtime_out = output_group.unsqueeze(2)
            else:
                q_ready = query.index_select(0, query_indices).unsqueeze(2).contiguous()
            runtime_kwargs = {
                "page_table": pages,
                "seqused_k": group["seqused_k"],
                "max_seqlen_k": page_count * block_size,
                "scale": float(impl.scale),
                "out": runtime_out,
            }
            if runtime_platform == "rocm":
                runtime_kwargs["cached_lengths"] = group["cached_lengths"]
                if group["page_bounds_epoch"] is not None:
                    runtime_kwargs["page_bounds_epoch"] = group["page_bounds_epoch"]
                runtime_kwargs["return_lse"] = False
                runtime_kwargs["page_table_validated"] = page_table_validated
                runtime_kwargs["cu_seqlens_q"] = group["cu_seqlens_q"]
                runtime_kwargs["kv_indptr"] = group["kv_indptr"]
            result = runtime.forward_paged_with_lse(
                q_ready,
                key_cache,
                value_cache,
                **runtime_kwargs,
            )
            result_output = result.out.squeeze(2)
            if group["query_contiguous"]:
                assert output_group is not None
                if result_output.data_ptr() != output_group.data_ptr():
                    output_group.copy_(result_output)
            else:
                output_heads.index_copy_(0, query_indices, result.out.squeeze(2))
            last_operator_provenance = _compact_attention_provenance(result.provenance)
        projection_collective_backend = "none"
        if runtime_platform == "rocm" and tp_world > 1:
            projection_collective_backend = "unbound"
            if self._projection_collective_backend is not None:
                projection_collective_backend = self._projection_collective_backend() or "unbound"
        phase = "decode" if int(getattr(attn_metadata, "num_decodes", 0)) > 0 else "prefill"
        self._record_phase_provenance(
            phase,
            {
                "framework_layout": "vllm_paged_kv",
                "materialization": (
                    "direct_vllm_paged_kv_to_aiter_batch_prefill_ck"
                    if runtime_platform == "rocm"
                    else "direct_paged_fa4"
                ),
                "dense_kv_materialized": False,
                "tp_world_size": tp_world,
                "tp_group_bound": tp_group is not None,
                "runtime_platform": runtime_platform,
                "triton_used": runtime_platform == "rocm",
                "deterministic_projection": _strict_attention_projection_provenance(
                    runtime_platform
                ),
                "deterministic_all_reduce_backend": projection_collective_backend,
                "direct_output_buffer": direct_output_buffer,
                **metadata_summary,
                "operator": last_operator_provenance,
            },
        )
        return output


class VllmFFNOperator:
    backend_id = FFN_BACKEND_ID

    def __init__(self, handle: SemanticOperatorHandle | None = None) -> None:
        self._handle = handle or SemanticOperatorHandle(
            target="rollout", semantic_op="ffn", backend_id=self.backend_id
        )
        self._last_provenance: dict[str, Any] = {}
        self._packed_inference_binding: tuple[int, int, str] | None = None
        self._tp_coordinates: tuple[int, int, Any] | None = None

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "interface": "vllm.qwen3.mlp.forward",
            "operator": self.backend_id,
            "fallback": False,
            "semantic_instance": self._handle.provenance,
            "execution": dict(self._last_provenance),
        }

    def bind_packed_inference(self, module: Any) -> tuple[int, int]:
        """Bind vLLM's TP group before torch.compile captures the model."""

        fused_gate_up = _weight(module.gate_up_proj, "gate_up_proj")
        down = _weight(module.down_proj, "down_proj")
        tp_world, tp_rank, tp_group = _vllm_tp_coordinates()
        self._tp_coordinates = (tp_world, tp_rank, tp_group)
        operator = self._handle.get(
            fused_gate_up,
            topology={
                "world_size": tp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": 1,
            },
        )
        prepare = getattr(operator, "prepare_packed_inference", None)
        if not callable(prepare):
            raise RuntimeError("strict FFN backend lacks packed inference preparation")
        collective_handle, bound_tp_world = prepare(
            fused_gate_up,
            down,
            tp_group=tp_group,
        )
        backend_id = "deterministic_all_reduce.ipc_localized_fixed_tree.v1"
        resolve_backend = getattr(operator, "packed_inference_backend_id", None)
        if callable(resolve_backend) and collective_handle:
            backend_id = str(resolve_backend(collective_handle))
        self._packed_inference_binding = (
            collective_handle,
            bound_tp_world,
            backend_id,
        )
        self._set_runtime_provenance(tp_world, backend_id)
        return collective_handle, bound_tp_world

    def _set_runtime_provenance(
        self, tp_world: int, deterministic_all_reduce_backend: str = "unbound"
    ) -> None:
        runtime_platform = "rocm" if torch.version.hip is not None else "cuda"
        self._last_provenance = {
            "framework_layout": "vllm_tensor_parallel",
            "tp_world_size": tp_world,
            "runtime_platform": runtime_platform,
            "actual_backend": f"rlkernel.{runtime_platform}.det_gemm_swiglu",
            "gemm_backend": det_gemm_backend_id(),
            "fallback": False,
            "gate_up_projection": "packed_single_launch",
            "deterministic_all_reduce_backend": deterministic_all_reduce_backend,
            "triton_used": runtime_platform == "rocm",
        }

    def __call__(self, module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_nvidia_cuda(hidden_states, "FFN")
        fused_gate_up = _weight(module.gate_up_proj, "gate_up_proj").contiguous()
        down = _weight(module.down_proj, "down_proj").contiguous()
        if self._tp_coordinates is None:
            self._tp_coordinates = _vllm_tp_coordinates()
        tp_world, _tp_rank, tp_group = self._tp_coordinates
        operator = self._handle.get(
            hidden_states,
            topology={
                "world_size": tp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": 1,
            },
        )
        bound_backend = "unbound"
        if self._packed_inference_binding is not None:
            bound_backend = self._packed_inference_binding[2]
        self._set_runtime_provenance(tp_world, bound_backend)
        if self._packed_inference_binding is not None:
            collective_handle, bound_tp_world, _ = self._packed_inference_binding
            if bound_tp_world != tp_world:
                raise RuntimeError("packed rollout FFN TP topology changed after binding")
            output = operator.packed_inference(
                hidden_states,
                fused_gate_up,
                down,
                collective_handle=collective_handle,
                tp_world_size=bound_tp_world,
            )
        else:
            if torch._dynamo.is_compiling():
                raise RuntimeError("packed rollout FFN was not bound before torch.compile capture")
            gate, up = _split_gate_up(fused_gate_up, "gate_up_proj")
            output = operator(
                hidden_states,
                gate,
                up,
                down,
                tp_group=tp_group,
                sequence_parallel=False,
                deterministic=True,
            )
        return output


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


def _full_vocab_contract(logits: torch.Tensor) -> LogprobContract:
    dtype = {
        torch.bfloat16: LogprobDType.BF16,
        torch.float16: LogprobDType.FP16,
        torch.float32: LogprobDType.FP32,
    }.get(logits.dtype)
    if dtype is None:
        raise RuntimeError(f"unsupported vLLM logit dtype {logits.dtype}")
    tokens, vocab = logits.shape
    return LogprobContract(
        role=LogprobRole.INFER,
        dtype=dtype,
        mask=MaskSpec(num_tokens=tokens, active_mask=(True,) * tokens),
        sharding=LogprobShardingSpec(
            tp_rank=0,
            tp_world_size=1,
            vocab_shard_bounds=((0, vocab),),
            real_vocab_size=vocab,
            padded_vocab_size=vocab,
        ),
        reduction=LogprobReductionSpec(),
    )


def _diagnostic_vocab_contract(
    logits: torch.Tensor,
    *,
    real_vocab_size: int | None,
    padded_vocab_size: int | None,
) -> LogprobContract | None:
    if real_vocab_size is None or padded_vocab_size is None:
        return None
    if real_vocab_size <= 0 or padded_vocab_size <= 0:
        return None
    if real_vocab_size > padded_vocab_size:
        return None
    tokens, vocab = logits.shape
    if padded_vocab_size != vocab:
        return None
    if padded_vocab_size % DEFAULT_NUM_VOCAB_TILES:
        return None
    dtype = {
        torch.bfloat16: LogprobDType.BF16,
        torch.float16: LogprobDType.FP16,
        torch.float32: LogprobDType.FP32,
    }.get(logits.dtype)
    if dtype is None:
        return None
    return LogprobContract(
        role=LogprobRole.INFER,
        dtype=dtype,
        mask=MaskSpec(num_tokens=tokens, active_mask=(True,) * tokens),
        sharding=LogprobShardingSpec(
            tp_rank=0,
            tp_world_size=1,
            vocab_shard_bounds=((0, padded_vocab_size),),
            real_vocab_size=real_vocab_size,
            padded_vocab_size=padded_vocab_size,
        ),
        reduction=LogprobReductionSpec(),
    )


class VllmLogpOperator:
    """Keep sampling, then replace sampled-token logp with the selected contract."""

    def __init__(
        self,
        native_forward: Any,
        *,
        worker_sampler: bool = False,
        strict_linear_logp: bool = False,
    ) -> None:
        self._native_forward = native_forward
        self._worker_sampler = worker_sampler
        self._strict_linear_logp = strict_linear_logp
        self._linear_logp = LinearLogpWrapper() if strict_linear_logp else None
        self._last_provenance: dict[str, Any] = {}

    @property
    def backend_id(self) -> str:
        if self._strict_linear_logp:
            assert self._linear_logp is not None
            return self._linear_logp.backend_id
        return LOGP_BACKEND_ID

    @property
    def provenance(self) -> Mapping[str, Any]:
        return dict(self._last_provenance)

    def _replace_sampled_value(
        self,
        result: Any,
        *,
        token_ids: torch.Tensor,
        selected: torch.Tensor,
        provenance: Mapping[str, Any],
    ) -> Any:
        logprobs_tensors = getattr(result, "logprobs_tensors", None)
        if logprobs_tensors is None:
            raise RuntimeError(
                "strict vLLM Logp requires sampled-token logprob tensors; "
                "native sampling returned none"
            )
        ids = logprobs_tensors.logprob_token_ids
        values = logprobs_tensors.logprobs.clone()
        if selected.numel() != token_ids.numel():
            raise RuntimeError(
                "strict vLLM logp selected output is not aligned with sampled tokens"
            )
        matches = ids == token_ids.unsqueeze(1)
        every_sample_present = matches.any(dim=1).all()
        if every_sample_present.is_cuda:
            torch._assert_async(
                every_sample_present,
                "vLLM logprob result does not contain every sampled token",
            )
        elif not bool(every_sample_present):
            raise RuntimeError("vLLM logprob result does not contain every sampled token")
        # vLLM prepends the sampled token and then appends top-k tokens. When
        # the sample is also in top-k, its API conversion keeps the last
        # duplicate, so every matching column must carry the strict value.
        values = torch.where(matches, selected.unsqueeze(1), values)
        self._last_provenance = {
            **dict(provenance),
            "interface": "vllm.sampler.selected_logprob",
            "operator": self.backend_id,
            "actual_backend": self.backend_id,
            "fallback": False,
            "sampled_token_ids": _tensor_metadata(token_ids),
            "logprobs_shape": list(values.shape),
            "logprob_token_ids_shape": list(ids.shape),
            "strict_selected_logp": _tensor_metadata(selected),
            "native_reference_compared": False,
        }
        if hasattr(logprobs_tensors, "_replace"):
            updated_tensors = logprobs_tensors._replace(logprobs=values)
        else:
            updated_tensors = replace(logprobs_tensors, logprobs=values)
        return replace(result, logprobs_tensors=updated_tensors)

    def __call__(
        self,
        sampler: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        predict_bonus_token: bool = False,
        logprobs_mode_override: Any = None,
    ) -> Any:
        # Native sampling owns logits; strict selected-logp preserves its raw
        # local shard before the sampler applies in-place transformations.
        source_logits = logits
        _require_nvidia_cuda(source_logits, "Logp")
        context = None
        local_logits = None
        if self._strict_linear_logp:
            context = take_rollout_linear_logp_context()
            if source_logits.ndim != 2:
                raise RuntimeError("vLLM sampler logits must be [tokens, vocab]")
            if context.hidden.size(0) != source_logits.size(0):
                raise RuntimeError(
                    "strict rollout linear_logp hidden/logits row mismatch: "
                    f"{context.hidden.size(0)} != {source_logits.size(0)}"
                )
            if source_logits.size(1) < context.real_vocab_size:
                raise RuntimeError(
                    "strict rollout source logits do not cover the real vocabulary: "
                    f"{source_logits.size(1)} < {context.real_vocab_size}"
                )
            local_vocab = int(context.lm_head_weight.size(0))
            available = max(
                0,
                min(
                    local_vocab,
                    source_logits.size(1) - context.vocab_start_index,
                ),
            )
            # Preserve raw model logits before vLLM's sampler transforms its
            # input in place (temperature, penalties, and masking).
            if available == local_vocab:
                # Rank 0 normally has a complete local shard. Narrowing first
                # avoids a fill kernel followed by a second device copy.
                local_logits = source_logits.narrow(
                    1,
                    context.vocab_start_index,
                    local_vocab,
                ).contiguous()
            else:
                # The final TP shard may include padded rows absent from the
                # serving logits; retain the exact -inf padding contract.
                local_logits = source_logits.new_full(
                    (source_logits.size(0), local_vocab),
                    float("-inf"),
                )
                if available:
                    local_logits[:, :available].copy_(
                        source_logits.narrow(
                            1,
                            context.vocab_start_index,
                            available,
                        )
                    )
        if self._worker_sampler:
            result = self._native_forward(sampler, logits, sampling_metadata)
        else:
            result = self._native_forward(
                sampler,
                logits,
                sampling_metadata,
                predict_bonus_token=predict_bonus_token,
                logprobs_mode_override=logprobs_mode_override,
            )
        logprobs_tensors = getattr(result, "logprobs_tensors", None)
        if logprobs_tensors is None:
            raise RuntimeError(
                "strict vLLM Logp requires sampled-token logprob tensors; "
                "native sampling returned none"
            )
        token_ids = result.sampled_token_ids.reshape(-1).to(torch.long)

        if self._strict_linear_logp:
            assert context is not None and local_logits is not None
            if context.hidden.size(0) != token_ids.numel():
                raise RuntimeError(
                    "strict rollout linear_logp hidden/sample row mismatch: "
                    f"{context.hidden.size(0)} != {token_ids.numel()}"
                )
            assert self._linear_logp is not None
            selected = self._linear_logp.from_local_logits(
                local_logits,
                token_ids,
                tp_group=context.tp_group,
                vocab_start_index=context.vocab_start_index,
                global_vocab_size=context.global_vocab_size,
                real_vocab_size=context.real_vocab_size,
                temperature=float(os.getenv("RL_KERNEL_VLLM_TEMPERATURE", "1.0")),
                target="rollout",
                diagnostics_hidden=context.hidden,
                diagnostics_lm_head_weight=context.lm_head_weight,
            )
            strict_provenance = self._linear_logp.provenance
            expected_entrypoint = (
                "rocm_vocab_parallel_logp_from_local_logits_tp"
                if torch.version.hip is not None
                else "sm90_deterministic_logp_from_local_logits_tp"
            )
            if (
                strict_provenance.get("deterministic_linear_logp") is not True
                or strict_provenance.get("actual_backend") != self._linear_logp.backend_id
                or strict_provenance.get("strict_entrypoint") != expected_entrypoint
            ):
                raise RuntimeError(
                    "strict vLLM rollout linear_logp did not execute the deterministic "
                    "linear-logp entry point"
                )
            provenance = {
                **dict(strict_provenance),
                "runtime_platform": _device_name(local_logits),
                "triton_used": torch.version.hip is not None,
                "execution": {
                    "role": "vllm_rollout_linear_logprob",
                    "strict_backend": True,
                    "sampling_logits_source": "rlkernel_det_gemm_vllm",
                    "logits_materialized": True,
                    "padded_lm_head_alignment": True,
                    "duplicate_lm_head_gemm": False,
                },
                "source_logits_shape": list(source_logits.shape),
                "source_logits_dtype": _dtype_name(source_logits),
            }
            return self._replace_sampled_value(
                result,
                token_ids=token_ids,
                selected=selected,
                provenance=provenance,
            )

        if source_logits.ndim != 2:
            raise RuntimeError("vLLM sampler logits must be [tokens, vocab]")
        if source_logits.size(1) % DEFAULT_NUM_VOCAB_TILES:
            raise RuntimeError(
                "strict vLLM logp requires the padded vocabulary to be divisible by "
                f"{DEFAULT_NUM_VOCAB_TILES}"
            )
        contract = _full_vocab_contract(source_logits)
        from rl_engine.kernels.registry import kernel_registry

        dispatch = kernel_registry.get_logprob_op(contract, requested_backend=self.backend_id)
        if (
            dispatch.provenance["actual_backend"] != self.backend_id
            or dispatch.provenance["fallback"]
        ):
            raise RuntimeError("strict vLLM Logp dispatch changed backend")
        selected, _lse = dispatch.op(
            source_logits,
            token_ids,
            contract=contract,
            num_vocab_tiles=DEFAULT_NUM_VOCAB_TILES,
            deterministic=True,
        )
        real_vocab_env = _optional_env_int("RL_KERNEL_VLLM_REAL_VOCAB_SIZE")
        padded_vocab_env = _optional_env_int("RL_KERNEL_VLLM_PADDED_VOCAB_SIZE")
        masked_vocab_diagnostic: dict[str, Any] = {
            "env_real_vocab_size": real_vocab_env,
            "env_padded_vocab_size": padded_vocab_env,
            "status": (
                "not_requested"
                if real_vocab_env is None and padded_vocab_env is None
                else "skipped"
            ),
        }
        diagnostic_contract = _diagnostic_vocab_contract(
            source_logits,
            real_vocab_size=real_vocab_env,
            padded_vocab_size=padded_vocab_env,
        )
        if diagnostic_contract is not None:
            try:
                diagnostic_dispatch = kernel_registry.get_logprob_op(
                    diagnostic_contract, requested_backend=self.backend_id
                )
                masked_selected, _masked_lse = diagnostic_dispatch.op(
                    source_logits,
                    token_ids,
                    contract=diagnostic_contract,
                    num_vocab_tiles=DEFAULT_NUM_VOCAB_TILES,
                    deterministic=True,
                )
                masked_vocab_diagnostic.update(
                    {
                        "status": "computed_not_applied",
                        "contract_real_vocab_size": diagnostic_contract.sharding.real_vocab_size,
                        "contract_padded_vocab_size": (
                            diagnostic_contract.sharding.padded_vocab_size
                        ),
                        "selected_stats": _tensor_debug_stats(masked_selected),
                        "diff_vs_native_selected": _diff_debug_stats(
                            masked_selected,
                            logprobs_tensors.logprobs[
                                torch.arange(
                                    logprobs_tensors.logprobs.size(0),
                                    device=logprobs_tensors.logprobs.device,
                                ),
                                (logprobs_tensors.logprob_token_ids == token_ids.unsqueeze(1))
                                .to(torch.int64)
                                .argmax(dim=1),
                            ],
                        ),
                        "diff_vs_current_rlkernel_selected": _diff_debug_stats(
                            masked_selected, selected
                        ),
                    }
                )
            except Exception as exc:  # pragma: no cover - diagnostic only
                masked_vocab_diagnostic.update(
                    {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                )
        elif real_vocab_env is not None or padded_vocab_env is not None:
            masked_vocab_diagnostic.update(
                {
                    "source_logits_vocab_width": int(source_logits.size(1)),
                    "reason": (
                        "env metadata must be positive, real<=padded, and "
                        "padded_vocab_size must match vLLM logits width"
                    ),
                }
            )

        provenance = {
            **dict(dispatch.provenance),
            "runtime_platform": _device_name(source_logits),
            "triton_used": torch.version.hip is not None,
            "source_logits_shape": list(source_logits.shape),
            "source_logits_dtype": _dtype_name(source_logits),
            "contract_real_vocab_size": contract.sharding.real_vocab_size,
            "contract_padded_vocab_size": contract.sharding.padded_vocab_size,
            "masked_vocab_diagnostic": masked_vocab_diagnostic,
        }
        return self._replace_sampled_value(
            result,
            token_ids=token_ids,
            selected=selected,
            provenance=provenance,
        )


__all__ = [
    "MegatronAttentionOperator",
    "MegatronFFNOperator",
    "MegatronLogpOperator",
    "SemanticOperatorHandle",
    "VllmAttentionOperator",
    "VllmFFNOperator",
    "VllmLogpOperator",
]
