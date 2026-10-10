# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tensor-layout adapters from framework boundaries to semantic operators.

This module owns no numerical kernel selection. Attention and FFN instances
are resolved by :class:`OperatorBridge`; Logp uses the existing contract-aware
``KernelRegistry`` dispatch shared with the Vime provider.
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any, Mapping, cast

import torch

from rl_engine.contracts.operators.attention import (
    AttentionContract,
    AttentionDType,
    AttentionMode,
    AttentionRole,
)
from rl_engine.contracts.operators.attention import ReductionSpec as AttentionReductionSpec
from rl_engine.contracts.operators.attention import ShardingSpec as AttentionShardingSpec
from rl_engine.contracts.operators.attention import SplitKVSpec
from rl_engine.ops.attention.projection import ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID
from rl_engine.ops.gemm.det_gemm import DetGemmOp, det_gemm_backend_id
from rl_engine.runtime.operators import OperatorBridge, OperatorOverride
from rl_engine.runtime.semantic_registry import OperatorRequirements

ATTENTION_BACKEND_ID = "rlkernel.attention.deterministic.v1"


FFN_BACKEND_ID = "rlkernel.ffn.qwen3.deterministic.v1"


def _device_name(tensor: torch.Tensor) -> str:
    if tensor.device.type == "cuda" and torch.version.hip is not None:
        return "rocm"
    return tensor.device.type


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


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


def _split_gate_up(weight: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.size(0) % 2:
        raise RuntimeError(f"{name}.weight first dimension must contain equal gate/up shards")
    gate, up = weight.chunk(2, dim=0)
    return gate.contiguous(), up.contiguous()


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


__all__ = [
    "MegatronAttentionOperator",
    "MegatronFFNOperator",
    "MegatronLogpOperator",
    "SemanticOperatorHandle",
    "VllmAttentionOperator",
    "VllmFFNOperator",
    "VllmLogpOperator",
]


_ENGINE_EXPORTS = {
    "MegatronFFNOperator": "rl_engine.integrations.engines.train.megatron.operators",
    "MegatronAttentionOperator": "rl_engine.integrations.engines.train.megatron.operators",
    "MegatronLogpOperator": "rl_engine.integrations.engines.train.megatron.operators",
    "VllmLogpOperator": "rl_engine.integrations.engines.rollout.vllm.operators",
    "VllmAttentionOperator": "rl_engine.integrations.engines.rollout.vllm.operators",
    "VllmFFNOperator": "rl_engine.integrations.engines.rollout.vllm.operators",
}


def __getattr__(name):
    if name not in _ENGINE_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    return getattr(import_module(_ENGINE_EXPORTS[name]), name)


if TYPE_CHECKING:
    from rl_engine.integrations.engines.rollout.vllm.operators import (
        VllmAttentionOperator,
        VllmFFNOperator,
        VllmLogpOperator,
    )
    from rl_engine.integrations.engines.train.megatron.operators import (
        MegatronAttentionOperator,
        MegatronFFNOperator,
        MegatronLogpOperator,
    )
