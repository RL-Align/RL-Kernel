# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""vLLM plugin hooks installed without editing vLLM source files."""

from __future__ import annotations

import importlib
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch

from rl_engine.distributed.collectives import (
    DETERMINISTIC_ALL_REDUCE_OP,
    collective_for_group,
    deterministic_all_reduce_inplace,
    deterministic_all_reduce_staged,
    deterministic_staging_reserve,
)
from rl_engine.integrations.ablation import (
    Implementation,
    IntegrationPlan,
    configure_integration_environment,
    integration_plan_from_environment,
)
from rl_engine.integrations.framework_operators import (
    VllmAttentionOperator,
    VllmFFNOperator,
    VllmLogpOperator,
    _strict_attention_projection_op,
)
from rl_engine.integrations.linear_logp import (
    clear_rollout_linear_logp_context,
    publish_rollout_linear_logp_context,
)
from rl_engine.integrations.state import get_active_integration, set_active_integration
from rl_engine.integrations.vllm import VllmIntegration
from rl_engine.kernels.ops.pytorch.ffn.ffn import register_packed_inference_observer
from rl_engine.kernels.ops.pytorch.norm.rms_norm import strict_add_rms_norm, strict_rms_norm

_PATCH_MARKER = "__rl_kernel_original_forward__"
_STRICT_MODEL_PATCH_MARKER = "__rl_kernel_original_strict_model_init__"
_STRICT_PROJECTION_MARKER = "__rl_kernel_strict_attention_projection__"
_STRICT_FFN_INIT_MARKER = "__rl_kernel_original_strict_ffn_init__"
_STRICT_RMS_NORM_INIT_MARKER = "__rl_kernel_original_strict_rms_norm_init__"
_STRICT_ATTENTION_RMS_NORM_MARKER = "__rl_kernel_strict_attention_rms_norm__"
_STRICT_ROTARY_INIT_MARKER = "__rl_kernel_original_strict_rotary_init__"
_STRICT_ROCM_ROPE_PATCH_MARKER = "__rl_kernel_original_strict_rocm_rope_forward__"
_STRICT_LM_HEAD_LINEAR_PATCH_MARKER = "__rl_kernel_original_lm_head_linear_apply__"
_STRICT_LM_HEAD_PROCESS_PATCH_MARKER = "__rl_kernel_original_lm_head_process_weights__"
_STRICT_LM_HEAD_TIE_PATCH_MARKER = "__rl_kernel_original_lm_head_tie_weights__"
_STRICT_LM_HEAD_CACHE_BUFFER = "_rl_kernel_lm_head_weight_t"
_STRICT_LM_HEAD_CACHE_STATE = "_rl_kernel_lm_head_weight_cache_state"
_STRICT_LM_HEAD_CACHE_HOOK = "_rl_kernel_lm_head_weight_cache_state_dict_hook"
_STRICT_LM_HEAD_TIED = "_rl_kernel_lm_head_tied_weight"
_STRICT_O_PROJ_COLLECTIVE_MARKER = "__rl_kernel_o_proj_collective__"
_STRICT_O_PROJ_FUSED_ALL_REDUCE_MARKER = "__rl_kernel_o_proj_fused_all_reduce__"
_STRICT_O_PROJ_COMPILED_COLLECTIVE_SLOT = "__rl_kernel_o_proj_compiled_collective_slot__"
_STRICT_ROW_PARALLEL_PATCH_MARKER = "__rl_kernel_original_row_parallel_forward__"
_STRICT_DIRECT_STAGING_MARKER = "__rl_kernel_direct_staging_active__"
_STRICT_LAYER_DIAGNOSTIC_PATCH_MARKER = "__rl_kernel_original_layer_diagnostic_forward__"
_STRICT_WEIGHT_CACHE_REFRESH_MARKER = "__rl_kernel_original_finish_weight_update__"
_RLK_ATTENTION_BACKEND: type[Any] | None = None
_RLK_ATTENTION_IMPL: type[Any] | None = None
_RLK_ATTENTION_BUILDER: type[Any] | None = None
_RLK_O_PROJ_COLLECTIVE_BACKEND: str | None = None
_VLLM_LAYER_DIAGNOSTIC_BUFFER: dict[str, Any] | None = None
_VLLM_LAYER_DIAGNOSTIC_CALLS = 0
_VLLM_LAYER_DIAGNOSTIC_ACTIVE_LAYER: int | None = None
_ROCM_STATEFUL_GRAPH_SPLITTING_OPS = (
    DETERMINISTIC_ALL_REDUCE_OP,
    "rl_kernel::rocm_det_gemm_linear_all_reduce_inference",
    "rl_kernel::qwen3_ffn_packed_tp_inference_rocm",
)
_ROCM_FULL_GRAPH_CACHE_NAMESPACE = "rl_kernel_rocm_full_graph_v4"
_ROCM_GRAPH_ROUTE_ENVIRONMENT = (
    "RL_KERNEL_ATTENTION_CASE",
    "RL_KERNEL_FFN_CASE",
    "RL_KERNEL_LOGP_CASE",
    "RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS",
    "RL_KERNEL_ROCM_FIXED_PAGED_TILE",
)


@dataclass
class _LmHeadWeightCacheState:
    source: torch.Tensor
    weight_t: torch.Tensor
    source_data_ptr: int
    source_shape: tuple[int, ...]
    source_stride: tuple[int, ...]
    source_dtype: torch.dtype
    source_device: torch.device
    source_version: int | None
    cache_data_ptr: int
    cache_version: int | None
    generation: int
    valid: bool
    refresh_pending: bool


def _tracked_tensor_version(value: torch.Tensor) -> int | None:
    try:
        return int(value._version)
    except RuntimeError:
        # Tensors created in inference mode do not carry a version counter.
        return None


def _invalidate_lm_head_weight_cache(weight: Any) -> None:
    state = getattr(weight, _STRICT_LM_HEAD_CACHE_STATE, None)
    if isinstance(state, _LmHeadWeightCacheState):
        state.valid = False
        state.refresh_pending = False


def _mark_lm_head_weight_cache_refreshable(weight: Any) -> None:
    state = getattr(weight, _STRICT_LM_HEAD_CACHE_STATE, None)
    if isinstance(state, _LmHeadWeightCacheState):
        state.refresh_pending = True


def _keep_lm_head_cache_non_persistent(layer: Any, *_args: Any) -> None:
    buffers = getattr(layer, "_buffers", {})
    non_persistent = getattr(layer, "_non_persistent_buffers_set", None)
    if _STRICT_LM_HEAD_CACHE_BUFFER in buffers and isinstance(non_persistent, set):
        non_persistent.add(_STRICT_LM_HEAD_CACHE_BUFFER)


def _record_lm_head_weight_cache_refresh(
    state: _LmHeadWeightCacheState,
    weight: torch.Tensor,
) -> None:
    state.source_data_ptr = int(weight.data_ptr())
    state.source_shape = tuple(int(dim) for dim in weight.shape)
    state.source_stride = tuple(int(stride) for stride in weight.stride())
    state.source_dtype = weight.dtype
    state.source_device = weight.device
    state.source_version = _tracked_tensor_version(weight)
    state.cache_data_ptr = int(state.weight_t.data_ptr())
    state.cache_version = _tracked_tensor_version(state.weight_t)
    state.generation += 1
    state.valid = True
    state.refresh_pending = False


def _refresh_lm_head_weight_cache(
    layer: Any,
    prepare_weight: Any,
) -> _LmHeadWeightCacheState:
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError("strict ROCm LM-head cache requires a tensor weight")
    if bool(getattr(layer, _STRICT_LM_HEAD_TIED, False)):
        raise RuntimeError("strict ROCm LM-head cache does not support tied embeddings")

    state = getattr(layer, _STRICT_LM_HEAD_CACHE_STATE, None)
    if state is None:
        weight_t = prepare_weight(weight)
        register_buffer = getattr(layer, "register_buffer", None)
        if callable(register_buffer):
            register_buffer(
                _STRICT_LM_HEAD_CACHE_BUFFER,
                weight_t,
                persistent=False,
            )
        else:
            # Kept for lightweight integration adapters used outside nn.Module.
            setattr(layer, _STRICT_LM_HEAD_CACHE_BUFFER, weight_t)
        register_state_dict_pre_hook = getattr(layer, "register_state_dict_pre_hook", None)
        if callable(register_state_dict_pre_hook) and not hasattr(
            layer, _STRICT_LM_HEAD_CACHE_HOOK
        ):
            handle = register_state_dict_pre_hook(_keep_lm_head_cache_non_persistent)
            setattr(layer, _STRICT_LM_HEAD_CACHE_HOOK, handle)
        state = _LmHeadWeightCacheState(
            source=weight,
            weight_t=weight_t,
            source_data_ptr=0,
            source_shape=(),
            source_stride=(),
            source_dtype=weight.dtype,
            source_device=weight.device,
            source_version=None,
            cache_data_ptr=int(weight_t.data_ptr()),
            cache_version=None,
            generation=0,
            valid=False,
            refresh_pending=False,
        )
        setattr(layer, _STRICT_LM_HEAD_CACHE_STATE, state)
    else:
        if not isinstance(state, _LmHeadWeightCacheState):
            raise RuntimeError("strict ROCm LM-head cache state has an invalid type")
        if (
            state.source is not weight
            or getattr(layer, _STRICT_LM_HEAD_CACHE_BUFFER, None) is not state.weight_t
        ):
            # Checkpoint-format layerwise reload temporarily replaces every
            # Parameter (and may remove derived buffers) before post-load
            # processing, then copies the canonical weight back into the
            # original stable storage.  Defer the transpose until the first
            # forward observes that stable source again.  This also avoids
            # depending on vLLM copying a non-persistent derived buffer.
            state.valid = False
            state.refresh_pending = True
            return state
        state.valid = False
        state.refresh_pending = False
        cache_data_ptr = int(state.weight_t.data_ptr())
        refreshed = prepare_weight(weight, out=state.weight_t)
        if refreshed is not state.weight_t or int(refreshed.data_ptr()) != cache_data_ptr:
            raise RuntimeError("strict ROCm LM-head refresh replaced stable cache storage")
        _record_lm_head_weight_cache_refresh(state, weight)

    if state.generation == 0:
        _record_lm_head_weight_cache_refresh(state, weight)
    setattr(weight, _STRICT_LM_HEAD_CACHE_STATE, state)
    return state


def _validated_lm_head_weight_cache(
    layer: Any,
    prepare_weight: Any | None = None,
) -> torch.Tensor:
    weight = getattr(layer, "weight", None)
    if bool(getattr(layer, _STRICT_LM_HEAD_TIED, False)):
        raise RuntimeError("strict ROCm LM-head cache does not support tied embeddings")
    _keep_lm_head_cache_non_persistent(layer)
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError("strict ROCm LM-head cache was not prepared after model loading")
    state_value = getattr(layer, _STRICT_LM_HEAD_CACHE_STATE, None)
    if not isinstance(state_value, _LmHeadWeightCacheState):
        raise RuntimeError("strict ROCm LM-head cache was not prepared after model loading")
    state: _LmHeadWeightCacheState = state_value
    cached_weight = state.weight_t
    if not isinstance(cached_weight, torch.Tensor):
        raise RuntimeError("strict ROCm LM-head cache state has an invalid weight")
    if (
        state.source is not weight
        or getattr(weight, _STRICT_LM_HEAD_CACHE_STATE, None) is not state
    ):
        raise RuntimeError("strict ROCm LM-head cache is not bound to the active weight")
    if getattr(layer, _STRICT_LM_HEAD_CACHE_BUFFER, None) is not cached_weight:
        raise RuntimeError("strict ROCm LM-head cache buffer was replaced")

    current_source = (
        int(weight.data_ptr()),
        tuple(int(dim) for dim in weight.shape),
        tuple(int(stride) for stride in weight.stride()),
        weight.dtype,
        weight.device,
        _tracked_tensor_version(weight),
    )
    expected_source = (
        state.source_data_ptr,
        state.source_shape,
        state.source_stride,
        state.source_dtype,
        state.source_device,
        state.source_version,
    )
    if current_source[:-1] != expected_source[:-1]:
        state.valid = False
        state.refresh_pending = False
        raise RuntimeError("strict ROCm LM-head weight storage changed without a cache refresh")
    if not cached_weight.is_contiguous() or int(cached_weight.data_ptr()) != state.cache_data_ptr:
        state.valid = False
        state.refresh_pending = False
        raise RuntimeError("strict ROCm LM-head cache storage changed after refresh")
    if state.valid:
        if current_source[-1] != expected_source[-1]:
            state.valid = False
            state.refresh_pending = False
            raise RuntimeError("strict ROCm LM-head weight changed without a cache refresh")
        if _tracked_tensor_version(cached_weight) != state.cache_version:
            state.valid = False
            state.refresh_pending = False
            raise RuntimeError("strict ROCm LM-head cache bytes changed after refresh")
        return cached_weight

    if not state.refresh_pending or prepare_weight is None:
        raise RuntimeError("strict ROCm LM-head cache is invalid during weight update")
    cache_data_ptr = int(cached_weight.data_ptr())
    try:
        refreshed = prepare_weight(weight, out=cached_weight)
    except Exception:
        state.refresh_pending = False
        raise
    if refreshed is not cached_weight or int(refreshed.data_ptr()) != cache_data_ptr:
        state.refresh_pending = False
        raise RuntimeError("strict ROCm LM-head refresh replaced stable cache storage")
    _record_lm_head_weight_cache_refresh(state, weight)
    return cached_weight


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


def _patch_qwen3_layer_alignment_diagnostics() -> None:
    """Record one bounded semantic decoder trace per vLLM model forward."""

    if not _alignment_diagnostics_enabled():
        return
    from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3DecoderLayer

    from rl_engine.kernels.ops.rocm.attention.strict_runtime import StrictRocmAttentionRuntime

    if hasattr(Qwen3DecoderLayer, _STRICT_LAYER_DIAGNOSTIC_PATCH_MARKER):
        return
    original_init = Qwen3DecoderLayer.__init__
    original_forward = Qwen3DecoderLayer.forward
    original_attention_forward = Qwen3Attention.forward
    original_gather_paged_row = StrictRocmAttentionRuntime._gather_paged_row

    def init_wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
        original_init(instance, *args, **kwargs)
        config = args[0] if args else kwargs.get("config")
        prefix = kwargs.get("prefix", args[3] if len(args) > 3 else "")
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(prefix))
        if match is None:
            raise RuntimeError(f"cannot recover Qwen3 decoder layer from prefix {prefix!r}")
        instance._rl_kernel_layer_diagnostic_index = int(match.group(1))
        instance._rl_kernel_layer_diagnostic_count = int(config.num_hidden_layers)

    def forward_wrapped(
        instance: Any,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global _VLLM_LAYER_DIAGNOSTIC_BUFFER
        global _VLLM_LAYER_DIAGNOSTIC_CALLS
        global _VLLM_LAYER_DIAGNOSTIC_ACTIVE_LAYER

        layer = int(instance._rl_kernel_layer_diagnostic_index)
        layer_count = int(instance._rl_kernel_layer_diagnostic_count)
        max_rows = int(os.getenv("RL_KERNEL_ALIGNMENT_MAX_ROWS", "64"))
        if layer == 0:
            semantic_input = hidden_states if residual is None else hidden_states + residual
            if semantic_input.ndim != 2:
                raise RuntimeError("vLLM layer diagnostics require [T,H] hidden states")
            _VLLM_LAYER_DIAGNOSTIC_BUFFER = (
                {
                    "positions": positions.detach(),
                    "input": semantic_input.detach(),
                    "outputs": [],
                }
                if int(semantic_input.size(0)) <= max_rows
                else None
            )

        _VLLM_LAYER_DIAGNOSTIC_ACTIVE_LAYER = layer
        try:
            result = original_forward(instance, positions, hidden_states, residual)
        finally:
            _VLLM_LAYER_DIAGNOSTIC_ACTIVE_LAYER = None
        output, output_residual = result
        buffer = _VLLM_LAYER_DIAGNOSTIC_BUFFER
        if buffer is None:
            return result
        semantic_output = output + output_residual
        buffer["outputs"].append(semantic_output.detach())
        if layer == 0:
            layer_zero = buffer.setdefault("layer0", {})
            layer_zero["attention_residual"] = output_residual.detach()
            layer_zero["mlp_norm"] = torch.nn.functional.rms_norm(
                output_residual,
                (output_residual.shape[-1],),
                instance.post_attention_layernorm.weight,
                instance.post_attention_layernorm.variance_epsilon,
            ).detach()
            layer_zero["mlp_output"] = output.detach()
        if layer != layer_count - 1:
            return result
        if len(buffer["outputs"]) != layer_count:
            raise RuntimeError(
                "vLLM layer diagnostics did not observe every decoder layer: "
                f"{len(buffer['outputs'])} != {layer_count}"
            )
        call_index = _VLLM_LAYER_DIAGNOSTIC_CALLS
        _VLLM_LAYER_DIAGNOSTIC_CALLS += 1
        rank = _diagnostic_rank()
        payload = {
            "schema_version": "rlkernel.layer_alignment_diagnostic.v1",
            "framework": "vllm",
            "pid": os.getpid(),
            "rank": rank,
            "call_index": call_index,
            "positions": buffer["positions"].cpu(),
            "input": buffer["input"].cpu(),
            "outputs": torch.stack(buffer["outputs"]).cpu(),
        }
        if "layer0" in buffer:
            payload["layer0"] = {name: value.cpu() for name, value in buffer["layer0"].items()}
        _VLLM_LAYER_DIAGNOSTIC_BUFFER = None
        root = os.getenv("RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR", "").strip()
        if root:
            output_dir = pathlib.Path(root) / "layers"
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                payload,
                output_dir / (f"vllm-pid{os.getpid()}-rank{rank:05d}-" f"call{call_index:08d}.pt"),
            )
        return result

    def attention_forward_wrapped(
        instance: Any,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        buffer = _VLLM_LAYER_DIAGNOSTIC_BUFFER
        if buffer is None or _VLLM_LAYER_DIAGNOSTIC_ACTIVE_LAYER != 0:
            return original_attention_forward(instance, positions, hidden_states)
        qkv, _ = instance.qkv_proj(hidden_states)
        q, k, v = qkv.split([instance.q_size, instance.kv_size, instance.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // instance.head_dim, instance.head_dim)
        q_by_head = instance.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // instance.head_dim, instance.head_dim)
        k_by_head = instance.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = instance.rotary_emb(positions, q, k)
        attention_core = instance.attn(q, k, v)
        output, _ = instance.o_proj(attention_core)
        buffer.setdefault("layer0", {}).update(
            {
                "attention_norm": hidden_states.detach(),
                "qkv": qkv.detach(),
                "query": q.view(*q.shape[:-1], -1, instance.head_dim).detach(),
                "key": k.view(*k.shape[:-1], -1, instance.head_dim).detach(),
                "value": v.view(*v.shape[:-1], -1, instance.head_dim).detach(),
                "attention_core": attention_core.detach(),
                "attention_output": output.detach(),
            }
        )
        return output

    def gather_paged_row_wrapped(
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_row: torch.Tensor,
        cached_length: int,
        *,
        validate_bounds: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key, value = original_gather_paged_row(
            k_cache,
            v_cache,
            page_row,
            cached_length,
            validate_bounds=validate_bounds,
        )
        buffer = _VLLM_LAYER_DIAGNOSTIC_BUFFER
        if buffer is not None and _VLLM_LAYER_DIAGNOSTIC_ACTIVE_LAYER == 0:
            buffer.setdefault("layer0", {}).update(
                {
                    "logical_key": key.detach(),
                    "logical_value": value.detach(),
                }
            )
        return key, value

    setattr(Qwen3DecoderLayer, _STRICT_LAYER_DIAGNOSTIC_PATCH_MARKER, original_forward)
    Qwen3DecoderLayer.__init__ = init_wrapped
    Qwen3DecoderLayer.forward = forward_wrapped
    setattr(Qwen3Attention, _STRICT_LAYER_DIAGNOSTIC_PATCH_MARKER, original_attention_forward)
    Qwen3Attention.forward = attention_forward_wrapped
    StrictRocmAttentionRuntime._gather_paged_row = staticmethod(gather_paged_row_wrapped)


def _o_proj_collective_backend() -> str | None:
    return _RLK_O_PROJ_COLLECTIVE_BACKEND


def _is_worker_sampler_profile_batch(value: Any) -> bool:
    """Return true for vLLM's KV-cache profiling dummy sampler batch."""

    req_ids = getattr(value, "req_ids", None)
    is_padding = getattr(value, "is_padding", None)
    if not isinstance(req_ids, list) or not req_ids:
        return False
    if not all(isinstance(item, str) and item.startswith("req_") for item in req_ids):
        return False
    if is_padding is None or not hasattr(is_padding, "all"):
        return False
    try:
        return bool(is_padding.all().item())
    except (AttributeError, RuntimeError, TypeError):
        return False


def _is_v1_sampler_profile_batch(value: Any) -> bool:
    """Identify the no-logprob dummy batch used to size vLLM's KV cache."""

    if getattr(value, "max_num_logprobs", object()) is not None:
        return False
    output_token_ids = getattr(value, "output_token_ids", None)
    spec_token_ids = getattr(value, "spec_token_ids", None)
    if not (
        isinstance(output_token_ids, list)
        and output_token_ids
        and all(isinstance(ids, list) and not ids for ids in output_token_ids)
        and isinstance(spec_token_ids, list)
        and len(spec_token_ids) == len(output_token_ids)
        and all(isinstance(ids, list) and not ids for ids in spec_token_ids)
    ):
        return False
    return (
        bool(getattr(value, "no_penalties", False))
        and getattr(value, "prompt_token_ids", None) is None
        and getattr(value, "allowed_token_ids_mask", None) is None
        and getattr(value, "bad_words_token_ids", None) == {}
        and getattr(value, "generators", None) == {}
    )


def plan_from_environment() -> IntegrationPlan:
    """Compatibility alias for the shared process plan loader."""

    return integration_plan_from_environment()


def configure_vllm_environment(plan: IntegrationPlan, *, readback_dir: str | None = None) -> None:
    """Export the plan inherited by Vime's vLLM server subprocess."""

    os.environ["RL_KERNEL_VLLM_INTEGRATION"] = "1"
    configure_integration_environment(plan, readback_dir=readback_dir)
    if plan.implementation_for("attention", "rollout") is Implementation.RL_KERNEL:
        os.environ["VLLM_ATTENTION_BACKEND"] = (
            "ROCM_AITER_FA" if torch.version.hip is not None else "FLASH_ATTN"
        )
    if plan.implementation_for("logp", "rollout") is Implementation.RL_KERNEL:
        real_vocab = os.getenv("RL_KERNEL_VLLM_REAL_VOCAB_SIZE", "").strip()
        padded_vocab = os.getenv("RL_KERNEL_VLLM_PADDED_VOCAB_SIZE", "").strip()
        if not real_vocab or not padded_vocab:
            raise RuntimeError(
                "strict rollout linear_logp requires "
                "RL_KERNEL_VLLM_REAL_VOCAB_SIZE and "
                "RL_KERNEL_VLLM_PADDED_VOCAB_SIZE"
            )


def _patch_qwen_lm_head_padding() -> None:
    """Make vLLM's TP LM-head partition identical to Megatron's padded layout."""

    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    if hasattr(ParallelLMHead, _PATCH_MARKER):
        return
    real_vocab = int(os.environ["RL_KERNEL_VLLM_REAL_VOCAB_SIZE"])
    padded_vocab = int(os.environ["RL_KERNEL_VLLM_PADDED_VOCAB_SIZE"])
    if padded_vocab <= real_vocab or padded_vocab % 2:
        raise RuntimeError(
            f"invalid strict rollout vocab contract: real={real_vocab}, padded={padded_vocab}"
        )
    padding_size = padded_vocab - real_vocab
    original = ParallelLMHead.__init__
    original_weight_loader = VocabParallelEmbedding.weight_loader
    original_tie_weights = getattr(ParallelLMHead, "tie_weights", None)

    def strict_weight_loader(instance: Any, param: Any, loaded_weight: torch.Tensor) -> None:
        # vLLM loaders write through ``param.data``, which does not reliably
        # advance the Parameter version counter.  Explicitly invalidate before
        # any write so a failed or incomplete hot update cannot reuse old
        # prepared LM-head bytes.
        _invalidate_lm_head_weight_cache(param)
        strict_real_vocab = getattr(instance, "_rl_kernel_real_vocab_size", None)
        output_dim = getattr(param, "output_dim", None)
        packed_dim = getattr(param, "packed_dim", None)
        if (
            strict_real_vocab is not None
            and output_dim is not None
            and packed_dim is None
            and loaded_weight.shape[output_dim] == int(strict_real_vocab)
        ):
            start_idx = int(instance.shard_indices.org_vocab_start_index)
            shard_size = int(instance.shard_indices.org_vocab_end_index - start_idx)
            available = max(0, min(shard_size, int(strict_real_vocab) - start_idx))
            if available:
                loaded_slice = loaded_weight.narrow(output_dim, start_idx, available)
                param[:available].data.copy_(loaded_slice)
            if available < shard_size:
                param[available:shard_size].data.fill_(0)
            if shard_size < param.shape[0]:
                param[shard_size:].data.fill_(0)
            _mark_lm_head_weight_cache_refreshable(param)
            return
        original_weight_loader(instance, param, loaded_weight)
        _mark_lm_head_weight_cache_refreshable(param)

    def wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
        if args:
            args = (padded_vocab, *args[1:])
        else:
            kwargs["num_embeddings"] = padded_vocab
        # vLLM normally stores original-vocab and added-vocab rows as separate
        # segments. Strict R/R needs Megatron's single contiguous padded vocab.
        kwargs["org_num_embeddings"] = padded_vocab
        kwargs["padding_size"] = padding_size
        original(instance, *args, **kwargs)
        instance._rl_kernel_real_vocab_size = real_vocab
        setattr(instance, _STRICT_PROJECTION_MARKER, "lm_head")
        if (
            int(getattr(instance, "org_vocab_size", -1)) != padded_vocab
            or int(getattr(instance, "num_embeddings_padded", -1)) != padded_vocab
        ):
            raise RuntimeError(
                "strict rollout LM-head padding did not produce the Megatron "
                f"padded vocab layout {padded_vocab}"
            )

    setattr(ParallelLMHead, _PATCH_MARKER, original)
    if callable(original_tie_weights) and not hasattr(
        ParallelLMHead, _STRICT_LM_HEAD_TIE_PATCH_MARKER
    ):

        def strict_tie_weights(instance: Any, embed_tokens: Any) -> Any:
            result = original_tie_weights(instance, embed_tokens)
            setattr(instance, _STRICT_LM_HEAD_TIED, True)
            return result

        setattr(
            ParallelLMHead,
            _STRICT_LM_HEAD_TIE_PATCH_MARKER,
            original_tie_weights,
        )
        setattr(ParallelLMHead, "tie_weights", strict_tie_weights)
    if not hasattr(VocabParallelEmbedding, "__rl_kernel_original_weight_loader__"):
        setattr(
            VocabParallelEmbedding,
            "__rl_kernel_original_weight_loader__",
            original_weight_loader,
        )
        setattr(VocabParallelEmbedding, "weight_loader", strict_weight_loader)
    setattr(ParallelLMHead, "__init__", wrapped)


def _patch_qwen_compute_logits(integration: VllmIntegration) -> None:
    """Publish the exact hidden and padded LM-head shard to the sampler."""

    from vllm.distributed import get_pp_group, get_tp_group
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

    classes = tuple(dict.fromkeys((Qwen3ForCausalLM, Qwen2ForCausalLM)))
    installed: list[str] = []
    for cls in classes:
        if hasattr(cls, _PATCH_MARKER):
            continue
        original = cls.compute_logits  # type: ignore[attr-defined]

        def wrapped(
            instance: Any,
            hidden_states: Any,
            *,
            _original: Any = original,
        ) -> Any:
            lm_head = getattr(instance, "lm_head", None)
            if lm_head is None:
                raise RuntimeError("strict rollout linear_logp requires a Qwen LM-head")
            model = getattr(instance, "model", None)
            if (
                torch.version.hip is not None
                and model is not None
                and lm_head is getattr(model, "embed_tokens", None)
            ):
                setattr(lm_head, _STRICT_LM_HEAD_TIED, True)
                raise RuntimeError("strict ROCm LM-head cache does not support tied embeddings")
            setattr(lm_head, _STRICT_PROJECTION_MARKER, "lm_head")
            logits = _original(instance, hidden_states)
            if not get_pp_group().is_last_rank:
                return logits
            shard_indices = getattr(lm_head, "shard_indices", None)
            weight = getattr(lm_head, "weight", None)
            if lm_head is None or shard_indices is None or not isinstance(weight, torch.Tensor):
                raise RuntimeError("strict rollout linear_logp requires a real Qwen LM-head shard")
            tp = get_tp_group()
            tp_world = int(tp.world_size)
            publish_rollout_linear_logp_context(
                hidden_states,
                weight,
                getattr(lm_head, "bias", None),
                tp_group=tp.device_group if tp_world > 1 else None,
                vocab_start_index=int(shard_indices.padded_org_vocab_start_index),
                global_vocab_size=int(lm_head.num_embeddings_padded),
                real_vocab_size=int(
                    getattr(lm_head, "_rl_kernel_real_vocab_size", lm_head.org_vocab_size)
                ),
            )
            return logits

        setattr(cls, _PATCH_MARKER, original)
        setattr(cls, "compute_logits", wrapped)
        installed.append(f"{cls.__module__}.{cls.__name__}.compute_logits")
    if installed:
        integration.record_installed_hook("logp", ",".join(installed))


def _patch_strict_lm_head_linear(
    *,
    linear_method_cls: type[Any] | None = None,
    det_gemm: Any | None = None,
    prepare_weight: Any | None = None,
) -> None:
    """Route vLLM's existing LM-head projection through RL-Kernel det_gemm."""

    if linear_method_cls is None:
        from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

        linear_method_cls = UnquantizedEmbeddingMethod
    if hasattr(linear_method_cls, _STRICT_LM_HEAD_LINEAR_PATCH_MARKER):
        return
    previous_apply = linear_method_cls.apply
    if det_gemm is None:
        from rl_engine.kernels.ops.matmul.det_gemm import DetGemmOp

        det_gemm = DetGemmOp()
    if torch.version.hip is not None and prepare_weight is None:
        from rl_engine.kernels.ops.rocm.matmul.det_gemm import prepare_det_gemm_linear_weight

        prepare_weight = prepare_det_gemm_linear_weight

    if not hasattr(linear_method_cls, _STRICT_LM_HEAD_PROCESS_PATCH_MARKER):
        previous_process_weights = linear_method_cls.process_weights_after_loading

        def process_weights_after_loading(
            method: Any,
            layer: Any,
        ) -> Any:
            result = previous_process_weights(method, layer)
            if (
                torch.version.hip is not None
                and getattr(layer, _STRICT_PROJECTION_MARKER, None) == "lm_head"
            ):
                if prepare_weight is None:
                    raise RuntimeError("strict ROCm LM-head weight preparation is unavailable")
                _refresh_lm_head_weight_cache(layer, prepare_weight)
            return result

        setattr(
            linear_method_cls,
            _STRICT_LM_HEAD_PROCESS_PATCH_MARKER,
            previous_process_weights,
        )
        setattr(
            linear_method_cls,
            "process_weights_after_loading",
            process_weights_after_loading,
        )

    def wrapped(
        method: Any,
        layer: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, _STRICT_PROJECTION_MARKER, None) != "lm_head":
            return previous_apply(method, layer, x, bias)
        x_2d = x.reshape(-1, x.shape[-1])
        if torch.version.hip is not None:
            linear_prepared = getattr(det_gemm, "linear_prepared", None)
            if not callable(linear_prepared):
                raise RuntimeError("strict ROCm LM-head prepared GEMM is unavailable")
            output_2d = linear_prepared(
                x_2d,
                _validated_lm_head_weight_cache(layer, prepare_weight),
            )
        else:
            output_2d = det_gemm.linear(x_2d, layer.weight)
        output = output_2d.reshape(*x.shape[:-1], layer.weight.size(0))
        return output if bias is None else output + bias

    setattr(
        linear_method_cls,
        _STRICT_LM_HEAD_LINEAR_PATCH_MARKER,
        previous_apply,
    )
    setattr(linear_method_cls, "apply", wrapped)


def _patch_strict_rocm_rotary_embedding(rotary_cls: type[Any]) -> None:
    """Bind vLLM Qwen3 RotaryEmbedding to the shared ROCm RoPE kernel."""

    if torch.version.hip is None or hasattr(rotary_cls, _STRICT_ROCM_ROPE_PATCH_MARKER):
        return
    from rl_engine.kernels.ops.rocm.rotary_embedding.rope import RocmDeterministicRoPEOp

    operator = RocmDeterministicRoPEOp()
    original = rotary_cls.forward_cuda

    def prepare_tables(instance: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        head_size = int(getattr(instance, "head_size", 0))
        max_positions = int(getattr(instance, "max_position_embeddings", 0))
        theta = float(getattr(instance, "base", 1_000_000.0))
        key = (device.type, device.index, head_size, max_positions, theta)
        if getattr(instance, "_rl_kernel_rope_table_key", None) == key:
            cos = getattr(instance, "_rl_kernel_rope_cos_fp32", None)
            sin = getattr(instance, "_rl_kernel_rope_sin_fp32", None)
            if isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor):
                return cos, sin
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "strict ROCm RoPE FP32 table must be initialized before HIP Graph capture"
            )
        cos, sin = operator.build_position_table(
            max_positions,
            head_size,
            device=device,
            theta=theta,
        )

        def register_table(name: str, table: torch.Tensor) -> None:
            if isinstance(instance, torch.nn.Module):
                buffers = instance._buffers
                if name in buffers:
                    buffers[name] = table
                else:
                    instance.register_buffer(name, table, persistent=False)
            else:
                setattr(instance, name, table)

        register_table("_rl_kernel_rope_cos_fp32", cos)
        register_table("_rl_kernel_rope_sin_fp32", sin)
        instance._rl_kernel_rope_table_key = key
        return cos, sin

    def strict_forward_cuda(
        instance: Any,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        head_size = int(getattr(instance, "head_size", 0))
        rotary_dim = int(getattr(instance, "rotary_dim", head_size))
        if head_size <= 0 or rotary_dim != head_size:
            raise RuntimeError(
                "strict ROCm Qwen3 RoPE requires full-dimension rotation: "
                f"rotary_dim={rotary_dim}, head_size={head_size}"
            )
        if positions.ndim not in (1, 2):
            raise RuntimeError(
                "strict ROCm vLLM RoPE positions must align with flattened query rows: "
                f"positions={tuple(positions.shape)}, query={tuple(query.shape)}"
            )
        torch._check(
            positions.numel() == query.shape[0],
            lambda: "strict ROCm vLLM RoPE positions must align with query rows",
        )
        flat_positions = positions.reshape(-1).to(device=query.device, dtype=torch.int64)
        if not flat_positions.is_contiguous():
            flat_positions = flat_positions.contiguous()
        cos, sin = prepare_tables(instance, query.device)

        def apply(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            if value.ndim != 2 or value.shape[1] % head_size:
                raise RuntimeError(
                    "strict ROCm vLLM RoPE expects flattened [tokens, heads*head_dim] tensors"
                )
            return operator.forward_token_major(
                value,
                flat_positions,
                cos,
                sin,
                head_dim=head_size,
            )

        return apply(query), apply(key)

    strict_forward_cuda.__name__ = getattr(original, "__name__", "forward_cuda")
    setattr(rotary_cls, "_rl_kernel_prepare_strict_rocm_tables", prepare_tables)
    setattr(rotary_cls, _STRICT_ROCM_ROPE_PATCH_MARKER, original)
    rotary_cls.forward_cuda = strict_forward_cuda


def _configure_strict_ffn_compilation(vllm_config: Any | None = None) -> None:
    """Keep device-sequenced TP reductions inside replayed accelerator graphs."""

    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        raise RuntimeError("strict TP FFN initialization requires an active vLLM config")

    compilation = vllm_config.compilation_config
    splitting_ops = compilation.splitting_ops
    if splitting_ops is None:
        raise RuntimeError("vLLM splitting operators were not finalized before model init")
    if torch.version.hip is None:
        # Preserve the CUDA full-graph path introduced by PR 377.
        splitting_ops[:] = [op for op in splitting_ops if op != DETERMINISTIC_ALL_REDUCE_OP]
        return

    from vllm import envs as vllm_envs
    from vllm.config import CUDAGraphMode

    route_key = "_".join(
        re.sub(r"[^a-z0-9]+", "-", os.getenv(name, "unset").lower()).strip("-")
        for name in _ROCM_GRAPH_ROUTE_ENVIRONMENT
    )
    cache_namespace = f"{_ROCM_FULL_GRAPH_CACHE_NAMESPACE}_{route_key}"
    cache_root = os.path.normpath(os.fspath(vllm_envs.VLLM_CACHE_ROOT))
    if os.path.basename(cache_root) != cache_namespace:
        # vLLM's AOT key cannot see implementations behind torch custom ops.
        # Keep its normal config/code/compiler hashing under an RL-Kernel ABI
        # namespace so an older custom-op artifact cannot be replayed silently.
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(cache_root, cache_namespace)
    compilation.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    # ROCm IPC generations are allocated and consumed on device. Replayed
    # reductions therefore advance their generation instead of reusing the
    # capture-time payload, so these ops can remain in the full HIP graph.
    splitting_ops[:] = [op for op in splitting_ops if op not in _ROCM_STATEFUL_GRAPH_SPLITTING_OPS]


def _patch_rocm_weight_cache_refresh() -> None:
    """Refresh stable transpose buffers after vLLM IPC weight transfer."""

    if torch.version.hip is None:
        return
    from vllm.v1.worker.gpu_worker import Worker
    from rl_engine.kernels.ops.rocm.matmul.det_gemm import (
        refresh_cached_weight_transposes,
    )

    if hasattr(Worker, _STRICT_WEIGHT_CACHE_REFRESH_MARKER):
        return
    original_finish = Worker.finish_weight_update

    def finish_weight_update_wrapped(instance: Any) -> None:
        original_finish(instance)
        model = instance.model_runner.get_model()
        refresh_cached_weight_transposes(model.parameters())

    setattr(Worker, _STRICT_WEIGHT_CACHE_REFRESH_MARKER, original_finish)
    Worker.finish_weight_update = finish_weight_update_wrapped


def _patch_qwen_ffn(integration: VllmIntegration) -> None:
    from vllm.model_executor.models.qwen2 import Qwen2MLP

    operator: VllmFFNOperator | None = None
    if integration.plan.implementation_for("ffn", "rollout") is Implementation.RL_KERNEL:
        operator = VllmFFNOperator()
        integration.install_operator("ffn", operator)
    if hasattr(Qwen2MLP, _PATCH_MARKER):
        raise RuntimeError("vLLM Qwen2MLP is already RL-Kernel patched")
    original = Qwen2MLP.forward

    if operator is not None:
        if hasattr(Qwen2MLP, _STRICT_FFN_INIT_MARKER):
            raise RuntimeError("vLLM Qwen2MLP init is already RL-Kernel patched")
        original_init = Qwen2MLP.__init__
        compiled_evidence_armed = False

        def wrapped_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            nonlocal compiled_evidence_armed
            original_init(instance, *args, **kwargs)
            _handle, tp_world_size = operator.bind_packed_inference(instance)
            if not compiled_evidence_armed:
                execution_mode = (
                    "compiled_hip_graph"
                    if getattr(torch.version, "hip", None) is not None
                    else "compiled_cuda_graph"
                )
                register_packed_inference_observer(
                    lambda: integration.record_execution(
                        "ffn", operator, execution_mode=execution_mode
                    )
                )
                compiled_evidence_armed = True
            if tp_world_size > 1:
                _configure_strict_ffn_compilation()

        setattr(Qwen2MLP, _STRICT_FFN_INIT_MARKER, original_init)
        setattr(Qwen2MLP, "__init__", wrapped_init)

    def wrapped(instance: Any, hidden_states: Any) -> Any:
        def native(_module: Any, value: Any) -> Any:
            return original(instance, value)

        return integration.execute("ffn", native, instance, hidden_states)

    setattr(Qwen2MLP, _PATCH_MARKER, original)
    setattr(Qwen2MLP, "forward", wrapped)
    integration.record_installed_hook("ffn", "vllm.model_executor.models.qwen2.Qwen2MLP.forward")


def _install_flash_attn_ops_compatibility() -> None:
    """Supply the legacy rotary import tree when an FA4-only package is installed."""

    try:
        importlib.import_module("flash_attn.ops.triton.rotary")
        return
    except (ImportError, OSError, RuntimeError):
        pass

    from vllm.vllm_flash_attn import ops as bundled_ops
    from vllm.vllm_flash_attn.ops import triton as bundled_triton
    from vllm.vllm_flash_attn.ops.triton import rotary as bundled_rotary

    sys.modules.setdefault("flash_attn.ops", bundled_ops)
    sys.modules.setdefault("flash_attn.ops.triton", bundled_triton)
    sys.modules.setdefault("flash_attn.ops.triton.rotary", bundled_rotary)


def _patch_qwen3_strict_model(
    *,
    rms_norm_cls: type[Any] | None = None,
    rotary_cls: type[Any] | None = None,
    linear_method_cls: type[Any] | None = None,
    attention_cls: type[Any] | None = None,
    row_parallel_cls: type[Any] | None = None,
    det_gemm: Any | None = None,
) -> None:
    """Align vLLM's RMSNorm and Attention projections with Megatron."""

    production_classes = rms_norm_cls is None or linear_method_cls is None or attention_cls is None
    if production_classes:
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
        from vllm.model_executor.models.qwen3 import Qwen3Attention

        rms_norm_cls = RMSNorm
        rotary_cls = RotaryEmbedding
        linear_method_cls = UnquantizedLinearMethod
        attention_cls = Qwen3Attention
        row_parallel_cls = RowParallelLinear
    assert rms_norm_cls is not None
    assert linear_method_cls is not None
    assert attention_cls is not None
    if rotary_cls is not None:
        _patch_strict_rocm_rotary_embedding(rotary_cls)
    if det_gemm is None:
        det_gemm = _strict_attention_projection_op()
    rocm_linear_all_reduce = None
    register_rocm_linear_staging = None
    if torch.version.hip is not None:
        from rl_engine.kernels.ops.rocm.matmul.det_gemm import (
            det_gemm_linear_all_reduce_inference,
            register_det_gemm_all_reduce_staging,
        )

        rocm_linear_all_reduce = det_gemm_linear_all_reduce_inference
        register_rocm_linear_staging = register_det_gemm_all_reduce_staging

    attention_init = attention_cls.__init__
    unquantized_apply = linear_method_cls.apply
    original_rms_forward_native = rms_norm_cls.forward_native

    def deterministic_linear_apply(
        method: Any,
        layer: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not hasattr(layer, _STRICT_PROJECTION_MARKER):
            return unquantized_apply(method, layer, x, bias)
        x_2d = x.reshape(-1, x.shape[-1])
        linear = getattr(det_gemm, "linear", None)
        collective = getattr(layer, _STRICT_O_PROJ_COLLECTIVE_MARKER, None)
        if collective is not None and torch.version.hip is not None:
            if bias is not None or rocm_linear_all_reduce is None:
                raise RuntimeError("strict ROCm o_proj fusion requires a bias-free linear")
            output_2d = rocm_linear_all_reduce(
                x_2d,
                layer.weight,
                collective_handle=int(getattr(layer, _STRICT_O_PROJ_COMPILED_COLLECTIVE_SLOT)),
            )
            return output_2d.reshape(*x.shape[:-1], layer.weight.shape[0])
        direct_output = None
        if collective is not None and bias is None and linear is not None:
            direct_output = collective.direct_staging_view(
                (x_2d.size(0), layer.weight.shape[0]),
                dtype=x.dtype,
            )
            if direct_output is not None:
                deterministic_staging_reserve(
                    direct_output,
                    collective_handle=int(collective._handle),
                )
        output_2d = (
            linear(x_2d, layer.weight, out=direct_output)
            if linear is not None
            else det_gemm(x_2d, layer.weight.t().contiguous())
        )
        setattr(layer, _STRICT_DIRECT_STAGING_MARKER, direct_output is not None)
        output = output_2d.reshape(*x.shape[:-1], layer.weight.shape[0])
        return output if bias is None else output + bias

    def strict_attention_rms_norm_forward(
        instance: Any,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual is not None:
            raise RuntimeError("strict Attention Q/K RMSNorm does not accept a residual")
        if instance.variance_size_override is not None:
            raise RuntimeError("strict Attention Q/K RMSNorm requires the full head dimension")
        weight = instance.weight.data if instance.has_weight else None
        if weight is None:
            raise RuntimeError("strict Attention Q/K RMSNorm requires a weight")
        return strict_rms_norm(x, weight, eps=instance.variance_epsilon)

    def require_rocm_graph_runtime() -> None:
        if not production_classes or torch.version.hip is None:
            return
        from vllm.config import CompilationMode, CUDAGraphMode, get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        model_config = None if config is None else config.model_config
        compilation_config = None if config is None else config.compilation_config
        if (
            model_config is None
            or model_config.enforce_eager is True
            or compilation_config is None
            or compilation_config.mode == CompilationMode.NONE
            or compilation_config.cudagraph_mode == CUDAGraphMode.NONE
        ):
            raise RuntimeError(
                "strict ROCm rollout requires compilation and HIP graph capture enabled"
            )

    def bind_attention_rms_norm(attention: Any, name: str) -> None:
        norm = getattr(attention, name, None)
        if not isinstance(norm, rms_norm_cls):
            raise RuntimeError(f"strict Qwen3 Attention requires an RMSNorm {name} instance")
        if getattr(norm, _STRICT_ATTENTION_RMS_NORM_MARKER, False):
            raise RuntimeError(f"strict Qwen3 Attention {name} is already bound")
        norm._forward_method = MethodType(strict_attention_rms_norm_forward, norm)
        setattr(norm, _STRICT_ATTENTION_RMS_NORM_MARKER, True)

    def strict_rms_norm_forward_cuda(
        instance: Any,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if instance.variance_size_override is not None:
            return original_rms_forward_native(instance, x, residual)
        weight = instance.weight.data if instance.has_weight else None
        if weight is None:
            return original_rms_forward_native(instance, x, residual)
        if residual is None:
            return strict_rms_norm(
                x,
                weight,
                eps=instance.variance_epsilon,
            )
        return strict_add_rms_norm(
            x,
            residual,
            weight,
            eps=instance.variance_epsilon,
        )

    def bind_o_proj_collective(module: Any) -> None:
        global _RLK_O_PROJ_COLLECTIVE_BACKEND

        if int(getattr(module, "tp_size", 1)) <= 1:
            _RLK_O_PROJ_COLLECTIVE_BACKEND = "none"
            return
        from vllm.distributed.parallel_state import get_tp_group

        coordinator = get_tp_group()
        group = getattr(coordinator, "device_group", coordinator)
        collective = collective_for_group(group)
        if collective is None:
            raise RuntimeError("strict rollout o_proj requires an initialized TP process group")
        setattr(module, _STRICT_O_PROJ_COLLECTIVE_MARKER, collective)
        backend_id = getattr(collective, "backend_id", None)
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise RuntimeError("strict rollout o_proj collective has no backend identity")
        _RLK_O_PROJ_COLLECTIVE_BACKEND = backend_id.strip()
        max_capture = int(os.getenv("RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE", "0"))
        if max_capture <= 0:
            raise RuntimeError(
                "strict rollout direct staging requires a positive graph capture size"
            )
        collective.prepare_direct_staging_views(
            ((batch, int(module.weight.shape[0])) for batch in range(1, max_capture + 1)),
            dtype=module.weight.dtype,
        )
        if torch.version.hip is not None and production_classes:
            staging = collective.direct_staging_view(
                (max_capture, int(module.weight.shape[0])),
                dtype=module.weight.dtype,
            )
            if staging is None:
                raise RuntimeError("strict ROCm o_proj staging allocation failed")
            if register_rocm_linear_staging is None:
                raise RuntimeError("strict ROCm o_proj staging registry is unavailable")
            compiled_slot = register_rocm_linear_staging(int(collective._handle), staging)
            setattr(module, _STRICT_O_PROJ_COMPILED_COLLECTIVE_SLOT, compiled_slot)
            setattr(module, _STRICT_O_PROJ_FUSED_ALL_REDUCE_MARKER, True)

    if row_parallel_cls is not None and not hasattr(
        row_parallel_cls, _STRICT_ROW_PARALLEL_PATCH_MARKER
    ):
        row_parallel_forward = row_parallel_cls.forward

        def strict_row_parallel_forward(instance: Any, input_: torch.Tensor) -> Any:
            collective = getattr(instance, _STRICT_O_PROJ_COLLECTIVE_MARKER, None)
            if collective is None:
                return row_parallel_forward(instance, input_)

            if instance.input_is_parallel:
                input_parallel = input_
            else:
                from vllm.distributed import split_tensor_along_last_dim

                input_parallel = split_tensor_along_last_dim(
                    input_, num_partitions=instance.tp_size
                )[instance.tp_rank].contiguous()

            assert instance.quant_method is not None
            bias_ = None if (instance.tp_rank > 0 or instance.skip_bias_add) else instance.bias
            output_parallel = instance.quant_method.apply(instance, input_parallel, bias_)

            if instance.reduce_results and instance.tp_size > 1:
                if bool(getattr(instance, _STRICT_O_PROJ_FUSED_ALL_REDUCE_MARKER, False)):
                    output = output_parallel
                elif bool(getattr(instance, _STRICT_DIRECT_STAGING_MARKER, False)):
                    output = deterministic_all_reduce_staged(
                        output_parallel,
                        collective_handle=int(collective._handle),
                    )
                else:
                    deterministic_all_reduce_inplace(
                        output_parallel,
                        collective_handle=int(collective._handle),
                    )
                    output = output_parallel
            else:
                output = output_parallel

            if not instance.return_bias:
                return output
            output_bias = instance.bias if instance.skip_bias_add else None
            return output, output_bias

        setattr(row_parallel_cls, _STRICT_ROW_PARALLEL_PATCH_MARKER, row_parallel_forward)
        row_parallel_cls.forward = strict_row_parallel_forward

    if not hasattr(rms_norm_cls, _STRICT_RMS_NORM_INIT_MARKER):
        rms_norm_init = rms_norm_cls.__init__

        def rms_norm_init_wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
            rms_norm_init(instance, *args, **kwargs)
            instance._forward_method = instance.forward_cuda

        setattr(rms_norm_cls, _STRICT_RMS_NORM_INIT_MARKER, rms_norm_init)
        rms_norm_cls.__init__ = rms_norm_init_wrapped
        rms_norm_cls.forward_cuda = strict_rms_norm_forward_cuda

    if rotary_cls is not None and not hasattr(rotary_cls, _STRICT_ROTARY_INIT_MARKER):
        rotary_init = rotary_cls.__init__

        def rotary_init_wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
            rotary_init(instance, *args, **kwargs)
            instance._forward_method = instance.forward_cuda
            cache = getattr(instance, "cos_sin_cache", None)
            prepare = getattr(instance, "_rl_kernel_prepare_strict_rocm_tables", None)
            if isinstance(cache, torch.Tensor) and cache.is_cuda and callable(prepare):
                prepare(cache.device)

        setattr(rotary_cls, _STRICT_ROTARY_INIT_MARKER, rotary_init)
        rotary_cls.__init__ = rotary_init_wrapped

    if hasattr(attention_cls, _STRICT_MODEL_PATCH_MARKER):
        return

    def attention_init_wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
        require_rocm_graph_runtime()
        attention_init(instance, *args, **kwargs)
        setattr(instance.qkv_proj, _STRICT_PROJECTION_MARKER, "qkv")
        setattr(instance.o_proj, _STRICT_PROJECTION_MARKER, "o_proj")
        if torch.version.hip is not None and production_classes:
            bind_attention_rms_norm(instance, "q_norm")
            bind_attention_rms_norm(instance, "k_norm")
        bind_o_proj_collective(instance.o_proj)

    setattr(attention_cls, _STRICT_MODEL_PATCH_MARKER, attention_init)
    linear_method_cls.apply = deterministic_linear_apply
    attention_cls.__init__ = attention_init_wrapped


def _patch_sampler(integration: VllmIntegration, *, strict_linear_logp: bool) -> None:
    from vllm.v1.sample.sampler import Sampler

    if hasattr(Sampler, _PATCH_MARKER):
        raise RuntimeError("vLLM Sampler is already RL-Kernel patched")
    original = Sampler.forward
    if strict_linear_logp:
        operator = VllmLogpOperator(original, strict_linear_logp=True)
        integration.install_operator("logp", operator)

    def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
        sampling_metadata = kwargs.get("sampling_metadata")
        if sampling_metadata is None and len(args) >= 2:
            sampling_metadata = args[1]
        if strict_linear_logp and _is_v1_sampler_profile_batch(sampling_metadata):
            # gpu_model_runner._dummy_sampler_run computes logits with a
            # synthetic no-logprob batch. It is an allocator warmup, not a
            # rollout scoring event, so do not route or count it as logp.
            try:
                return original(instance, *args, **kwargs)
            finally:
                clear_rollout_linear_logp_context()

        def native(_sampler: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            return original(instance, *call_args, **call_kwargs)

        return integration.execute("logp", native, instance, *args, **kwargs)

    setattr(Sampler, _PATCH_MARKER, original)
    setattr(Sampler, "forward", wrapped)
    integration.record_installed_hook("logp", "vllm.v1.sample.sampler.Sampler.forward")


def _patch_worker_sampler(integration: VllmIntegration, *, strict_linear_logp: bool) -> None:
    """Patch the CUDA worker sampler used by vLLM 0.27's V1 runner.

    vLLM has two sampler implementations: the graph-friendly
    ``vllm.v1.sample.Sampler`` and the CUDA worker sampler used by the
    production GPU runner. Qwen3 rollout on vLLM 0.27 uses the latter.
    """

    try:
        from vllm.v1.worker.gpu.sample.sampler import Sampler
    except ImportError:
        return

    if hasattr(Sampler, _PATCH_MARKER):
        raise RuntimeError("vLLM GPU worker Sampler is already RL-Kernel patched")
    original = Sampler.__call__
    operator = VllmLogpOperator(
        original,
        worker_sampler=True,
        strict_linear_logp=strict_linear_logp,
    )
    integration.install_operator("logp", operator)

    def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
        def native(_sampler: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            return original(instance, *call_args, **call_kwargs)

        if args and _is_worker_sampler_profile_batch(args[1] if len(args) > 1 else None):
            return original(instance, *args, **kwargs)
        return integration.execute("logp", native, instance, *args, **kwargs)

    setattr(Sampler, _PATCH_MARKER, original)
    setattr(Sampler, "__call__", wrapped)
    integration.record_installed_hook("logp", "vllm.v1.worker.gpu.sample.sampler.Sampler.__call__")


def _register_attention_backend(integration: VllmIntegration) -> None:
    global _RLK_ATTENTION_BACKEND, _RLK_ATTENTION_BUILDER, _RLK_ATTENTION_IMPL

    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    if torch.version.hip is not None:
        from vllm.v1.attention.backends.rocm_aiter_fa import (
            AiterFlashAttentionBackend as PlatformAttentionBackend,
        )
        from vllm.v1.attention.backends.rocm_aiter_fa import (
            AiterFlashAttentionImpl as PlatformAttentionImpl,
        )
        from vllm.v1.attention.backends.rocm_aiter_fa import (
            AiterFlashAttentionMetadataBuilder as PlatformAttentionMetadataBuilder,
        )

        selected_backend = AttentionBackendEnum.ROCM_AITER_FA
    else:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend as PlatformAttentionBackend,
        )
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionImpl as PlatformAttentionImpl,
        )
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionMetadataBuilder as PlatformAttentionMetadataBuilder,
        )

        selected_backend = AttentionBackendEnum.FLASH_ATTN

    operator: VllmAttentionOperator | None = None
    if integration.plan.implementation_for("attention", "rollout") is Implementation.RL_KERNEL:
        operator = VllmAttentionOperator(projection_collective_backend=_o_proj_collective_backend)
        integration.install_operator("attention", operator)

    class RlKernelAttentionImpl(PlatformAttentionImpl):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if operator is not None:
                operator.bind_inference()
            if torch.version.hip is not None and operator is not None:
                from vllm.config import get_current_vllm_config_or_none

                config = get_current_vllm_config_or_none()
                model_config = None if config is None else config.model_config
                dtype = None if model_config is None else model_config.dtype
                if dtype in (torch.float16, torch.bfloat16):
                    operator.warmup_rocm_decode(self, dtype=dtype)

        def _split_kv_cache(self, kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if torch.version.hip is not None and operator is not None:
                if (
                    kv_cache.ndim != 4
                    or kv_cache.size(2) != int(self.num_kv_heads)
                    or kv_cache.size(-1) != 2 * int(self.head_size)
                ):
                    raise RuntimeError(
                        "RL-Kernel ROCm KV cache must use " "[blocks, block, heads, 2 * head_size]"
                    )
                return kv_cache.split(int(self.head_size), dim=-1)
            return super()._split_kv_cache(kv_cache)

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            active = get_active_integration("vllm")
            if active is not integration:
                raise RuntimeError("vLLM Attention executed without its installed integration")

            def native(_impl: Any, *call_args: Any, **call_kwargs: Any) -> Any:
                return PlatformAttentionImpl.forward(self, *call_args, **call_kwargs)

            return integration.execute("attention", native, self, *args, **kwargs)

    class RlKernelAttentionMetadataBuilder(PlatformAttentionMetadataBuilder):
        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: Any,
            fast_build: bool = False,
        ) -> Any:
            metadata = super().build(
                common_prefix_len,
                common_attn_metadata,
                fast_build=fast_build,
            )
            # The ROCm RL-Kernel adapter can reuse vLLM's exact CPU snapshot
            # for pure decode scheduling. Keep this private metadata on the
            # adapter-owned object; native vLLM code remains untouched.
            if torch.version.hip is not None:
                # attn_utils reconstructs CommonAttentionMetadata without the
                # deprecated private CPU cache. Its upper-bound snapshot is
                # exact for prefill, which is the only path consuming it here.
                host_lengths = getattr(common_attn_metadata, "seq_lens_cpu_upper_bound", None)
                if isinstance(host_lengths, torch.Tensor) and host_lengths.device.type == "cpu":
                    setattr(metadata, "_rlk_seq_lens_cpu", host_lengths)
                starts = getattr(common_attn_metadata, "query_start_loc_cpu", None)
                if isinstance(starts, torch.Tensor) and starts.device.type == "cpu":
                    setattr(metadata, "_rlk_query_start_loc_cpu", starts)
            return metadata

    class RlKernelAttentionBackend(PlatformAttentionBackend):
        @staticmethod
        def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_dtype_str: str = "auto",
        ) -> tuple[int, ...]:
            if torch.version.hip is not None and operator is not None:
                if block_size % 16 != 0:
                    raise ValueError("Block size must be a multiple of 16.")
                # Token-major pages let AITER mha_batch_prefill consume K/V
                # directly.  The inherited cache-update methods call the
                # adapter-owned _split_kv_cache above, so no vLLM source or
                # cache-update kernel needs to change.
                return (num_blocks, block_size, num_kv_heads, 2 * head_size)
            return PlatformAttentionBackend.get_kv_cache_shape(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                cache_dtype_str,
            )

        @staticmethod
        def get_impl_cls() -> type[Any]:
            return RlKernelAttentionImpl

        @staticmethod
        def get_builder_cls() -> type[Any]:
            return RlKernelAttentionMetadataBuilder

    RlKernelAttentionImpl.__module__ = __name__
    RlKernelAttentionImpl.__qualname__ = "RlKernelAttentionImpl"
    RlKernelAttentionMetadataBuilder.__module__ = __name__
    RlKernelAttentionMetadataBuilder.__qualname__ = "RlKernelAttentionMetadataBuilder"
    RlKernelAttentionBackend.__module__ = __name__
    RlKernelAttentionBackend.__qualname__ = "RlKernelAttentionBackend"
    _RLK_ATTENTION_IMPL = RlKernelAttentionImpl
    _RLK_ATTENTION_BUILDER = RlKernelAttentionMetadataBuilder
    _RLK_ATTENTION_BACKEND = RlKernelAttentionBackend
    globals()["RlKernelAttentionImpl"] = RlKernelAttentionImpl
    globals()["RlKernelAttentionMetadataBuilder"] = RlKernelAttentionMetadataBuilder
    globals()["RlKernelAttentionBackend"] = RlKernelAttentionBackend
    # Override the platform-selected enum so vLLM keeps its native metadata and
    # cache update path while RL-Kernel owns the attention arithmetic call.
    register_backend(
        selected_backend,
        f"{__name__}.RlKernelAttentionBackend",
    )
    integration.record_installed_hook("attention", f"{__name__}.RlKernelAttentionBackend")


def install_vllm_integration(plan: IntegrationPlan) -> VllmIntegration:
    """Install vLLM paged Attention, Qwen dense FFN and sampler Logp routes."""

    existing = get_active_integration("vllm")
    if existing is not None:
        if not isinstance(existing, VllmIntegration):
            raise RuntimeError("active vLLM integration has an unexpected type")
        if existing.plan != plan:
            raise RuntimeError("vLLM integration is already installed with another plan")
        return existing
    integration = VllmIntegration(plan, rl_kernel_operators={})
    set_active_integration("vllm", integration)
    # vLLM imports this legacy rotary path even when every operator is routed
    # to production. FA4-only installations need the bundled compatibility
    # namespace before any attention backend is imported.
    if torch.version.hip is None:
        _install_flash_attn_ops_compatibility()
    strict_linear_logp = plan.implementation_for("logp", "rollout") is Implementation.RL_KERNEL
    strict_attention = plan.implementation_for("attention", "rollout") is Implementation.RL_KERNEL
    if strict_attention:
        _patch_qwen3_strict_model()
        _patch_rocm_weight_cache_refresh()
    _patch_qwen3_layer_alignment_diagnostics()
    if strict_linear_logp:
        _patch_qwen_lm_head_padding()
        _patch_strict_lm_head_linear()
        _patch_qwen_compute_logits(integration)

    # Patch every boundary so P/R cases also produce production readback. The
    # integration object chooses native versus RL-Kernel per module.
    _register_attention_backend(integration)
    _patch_qwen_ffn(integration)
    if torch.version.hip is not None:
        # Current ROCm deployments use vLLM's V2 runner and its worker sampler.
        _patch_worker_sampler(integration, strict_linear_logp=strict_linear_logp)
    else:
        _patch_sampler(integration, strict_linear_logp=strict_linear_logp)
    if strict_linear_logp:
        sampler_hook = (
            "vllm.v1.worker.gpu.sample.sampler.Sampler.__call__"
            if torch.version.hip is not None
            else "vllm.v1.sample.sampler.Sampler.forward"
        )
        integration.record_installed_hook(
            "logp",
            "vllm.model_executor.models.qwen3.Qwen3ForCausalLM.compute_logits," f"{sampler_hook}",
        )
    return integration


def register_vllm_plugin() -> None:
    """vLLM general-plugin entry point; inactive unless Vime exported a plan."""

    if os.getenv("RL_KERNEL_VLLM_INTEGRATION", "").strip() not in {"1", "true", "True"}:
        return
    install_vllm_integration(plan_from_environment())


__all__ = [
    "configure_vllm_environment",
    "install_vllm_integration",
    "plan_from_environment",
    "register_vllm_plugin",
]
