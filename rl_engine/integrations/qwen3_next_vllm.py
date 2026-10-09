# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Actual vLLM GDN layer/cache adapter for the shared TP4 provider.

The CUSTOM backend deliberately does not advertise batch invariance. Registering
this adapter cannot bypass vLLM's strict capability check. Real metadata/step
acceptance must pass before a separately reviewed capability registration.
"""

import torch
from torch import nn
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum, register_backend

from rl_engine.integrations.qwen3_next_provider import GDNProviderConfig, GDNState
from rl_engine.integrations.qwen3_next_tp import _ReduceFromTP
from rl_engine.integrations.qwen3_next_tp_gdn import TP4GDN


class SharedGDNMetadataBuilder(GDNAttentionMetadataBuilder):
    """Use vLLM's scheduler ordering and record resets even for fresh one-token prompts."""

    _cudagraph_support = AttentionCGSupport.NEVER

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        if common_prefix_len:
            raise ValueError("Shared GDN does not enable prefix reuse")
        metadata = super().build(common_prefix_len, common_attn_metadata, **kwargs)
        if metadata.num_spec_decodes or metadata.spec_sequence_masks is not None:
            raise ValueError("Shared GDN does not enable speculative decoding")
        metadata.has_initial_state = common_attn_metadata.compute_num_computed_tokens() > 0
        return metadata


class SharedGDNBackend(GDNAttentionBackend):
    @staticmethod
    def get_name():
        return "RL_KERNEL_QWEN3_NEXT_GDN"

    @staticmethod
    def get_builder_cls():
        return SharedGDNMetadataBuilder

    @staticmethod
    def supports_batch_invariance():
        from rl_engine.integrations.qwen3_next_profile import profile_is_certified

        return profile_is_certified()


def metadata_sequences(metadata, tokens):
    """Extract active packed sequences, excluding zero-token padding without touching null state."""
    if not isinstance(metadata, GDNAttentionMetadata):
        raise ValueError("Shared GDN requires actual vLLM GDNAttentionMetadata")
    if metadata.num_spec_decodes or metadata.spec_sequence_masks is not None:
        raise ValueError("Speculative GDN metadata is outside the shared profile")
    cu = metadata.non_spec_query_start_loc
    indices = metadata.non_spec_state_indices_tensor
    if cu is None or indices is None:
        raise ValueError("GDN metadata lacks packed sequence boundaries or state slots")
    cu = cu.to(device="cpu", dtype=torch.int64)
    if cu.ndim != 1 or cu.numel() != indices.numel() + 1:
        raise ValueError("GDN metadata boundaries and slot counts disagree")
    lengths = cu.diff()
    if cu[0] != 0 or cu[-1] != tokens or bool((lengths < 0).any()):
        raise ValueError("GDN metadata boundaries must cover exactly the actual tokens")
    active = torch.nonzero(lengths > 0, as_tuple=True)[0]
    chosen = indices.index_select(0, active.to(indices.device))
    if bool((chosen <= 0).any()) or chosen.unique().numel() != chosen.numel():
        raise ValueError("Active GDN requests require distinct positive state slots")
    compact_cu = torch.cat((torch.zeros(1, dtype=torch.int64), lengths[active].cumsum(0)))
    has_initial = metadata.has_initial_state
    if has_initial is None or has_initial.numel() != indices.numel():
        raise ValueError("Shared GDN metadata must declare initial state for every request")
    reset = chosen[
        ~has_initial.to(device=indices.device, dtype=torch.bool)[active.to(indices.device)]
    ]
    return chosen, compact_cu, reset


def _register_backend():
    path = "rl_engine.integrations.qwen3_next_vllm.SharedGDNBackend"
    try:
        current = MambaAttentionBackendEnum.CUSTOM.get_path()
    except ValueError:
        current = None
    if current is not None and current != path:
        raise RuntimeError("A different custom Mamba backend is already registered")
    register_backend(MambaAttentionBackendEnum.CUSTOM, path, is_mamba=True)


class SharedQwenGDN(QwenGatedDeltaNetAttention):
    """Preserve real vLLM HF weight loaders and bind shared forward to its owned cache."""

    def __init__(self, config, vllm_config, prefix="", **kwargs):
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 4 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Shared vLLM GDN requires TP4 and PP1")
        if getattr(parallel, "decode_context_parallel_size", 1) != 1:
            raise ValueError("Shared vLLM GDN requires CP1")
        if getattr(parallel, "prefill_context_parallel_size", 1) != 1:
            raise ValueError("Shared vLLM GDN requires prefill CP1")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("Shared vLLM GDN requires BF16 weights")
        if vllm_config.quant_config is not None or vllm_config.speculative_config is not None:
            raise ValueError("Shared GDN requires unquantized weights and no speculation")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Shared GDN requires prefix reuse disabled")
        if vllm_config.cache_config.mamba_cache_mode != "none":
            raise ValueError("Shared GDN requires the non-reusing mamba cache mode")
        if kwargs.get("reduce_results", True) is not True:
            raise ValueError("Shared GDN performs its own TP output reduction")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("Shared GDN currently requires eager execution")
        if not kwargs.get("gqa_interleaved_layout", False) or config.hidden_size != 2048:
            raise ValueError("Shared GDN requires Qwen3-Next interleaved TP4 projections")
        _register_backend()
        super().__init__(config, vllm_config, prefix, **kwargs)
        if self.get_state_dtype() != (torch.bfloat16, torch.float32):
            raise ValueError("Shared GDN requires BF16 convolution and FP32 recurrent state")
        # HF stores A_log and dt_bias in BF16; upstream widens A_log at allocation.
        # Keep the checkpoint weight identity, widening only at the recurrence call.
        for name in ("A_log", "dt_bias"):
            original = getattr(self, name)
            parameter = nn.Parameter(original.to(dtype=torch.bfloat16), requires_grad=False)
            parameter.weight_loader = original.weight_loader
            setattr(self, name, parameter)
        self.group = get_tp_group().device_group
        self.provider_config = GDNProviderConfig()
        self.eps = config.rms_norm_eps

    @property
    def mamba_type(self):
        return MambaAttentionBackendEnum.CUSTOM

    @torch.no_grad()
    def forward(self, hidden_states):
        """Match the installed vLLM layer contract: one input, one returned tensor."""
        output = torch.empty_like(hidden_states)
        self._forward_into(hidden_states, output)
        return output

    @torch.no_grad()
    def _forward_into(self, hidden_states, output):
        context = get_forward_context()
        metadata = context.attn_metadata
        metadata = metadata.get(self.prefix) if isinstance(metadata, dict) else None
        if metadata is None:
            # vLLM's pre-cache memory profiling has no request or persistent state.
            from rl_engine.integrations.qwen3_next_forward import shared_linear

            qkvz = shared_linear(hidden_states, self.in_proj_qkvz.weight)
            ba = shared_linear(hidden_states, self.in_proj_ba.weight)
            zeros = qkvz.new_zeros((ba.shape[0], 1024))
            projected = shared_linear(zeros, self.out_proj.weight)
            output.copy_(_ReduceFromTP.apply(projected, self.group))
            del qkvz, ba
            return
        tokens = metadata.num_actual_tokens
        if tokens > hidden_states.shape[0] or output.shape != hidden_states.shape:
            raise ValueError("vLLM GDN buffers do not cover the actual tokens")
        indices, cu, reset = metadata_sequences(metadata, tokens)
        output.zero_()
        if not tokens:
            return
        conv = self.kv_cache[0]
        if not is_conv_state_dim_first():
            conv = conv.transpose(-1, -2)
        recurrent = self.kv_cache[1]
        if reset.numel():
            state = GDNState(conv.clone(), recurrent.clone())
            state.convolution.index_fill_(0, reset.long(), 0)
            state.recurrent.index_fill_(0, reset.long(), 0)
        else:
            state = GDNState(conv, recurrent)
        value, next_state = TP4GDN.forward(self, hidden_states[:tokens], state, indices, cu)
        trace_state_hook = getattr(self, "trace_state_hook", None)
        if trace_state_hook is not None:
            trace_state_hook(next_state, indices, cu)
        output[:tokens].copy_(value)
        conv.copy_(next_state.convolution)
        recurrent.copy_(next_state.recurrent)


def register_shared_gdn_adapter():
    """Opt in to the actual GDN layer adapter, without claiming strict capability."""
    from vllm.model_executor.custom_op import op_registry_oot

    _register_backend()
    existing = op_registry_oot.get("QwenGatedDeltaNetAttention")
    if existing is SharedQwenGDN:
        return
    if existing is not None:
        raise RuntimeError("A different out-of-tree Qwen GDN layer is already registered")
    QwenGatedDeltaNetAttention.register_oot(SharedQwenGDN)
