# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Opt-in vLLM model using shared Qwen3-Next arithmetic and native weight loaders.

Construction preserves every native Parameter and loader. The existing modules
are bound to subclasses before weights load; no second copy of the 80B model is
allocated. Only the selected eager profile is supported. Registering it does not
declare scheduler batch invariance or completion of full-model L2 acceptance.
"""

from importlib.metadata import version

import torch
import torch.nn.functional as F
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextForCausalLM,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from rl_engine.contracts.operators.logprob import LogprobRole
from rl_engine.distributed.algorithms.collectives import collective_for_group
from rl_engine.models.qwen3_next.qwen3_next_ep import expert_map, shared_moe_ep
from rl_engine.models.qwen3_next.qwen3_next_forward import (
    FORWARD_PROVIDER_ID,
    shared_linear,
    shared_logprob,
    shared_moe,
)
from rl_engine.models.qwen3_next.qwen3_next_logprob import qwen_logprob_contract
from rl_engine.models.qwen3_next.qwen3_next_tp import _ReduceFromTP
from rl_engine.models.qwen3_next.qwen3_next_tp_blocks import TP4FullAttention

ARCHITECTURE = "RLKernelQwen3NextForCausalLM"


def profile_capabilities():
    return {
        "provider": FORWARD_PROVIDER_ID,
        "architecture": ARCHITECTURE,
        "layers": 48,
        "tensor_parallel_size": 4,
        "scheduler_batch_invariance_verified": False,
        "full_model_l2_verified": False,
    }


def shared_expert_parallel(vllm_config):
    """EP size from the shared identity's topology (1 when there is no identity)."""
    identity = (vllm_config.additional_config or {}).get("qwen3_next_shared_identity")
    ep = 1 if identity is None else identity["topology"]["ep"]
    if ep not in (1, 4):
        raise ValueError("Shared model supports EP 1 or EP 4 over the TP4 group")
    return ep


def validate_profile(vllm_config):
    if version("vllm") != "0.30.0":
        raise ValueError("Shared model requires pinned vLLM 0.30.0")
    model, parallel = vllm_config.model_config, vllm_config.parallel_config
    config = model.hf_text_config
    required = {
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 151936,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144,
    }
    for name, expected in required.items():
        if getattr(config, name, None) != expected:
            raise ValueError(f"Shared model requires official {name}={expected}")
    expected_types = [
        "full_attention" if index % 4 == 3 else "linear_attention" for index in range(48)
    ]
    if list(config.layer_types) != expected_types:
        raise ValueError("Shared model requires the official 36 GDN / 12 full-attention layers")
    rope = config.rope_parameters
    if (
        rope.get("rope_theta") != 10000000
        or rope.get("partial_rotary_factor") != 0.25
        or rope.get("rope_type", "default") != "default"
    ):
        raise ValueError("Shared model requires default theta=1e7 and rotary dimension 64")
    topology = tuple(
        getattr(parallel, name, 1)
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    )
    if topology != (4, 1, 1, 1, 1):
        raise ValueError("Shared model requires TP4/PP1/DP1/CP1")
    unsupported = (
        model.dtype != torch.bfloat16,
        not model.enforce_eager,
        vllm_config.quant_config is not None,
        vllm_config.speculative_config is not None,
        vllm_config.lora_config is not None,
        vllm_config.cache_config.enable_prefix_caching,
        parallel.enable_expert_parallel != (shared_expert_parallel(vllm_config) == 4),
        getattr(parallel, "expert_placement_strategy", "linear") != "linear",
        parallel.enable_eplb,
        parallel.use_sequence_parallel_moe,
        parallel.enable_dbo,
        getattr(config, "attention_dropout", 0) != 0,
        getattr(config, "mlp_only_layers", []),
        getattr(config, "decoder_sparse_step", 1) != 1,
        getattr(config, "layer_scale", False),
        not getattr(config, "norm_topk_prob", True),
        getattr(config, "hidden_act", "silu") != "silu",
        getattr(config, "qkv_bias", False),
        not getattr(config, "attn_output_gate", True),
        not getattr(config, "is_causal", True),
        getattr(config, "tie_word_embeddings", False),
    )
    if any(unsupported):
        raise ValueError("Unsupported extension to the fixed BF16 eager shared model profile")
    # This selects the native plain-weight allocation, not the execution path.
    # The shared MoE below owns all forward arithmetic and prevents repacking.
    if vllm_config.kernel_config.moe_backend != "triton":
        raise ValueError("Set kernel_config.moe_backend='triton' for plain native BF16 allocation")
    if vllm_config.attention_config.backend != AttentionBackendEnum.CUSTOM:
        raise ValueError("Set attention_config.backend='CUSTOM' for the shared paged-KV backend")


def _rms(value, module):
    from rl_engine.backends.cuda.norm.rmsnorm import Qwen3NextRMSNormCudaOp

    return Qwen3NextRMSNormCudaOp()(value, module.weight, eps=1e-6)


def _require_weight(value, shape, name):
    if value.dtype != torch.bfloat16 or value.shape != shape or not value.is_contiguous():
        raise ValueError(f"{name} must remain contiguous BF16 {shape}")


class SharedPlainBF16MoEMethod(UnquantizedFusedMoEMethod):
    """Retain HF-loaded plain tensors; the shared model never calls native MoE."""

    def process_weights_after_loading(self, layer):
        if layer.expert_map is None:
            _require_weight(layer.w13_weight, (512, 256, 2048), "routed gate/up")
            _require_weight(layer.w2_weight, (512, 2048, 128), "routed down")
        else:
            _require_weight(layer.w13_weight, (128, 1024, 2048), "EP4 routed gate/up")
            _require_weight(layer.w2_weight, (128, 2048, 512), "EP4 routed down")
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    def apply(self, *args, **kwargs):
        raise RuntimeError("Use SharedVLLMMoE; native MoE arithmetic is outside this provider")

    def apply_monolithic(self, *args, **kwargs):
        raise RuntimeError("Use SharedVLLMMoE; native MoE arithmetic is outside this provider")


class SharedVLLMAttention(Qwen3NextAttention):
    def forward(self, positions, hidden_states):
        if positions.ndim != 1 or positions.shape[0] != hidden_states.shape[0]:
            raise ValueError("Shared text attention requires one absolute position per token")
        if bool(((positions < 0) | (positions >= 262144)).any()):
            raise ValueError("Attention positions exceed the official RoPE range")
        # Keep the three GEMM invocations identical to the training TP4 module;
        # native qkv_proj owns one packed allocation and all three HF loaders.
        q_weight, k_weight, v_weight = self.qkv_proj.weight.split((2048, 256, 256))
        query, gate = shared_linear(hidden_states, q_weight).reshape(-1, 4, 512).chunk(2, -1)
        key = shared_linear(hidden_states, k_weight).reshape(-1, 1, 256)
        value = shared_linear(hidden_states, v_weight)
        query, key = _rms(query, self.q_norm), _rms(key, self.k_norm)
        query = TP4FullAttention._rotate(self, query, positions).reshape(-1, 1024)
        key = TP4FullAttention._rotate(self, key, positions).reshape(-1, 256)
        attended = self.attn(query, key, value).reshape(-1, 4, 256) * gate.sigmoid()
        local = shared_linear(attended.reshape(-1, 1024), self.o_proj.weight)
        return _ReduceFromTP.apply(local, self.shared_tp_group)


class SharedVLLMMoE(Qwen3NextSparseMoeBlock):
    subnode_hook = None

    def forward(self, hidden_states, already_sequence_parallel=False):
        if already_sequence_parallel:
            raise ValueError("Shared MoE does not enable sequence parallelism")
        routed = self.experts.routed_experts
        ep = getattr(self, "shared_ep", 1)
        if ep == 4:
            output, routes = shared_moe_ep(
                hidden_states,
                self.gate.weight,
                routed.w13_weight,
                routed.w2_weight,
                group=self.shared_tp_group,
            )
        else:
            output, routes = shared_moe(
                hidden_states, self.gate.weight, routed.w13_weight, routed.w2_weight
            )
        if self.subnode_hook is not None:
            self.subnode_hook("router", routes.indices)
        gate_weight, up_weight = self.shared_expert.gate_up_proj.weight.chunk(2, dim=0)
        gate = shared_linear(hidden_states, gate_weight)
        up = shared_linear(hidden_states, up_weight)
        activated = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        shared = shared_linear(activated, self.shared_expert.down_proj.weight)
        shared_gate = shared_linear(hidden_states, self.shared_expert_gate.weight).sigmoid()
        if ep == 4:
            # Routed output is complete on every rank; only the shared expert is TP.
            return output + _ReduceFromTP.apply(shared * shared_gate, self.shared_tp_group)
        return _ReduceFromTP.apply(output + shared * shared_gate, self.shared_tp_group)


class SharedVLLMDecoderLayer(Qwen3NextDecoderLayer):
    # Set by a trace owner to observe the sub-node outputs named in
    # qwen3_next_subnode.SUBNODES; observing never changes a value.
    subnode_hook = None

    def forward(self, hidden_states, residual, positions, **kwargs):
        trace = self.subnode_hook
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = _rms(residual, self.input_layernorm)
        if trace is not None:
            trace("input_layernorm", hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        if trace is not None:
            trace("mixer", hidden_states)
        residual = residual + hidden_states
        hidden_states = _rms(residual, self.post_attention_layernorm)
        if trace is not None:
            trace("post_attention_layernorm", hidden_states)
        hidden_states = self.mlp(hidden_states)
        if trace is not None:
            trace("mlp", hidden_states)
        return hidden_states, residual


class SharedVLLMModel(Qwen3NextModel):
    subnode_hook = None

    def embed_input_ids(self, input_ids):
        from rl_engine.models.qwen3_next.qwen3_next_embedding import shared_embedding

        if input_ids.ndim != 1 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Shared embedding requires a packed integer token vector")
        return shared_embedding(input_ids.long(), self.embed_tokens.weight, self.shared_tp_group)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None or inputs_embeds is not None:
            raise ValueError("Shared text-only PP1 profile takes token IDs directly")
        if self.shared_forward_hook is not None:
            self.shared_forward_hook(input_ids, positions)
        hidden_states = self.embed_input_ids(input_ids)
        if self.subnode_hook is not None:
            self.subnode_hook("embedding", hidden_states)
        residual = None
        for index, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, residual, positions)
            if self.shared_trace_hook is not None:
                self.shared_trace_hook(index, hidden_states + residual, layer)
        return _rms(hidden_states + residual, self.norm)


def _bind_native_model(model, group, ep=1):
    """Attach shared behavior without replacing parameters or their HF loaders."""
    from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm import SharedQwenGDN
    from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm_attention import (
        SharedAttentionBackend,
    )

    if len(model.layers) != 48 or (model.start_layer, model.end_layer) != (0, 48):
        raise ValueError("The native model must own all 48 layers under PP1")
    _require_weight(model.embed_tokens.weight, (37984, 2048), "embedding")
    _require_weight(model.norm.weight, (2048,), "final norm")
    for layer in model.layers:
        layer.__class__ = SharedVLLMDecoderLayer
        if layer.layer_type == "linear_attention":
            if not isinstance(layer.linear_attn, SharedQwenGDN):
                raise ValueError("Native GDN was not constructed with the shared cache adapter")
        else:
            attention = layer.self_attn
            if attention.attn.attn_backend is not SharedAttentionBackend:
                raise ValueError("Full attention must use the explicit shared paged-KV backend")
            _require_weight(attention.qkv_proj.weight, (2560, 2048), "attention QKV/gate")
            _require_weight(attention.o_proj.weight, (2048, 1024), "attention output")
            attention.__class__ = SharedVLLMAttention
            attention.shared_tp_group = group
            # TP4FullAttention._rotate derives FP32 inverse frequencies from this
            # float on every call, so both engines rotate with identical tables.
            attention.rope_theta = 10000000.0
        moe = layer.mlp
        if (
            not isinstance(moe, Qwen3NextSparseMoeBlock)
            or moe.shared_expert is None
            or moe.is_fused_shared_expert_enabled
            or moe.replicate_shared_expert
        ):
            raise ValueError(
                "Every layer requires all512 unfused TP experts plus a sharded shared expert"
            )
        routed = moe.experts.routed_experts
        if type(routed.quant_method) is not UnquantizedFusedMoEMethod:
            raise ValueError("Shared MoE requires native unquantized BF16 parameter loaders")
        if ep == 1:
            if routed.expert_map is not None:
                raise ValueError("EP1 shared MoE must hold all 512 experts")
            _require_weight(routed.w13_weight, (512, 256, 2048), "routed gate/up")
            _require_weight(routed.w2_weight, (512, 2048, 128), "routed down")
        else:
            rank = torch.distributed.get_rank(group)
            if (routed.global_num_experts, routed.local_num_experts) != (512, 128) or not (
                torch.equal(routed.expert_map.cpu().int(), expert_map(rank, "cpu"))
            ):
                raise ValueError("EP4 shared MoE requires experts [128r, 128r+128) on rank r")
            _require_weight(routed.w13_weight, (128, 1024, 2048), "EP4 routed gate/up")
            _require_weight(routed.w2_weight, (128, 2048, 512), "EP4 routed down")
        _require_weight(moe.shared_expert.gate_up_proj.weight, (256, 2048), "shared gate/up")
        _require_weight(moe.shared_expert.down_proj.weight, (2048, 128), "shared down")
        routed.quant_method.__class__ = SharedPlainBF16MoEMethod
        moe.__class__ = SharedVLLMMoE
        moe.shared_tp_group, moe.shared_ep = group, ep
    model.__class__ = SharedVLLMModel
    model.shared_tp_group, model.shared_trace_hook = group, None
    model.shared_forward_hook = None


class RLKernelQwen3NextForCausalLM(Qwen3NextForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        validate_profile(vllm_config)
        from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm import (
            register_shared_gdn_adapter,
        )
        from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm_attention import (
            register_shared_attention_backend,
        )

        register_shared_gdn_adapter()
        register_shared_attention_backend()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.shared_tp_group = get_tp_group().device_group
        _bind_native_model(self.model, self.shared_tp_group, shared_expert_parallel(vllm_config))
        _require_weight(self.lm_head.weight, (37984, 2048), "LM head")
        self.shared_logits_hook = None

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs
    ):
        # Call the explicit eager method instead of the native compile wrapper.
        return self.model.forward(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_local_logits(self, hidden_states):
        return shared_linear(hidden_states, self.lm_head.weight)

    def compute_logits(self, hidden_states):
        local = self.compute_local_logits(hidden_states)
        if not torch.isfinite(local).all():
            raise ValueError("Shared model produced non-finite logits")
        source = local.t().contiguous()
        collective = collective_for_group(
            self.shared_tp_group, min_size_bytes=source.numel() * source.element_size()
        )
        logits = collective.all_gather(source).t().contiguous()
        if self.shared_logits_hook is not None:
            self.shared_logits_hook(hidden_states, local, logits)
        return logits

    def compute_selected_logprobs(self, hidden_states, targets):
        """The L2/fixture path; native sampler logprobs are not provider evidence."""
        local = self.compute_local_logits(hidden_states)
        contract = qwen_logprob_contract(
            local,
            rank=get_tp_group().rank_in_group,
            world_size=4,
            role=LogprobRole.INFER,
        )
        return shared_logprob(local, targets, contract=contract, tp_group=self.shared_tp_group)[0]


def register_shared_model():
    """Call in every vLLM process before constructing the explicit architecture."""
    from vllm.model_executor.models import ModelRegistry

    path = (
        "rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm_model"
        ":RLKernelQwen3NextForCausalLM"
    )
    if ARCHITECTURE in ModelRegistry.get_supported_archs():
        if ModelRegistry.models[ARCHITECTURE].load_model_cls() is not RLKernelQwen3NextForCausalLM:
            raise ValueError(
                "A different model is already registered under the shared architecture"
            )
    else:
        ModelRegistry.register_model(ARCHITECTURE, path)
    return ARCHITECTURE
