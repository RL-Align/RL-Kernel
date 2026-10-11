# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""VIME ModuleSpec and HF weight hooks for the explicit shared TP4 GDN provider.

    --spec rl_engine.integrations.orchestrators.vime.providers.qwen3_next_vime \
        get_qwen3_next_shared_spec

Installing this spec installs Qwen3-Next-only weight conversion hooks in this
training process. Other model families keep their existing conversion paths.
The decoder spec covers both attention families, routed/shared experts, and
decoder norms. Call install_vime_model_providers after constructing the GPT
model to replace its embedding, final norm and output layer as well.
"""

import re

import torch
from megatron.core import mpu
from megatron.core.tensor_parallel.layers import (
    set_defaults_if_not_set_tensor_model_parallel_attributes,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from torch import nn
from vime_plugins.models.hf_attention import HuggingfaceAttention

from rl_engine.integrations.engines.train.vllm.qwen3_next_provider import GDNProviderConfig
from rl_engine.models.qwen3_next.qwen3_next_engine_weights import (
    conv_hf_to_rank_major,
    conv_rank_major_to_hf,
    expert_hf_to_rank_major,
    expert_rank_major_to_hf,
    kv_hf_to_rank_major,
    kv_rank_major_to_hf,
    set_gdn_tp_metadata,
    validate_training_topology,
)
from rl_engine.models.qwen3_next.qwen3_next_tp import _CopyToTP
from rl_engine.models.qwen3_next.qwen3_next_tp_blocks import TP4FullAttention, TP4MoE
from rl_engine.models.qwen3_next.qwen3_next_tp_gdn import TP4GDN


class SharedInputNorm(nn.Module):
    def __init__(self, eps, device, weight=None):
        super().__init__()
        self.eps = eps
        if weight is not None and not isinstance(weight, nn.Parameter):
            raise ValueError("Shared decoder norm weight must be an existing Parameter")
        if weight is not None and (weight.shape != (2048,) or weight.dtype != torch.bfloat16):
            raise ValueError("Shared decoder norm requires a BF16 [2048] weight")
        self.weight = (
            weight
            if weight is not None
            else nn.Parameter(torch.zeros(2048, dtype=torch.bfloat16, device=device))
        )
        self.weight.tensor_model_parallel = False
        self.weight.partition_dim = -1
        self.weight.partition_stride = 1
        self.weight.sequence_parallel = False

    def forward(self, value):
        from rl_engine.backends.cuda.norm.rmsnorm import Qwen3NextRMSNormCudaOp

        return Qwen3NextRMSNormCudaOp()(value, self.weight, eps=self.eps)


class SharedVIMELMHead(nn.Module):
    """Megatron output-layer API with shared GEMM and the original weight name.

    A forward hook on this module observes BF16 logits before Float16Module
    widens the model output for VIME's policy loss.
    """

    def __init__(self, weight, group):
        super().__init__()
        if not isinstance(weight, nn.Parameter):
            raise ValueError("Shared TP4 LM head weight must be an existing Parameter")
        if weight.shape != (37984, 2048) or weight.dtype != torch.bfloat16:
            raise ValueError("Shared TP4 LM head requires BF16 [37984, 2048] weight")
        if torch.distributed.get_world_size(group) != 4:
            raise ValueError("Shared LM head requires TP4")
        self.weight = weight
        self.weight.tensor_model_parallel = True
        self.weight.partition_dim = 0
        self.weight.partition_stride = 1
        self.group = group
        self.gather_output = False
        self.sequence_parallel = False

    def forward(self, input_, weight=None, runtime_gather_output=None):
        from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

        from rl_engine.models.qwen3_next.qwen3_next_forward import shared_linear

        if self.sequence_parallel:
            raise ValueError("Shared LM head requires sequence parallelism disabled")
        selected = self.weight if weight is None else weight
        if selected.shape != self.weight.shape or selected.dtype != torch.bfloat16:
            raise ValueError("Shared LM head received an incompatible tied weight")
        value = shared_linear(_CopyToTP.apply(input_, self.group), selected)
        gather = self.gather_output if runtime_gather_output is None else runtime_gather_output
        if gather:
            value = gather_from_tensor_model_parallel_region(value, group=self.group)
        return value, None


class SharedVIMEGDN(HuggingfaceAttention):
    def __init__(
        self, args, config, layer_number, cp_comm_type="p2p", pg_collection=None, name=None
    ):
        validate_training_topology(config, args)
        super().__init__(args, config, layer_number, cp_comm_type, pg_collection)
        if self.hf_config.hidden_size != 2048:
            raise ValueError("Shared VIME GDN requires the pinned Qwen3-Next architecture")
        group = (
            pg_collection.tp if pg_collection is not None else mpu.get_tensor_model_parallel_group()
        )
        device = torch.device("cuda", torch.cuda.current_device())
        self.linear_attn = TP4GDN(
            group=group, device=device, config=GDNProviderConfig(), eps=self.hf_config.rms_norm_eps
        )
        set_gdn_tp_metadata(self.linear_attn)
        self.input_layernorm = SharedInputNorm(self.hf_config.rms_norm_eps, device)
        self.trace_state_hook = None
        self.subnode_hook = None

    def hf_forward(self, hidden_states, packed_seq_params):
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError("VIME packed GDN expects [1, tokens, 2048]")
        if packed_seq_params is None:
            raise ValueError("VIME GDN requires packed sequence boundaries")
        cu = packed_seq_params.cu_seqlens_q.to(device="cpu")
        hidden = self.input_layernorm(hidden_states[0])
        if self.subnode_hook is not None:
            self.subnode_hook("input_layernorm", hidden)
        indices = torch.arange(1, cu.numel(), device=hidden.device, dtype=torch.int32)
        initial = self.linear_attn.initial_state(cu.numel())
        output, state = self.linear_attn(hidden, initial, indices, cu)
        if self.trace_state_hook is not None:
            self.trace_state_hook(state, indices, cu)
        if self.subnode_hook is not None:
            self.subnode_hook("mixer", output)
        return output.unsqueeze(0)


class SharedVIMEFullAttention(HuggingfaceAttention):
    def __init__(
        self, args, config, layer_number, cp_comm_type="p2p", pg_collection=None, name=None
    ):
        validate_training_topology(config, args)
        super().__init__(args, config, layer_number, cp_comm_type, pg_collection)
        group = pg_collection.tp if pg_collection else mpu.get_tensor_model_parallel_group()
        device = torch.device("cuda", torch.cuda.current_device())
        rope = getattr(self.hf_config, "rope_parameters", None)
        if rope is None:
            rope = {"rope_theta": self.hf_config.rope_theta, "partial_rotary_factor": 0.25}
        if any(
            (
                rope.get("partial_rotary_factor") != 0.25,
                rope.get("rope_type", "default") != "default",
            )
        ):
            raise ValueError("Shared full attention requires default RoPE with rotary dimension 64")
        self.self_attn = TP4FullAttention(
            group=group,
            device=device,
            eps=self.hf_config.rms_norm_eps,
            rope_theta=rope["rope_theta"],
        )
        self.input_layernorm = SharedInputNorm(self.hf_config.rms_norm_eps, device)
        self.trace_state_hook = None
        self.subnode_hook = None
        for parameter in self.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(parameter)
            parameter.sequence_parallel = False

    def hf_forward(self, hidden_states, packed_seq_params):
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1 or packed_seq_params is None:
            raise ValueError("Shared VIME attention requires packed [1, tokens, 2048]")
        hidden = self.input_layernorm(hidden_states[0])
        if self.subnode_hook is not None:
            self.subnode_hook("input_layernorm", hidden)
        cu = packed_seq_params.cu_seqlens_q.to(device="cpu")
        indices = torch.arange(1, cu.numel(), device=hidden.device, dtype=torch.int32)
        output, state = self.self_attn(
            hidden, self.self_attn.initial_state(cu.numel()), indices, cu
        )
        if self.trace_state_hook is not None:
            self.trace_state_hook(state, indices, cu)
        if self.subnode_hook is not None:
            self.subnode_hook("mixer", output)
        return output.unsqueeze(0)


class SharedVIMEMoE(TP4MoE):
    def __init__(self, config, submodules=None):
        validate_training_topology(config, config)
        if config.expert_tensor_parallel_size != 4:
            raise ValueError("Shared MoE requires expert tensor parallel size 4")
        super().__init__(
            group=mpu.get_tensor_model_parallel_group(),
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        # Native VIME only recognizes stride-two FC1 names; our paired loader
        # already packs complete per-rank gate/up blocks before the generic split.
        self.experts.gate_up.partition_stride = 1
        for parameter in self.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(parameter)
            parameter.sequence_parallel = False

    def forward(self, hidden_states, padding_mask=None):
        if padding_mask is not None and bool(padding_mask.any()):
            raise ValueError("Shared VIME MoE requires packed tokens without padding")
        value = super().forward(hidden_states.reshape(-1, 2048))
        if self.subnode_hook is not None:
            self.subnode_hook("mlp", value)
        return value.reshape_as(hidden_states), None


class SharedDecoderNorm(SharedInputNorm):
    def __init__(self, config, hidden_size, eps):
        if hidden_size != 2048 or config.params_dtype != torch.bfloat16:
            raise ValueError("Shared decoder norm requires the official BF16 architecture")
        super().__init__(eps, torch.device("cuda", torch.cuda.current_device()))
        # Set by a trace owner: this instance is a layer's pre-MLP norm.
        self.subnode_hook = None

    def forward(self, value):
        output = super().forward(value)
        if self.subnode_hook is not None:
            self.subnode_hook("post_attention_layernorm", output.reshape(-1, output.shape[-1]))
        return output


class SharedVIMEEmbedding(nn.Module):
    def __init__(self, weight, group):
        super().__init__()
        if weight.shape != (37984, 2048) or weight.dtype != torch.bfloat16:
            raise ValueError("Shared embedding requires the official BF16 TP4 weight")
        self.weight = weight
        self.group = group
        self.subnode_hook = None

    def forward(self, tokens):
        from rl_engine.models.qwen3_next.qwen3_next_embedding import shared_embedding

        output = shared_embedding(tokens, self.weight, self.group)
        if self.subnode_hook is not None:
            self.subnode_hook("embedding", output)
        return output


def install_vime_model_providers(model):
    """Replace outer GPT providers while retaining each original parameter object."""
    group = mpu.get_tensor_model_parallel_group()
    validate_training_topology(model.config, model.config)
    if len(model.decoder.layers) != 48:
        raise ValueError("The complete provider requires all 48 decoder layers")
    for layer in model.decoder.layers:
        if not isinstance(layer.self_attention, (SharedVIMEGDN, SharedVIMEFullAttention)):
            raise ValueError("Construct the model with get_qwen3_next_shared_spec first")
        if not isinstance(layer.mlp, SharedVIMEMoE):
            raise ValueError("Shared decoder MoE provider is missing")
    model.embedding.word_embeddings = SharedVIMEEmbedding(
        model.embedding.word_embeddings.weight, group
    )
    previous = model.decoder.final_layernorm
    model.decoder.final_layernorm = SharedInputNorm(
        model.config.layernorm_epsilon, previous.weight.device, weight=previous.weight
    )
    model.output_layer = SharedVIMELMHead(model.output_layer.weight, group)
    return model


def install_vime_gdn_weight_hooks():
    """Install exact shared-model layouts in VIME's real load/export entrypoints."""
    import vime.backends.megatron_utils.hf_to_megatron as load_api
    import vime.backends.megatron_utils.megatron_to_hf as export_api
    from vime.backends.megatron_utils.hf_to_megatron.common import strip_mcore_wrappers

    original_load = load_api._LOADERS["qwen3_next"]
    if getattr(original_load, "_rl_kernel_tp4_gdn", False):
        return
    original_export = export_api.convert_qwen3_next_to_hf

    def layer_parts(name):
        match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", strip_mcore_wrappers(name))
        return match.groups() if match else (None, None)

    def direct_moe(rest):
        return rest in {
            "mlp.gate.weight",
            "mlp.shared_expert_gate.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert.down_proj.weight",
        }

    def load(name, reader, config):
        layer, rest = layer_parts(name)
        prefix = f"model.layers.{layer}"
        if rest == "mlp.experts.gate_up":
            value = torch.empty((512, 1024, 2048), dtype=torch.bfloat16)
            for expert in range(512):
                gate, up = (
                    reader.get_tensor(f"{prefix}.mlp.experts.{expert}.{part}_proj.weight")
                    for part in ("gate", "up")
                )
                value[expert].copy_(expert_hf_to_rank_major(gate, up))
            return value
        if rest == "mlp.experts.down":
            value = torch.empty((512, 2048, 512), dtype=torch.bfloat16)
            for expert in range(512):
                weight = reader.get_tensor(f"{prefix}.mlp.experts.{expert}.down_proj.weight")
                if weight.shape != (2048, 512) or weight.dtype != torch.bfloat16:
                    raise ValueError("Official expert down projection requires BF16 [2048, 512]")
                value[expert].copy_(weight)
            return value
        if direct_moe(rest):
            return reader.get_tensor(f"{prefix}.{rest}")
        value = original_load(name, reader, config)
        if name.endswith("self_attention.linear_attn.conv1d.weight"):
            if mpu.get_tensor_model_parallel_world_size() != 4:
                raise ValueError("The shared GDN weight loader requires TP4")
            value = conv_hf_to_rank_major(value)
        if rest in (
            "self_attention.self_attn.k_proj.weight",
            "self_attention.self_attn.v_proj.weight",
        ):
            value = kv_hf_to_rank_major(value)
        return value

    def export(args, name, value):
        layer, rest = layer_parts(name)
        prefix = f"model.layers.{layer}"
        if rest == "mlp.experts.gate_up":
            if value.shape != (512, 1024, 2048):
                raise ValueError("Gathered routed gate/up shape is not the complete TP4 tensor")
            return [
                (f"{prefix}.mlp.experts.{expert}.{part}_proj.weight", weight)
                for expert in range(512)
                for part, weight in zip(("gate", "up"), expert_rank_major_to_hf(value[expert]))
            ]
        if rest == "mlp.experts.down":
            if value.shape != (512, 2048, 512) or value.dtype != torch.bfloat16:
                raise ValueError("Gathered routed down shape is not the complete TP4 tensor")
            return [
                (f"{prefix}.mlp.experts.{expert}.down_proj.weight", value[expert])
                for expert in range(512)
            ]
        if direct_moe(rest):
            return [(f"{prefix}.{rest}", value)]
        if name.endswith("self_attention.linear_attn.conv1d.weight"):
            if mpu.get_tensor_model_parallel_world_size() != 4:
                raise ValueError("The shared GDN weight exporter requires TP4")
            value = conv_rank_major_to_hf(value)
        if rest in (
            "self_attention.self_attn.k_proj.weight",
            "self_attention.self_attn.v_proj.weight",
        ):
            value = kv_rank_major_to_hf(value)
        return original_export(args, name, value)

    load._rl_kernel_tp4_gdn = True
    load_api._LOADERS["qwen3_next"] = load
    export_api.convert_qwen3_next_to_hf = export


def get_qwen3_next_shared_spec(args, config, vp_stage=None):
    """Use the explicit provider for every decoder normalization and sublayer."""
    from vime_plugins.models.qwen3_next import Attention, get_qwen3_next_spec

    validate_training_topology(config, args)
    if config.expert_tensor_parallel_size != 4:
        raise ValueError("Shared model requires expert tensor parallel size 4")
    if vp_stage is not None:
        raise ValueError("The shared GDN profile does not use virtual pipeline stages")
    spec = get_qwen3_next_spec(args, config, vp_stage)
    replaced = 0
    for layer in spec.layer_specs:
        if layer.submodules.self_attention.module is Attention:
            layer.submodules.self_attention = ModuleSpec(
                module=SharedVIMEGDN, params={"args": args}
            )
            replaced += 1
        else:
            layer.submodules.self_attention = ModuleSpec(
                module=SharedVIMEFullAttention, params={"args": args}
            )
        layer.submodules.input_layernorm = IdentityOp
        layer.submodules.pre_mlp_layernorm = SharedDecoderNorm
        layer.submodules.mlp = ModuleSpec(module=SharedVIMEMoE)
        # The native TE sharded-state map refers to fused norm/projection names.
        # Shared adapters retain explicit HF-compatible norm parameter names.
        layer.submodules.sharded_state_dict_keys_map = {}
    if replaced != 36:
        raise ValueError(f"Pinned Qwen3-Next must contain 36 GDN layers, found {replaced}")
    if len(spec.layer_specs) != 48:
        raise ValueError("Pinned Qwen3-Next must contain all 48 decoder layers")
    spec.layer_norm = SharedDecoderNorm
    install_vime_gdn_weight_hooks()
    return spec
