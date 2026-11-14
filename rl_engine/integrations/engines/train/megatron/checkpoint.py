# SPDX-License-Identifier: Apache-2.0
"""Explicit TP/PP/EP/CP=1 original DSV4 checkpoint importer for native MCore.

Frozen quantized bases remain FP4/FP8 plus original scales. BF16/FP32 native
parameters (head, norms, mHC, router, compressors) retain their source dtype.
This initial eager importer is intentionally separate from Bridge's BF16 import.
"""

from pathlib import Path

import torch
from safetensors import safe_open

from rl_engine.backends.rocm.gemm.checkpoint_fp8 import CheckpointFP8Group, CheckpointFP8Linear
from rl_engine.backends.rocm.moe.expert_training import RocmExpertTrainingBackend
from rl_engine.backends.rocm.moe.packed_mx import RocmPackedMXBackend
from rl_engine.backends.rocm.moe.route_weight_grad import RocmRouteWeightGradient
from rl_engine.backends.rocm.norm.qnorm_fp8 import FusedQNorm
from rl_engine.contracts.operators.expert_training import ExpertTrainingConfig, PackedExpertWeight
from rl_engine.integrations.engines.train.megatron.native_group_projection import (
    bind_grouped_projection,
)
from rl_engine.integrations.engines.train.megatron.native_rotation import bind_indexer_rotation
from rl_engine.ops.moe.expert_training import routed_expert_training


class _GateUp(torch.nn.Module):
    def __init__(self, gate, up):
        super().__init__()
        self.gate = gate
        self.up = up

    def forward(self, x, **kwargs):
        return torch.cat((self.gate(x)[0], self.up(x)[0]), dim=-1), None

    def backward_dw(self):
        pass


class NativePackedExperts(torch.nn.Module):
    def __init__(self, w1, w2, config):
        super().__init__()
        self.register_buffer("w1_codes", w1.codes)
        self.register_buffer("w1_scales", w1.scales)
        self.register_buffer("w2_codes", w2.codes)
        self.register_buffer("w2_scales", w2.scales)
        self.config = ExpertTrainingConfig(
            config.hidden_size, config.moe_ffn_hidden_size, config.num_moe_experts, mode="base_only"
        )
        self.backend = RocmExpertTrainingBackend(
            base=RocmPackedMXBackend(), route_gradient=RocmRouteWeightGradient()
        )

    def forward(self, x, tokens_per_expert, permuted_probs, **kwargs):
        if kwargs:
            raise ValueError("packed expert requires eager ordinary dispatch buffers")
        offsets = torch.cat(
            (
                torch.zeros(1, device=x.device, dtype=torch.int64),
                tokens_per_expert.to(device=x.device, dtype=torch.int64).cumsum(0),
            )
        )
        w1 = PackedExpertWeight(
            self.w1_codes,
            self.w1_scales,
            (self.config.experts, 2 * self.config.ffn, self.config.hidden),
        )
        w2 = PackedExpertWeight(
            self.w2_codes,
            self.w2_scales,
            (self.config.experts, self.config.hidden, self.config.ffn),
        )
        y = routed_expert_training(
            self.config,
            self.backend,
            x.contiguous(),
            permuted_probs.reshape(-1).float(),
            offsets,
            w1,
            w2,
            (None,) * 4,
        )
        return y, None

    def backward_dw(self):
        pass


def load_native_checkpoint(model, directory):
    cfg = model.config
    if any(
        getattr(cfg, n) != 1
        for n in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "expert_model_parallel_size",
            "context_parallel_size",
        )
    ):
        raise ValueError("initial packed checkpoint importer requires TP=PP=EP=CP=1")
    hf_config = None
    if getattr(model, "_rlk_original_qkv_codec", False):
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(directory)
    files = list(Path(directory).glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError("initial packed importer requires a single four-layer safetensors file")
    source = safe_open(str(files[0]), framework="pt")
    keys = set(source.keys())
    loaded = set()
    device = next(model.parameters()).device
    export = []
    model._rlk_checkpoint_export = export
    model._rlk_checkpoint_export_parameters = {}

    def get(key):
        loaded.add(key)
        return source.get_tensor(key).to(device)

    def copy(dest, key):
        value = get(key)
        if value.dtype not in (torch.bfloat16, torch.float32, torch.int32, torch.int64):
            raise TypeError(f"quantized source requires a packed adapter: {key}")
        if tuple(dest.shape) != tuple(value.shape):
            raise ValueError(f"shape mismatch: {key}: {value.shape} -> {dest.shape}")
        with torch.no_grad():
            dest.copy_(value)
        export.append((key, dest))
        if isinstance(dest, torch.nn.Parameter):
            model._rlk_checkpoint_export_parameters[key] = (dest,)

    def fp8(prefix):
        return CheckpointFP8Linear(get(prefix + ".weight"), get(prefix + ".scale"))

    def hc(target, prefix):
        copy(target.mapping_proj.weight, prefix + "_fn")
        copy(target.bias, prefix + "_base")
        scale = get(prefix + "_scale")
        export.append(
            (
                prefix + "_scale",
                lambda target=target: torch.cat(
                    [target.alpha_pre, target.alpha_post, target.alpha_res]
                ),
            )
        )
        model._rlk_checkpoint_export_parameters[prefix + "_scale"] = (
            target.alpha_pre,
            target.alpha_post,
            target.alpha_res,
        )
        with torch.no_grad():
            for i, n in enumerate(("alpha_pre", "alpha_post", "alpha_res")):
                getattr(target, n).copy_(scale[i : i + 1])

    def compressor(target, prefix):
        for n in ("wkv", "wgate"):
            copy(getattr(target, "linear_" + n).weight, prefix + "." + n + ".weight")
        copy(target.ape, prefix + ".ape")
        copy(target.norm.weight, prefix + ".norm.weight")

    copy(model.embedding.word_embeddings.weight, "embed.weight")
    copy(model.output_layer.weight, "head.weight")
    for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        copy(getattr(model.decoder, n), n)
    copy(model.decoder.final_norm.weight, "norm.weight")
    for i in range(len(model.decoder.layers) // 2):
        prefix = f"layers.{i}"
        attn_layer, moe_layer = model.decoder.layers[2 * i : 2 * i + 2]
        hc(attn_layer.hyper_connection, prefix + ".hc_attn")
        hc(moe_layer.hyper_connection, prefix + ".hc_ffn")
        attn = attn_layer.inner_layer.self_attention
        copy(attn_layer.inner_layer.input_layernorm.weight, prefix + ".attn_norm.weight")
        for name, key in (
            ("linear_q_down_proj", "wq_a"),
            ("linear_q_up_proj", "wq_b"),
            ("linear_kv_proj", "wkv"),
            ("linear_proj", "wo_b"),
        ):
            setattr(attn, name, fp8(prefix + ".attn." + key))
        copy(attn.q_layernorm.weight, prefix + ".attn.q_norm.weight")
        copy(attn.kv_layernorm.weight, prefix + ".attn.kv_norm.weight")
        group_projection = CheckpointFP8Group(
            get(prefix + ".attn.wo_a.weight"),
            get(prefix + ".attn.wo_a.scale"),
            cfg.output_projection_groups,
        )
        bind_grouped_projection(attn, group_projection)
        attn.linear_o_group_proj = torch.nn.Parameter(
            torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False
        )
        core = attn.core_attention
        copy(core.attn_sink, prefix + ".attn.attn_sink")
        if getattr(core, "compressor", None) is not None:
            compressor(core.compressor, prefix + ".attn.compressor")
        if getattr(core, "indexer", None) is not None:
            indexer = core.indexer
            indexer.linear_wq_b = fp8(prefix + ".attn.indexer.wq_b")
            copy(indexer.linear_weights_proj.weight, prefix + ".attn.indexer.weights_proj.weight")
            compressor(indexer.compressor, prefix + ".attn.indexer.compressor")
            bind_indexer_rotation(indexer)
        if getattr(model, "_rlk_fused_qnorm", False):
            norm = FusedQNorm(attn.q_layernorm, cfg.layernorm_epsilon)
            attn.q_layernorm = norm
            from rl_engine.backends.rocm.norm.qnorm_fp8 import FusedKVNorm

            original_kv_weight = attn.kv_layernorm.weight
            attn.kv_layernorm = FusedKVNorm(attn.kv_layernorm, norm, cfg.layernorm_epsilon)
            assert attn.kv_layernorm.weight is original_kv_weight
            attn.linear_q_up_proj.bind_qnorm(norm)
            if getattr(core, "indexer", None) is not None:
                core.indexer.linear_wq_b.bind_qnorm(norm)
        if hf_config is not None:
            from rl_engine.integrations.engines.train.megatron.native_qkv import bind_native_qkv

            bind_native_qkv(attn, hf_config, attn._dsv4_compress_ratio)
        inner = moe_layer.inner_layer
        moe = inner.mlp
        copy(inner.pre_mlp_layernorm.weight, prefix + ".ffn_norm.weight")
        copy(moe.router.weight, prefix + ".ffn.gate.weight")
        if prefix + ".ffn.gate.bias" in keys:
            copy(moe.router.expert_bias, prefix + ".ffn.gate.bias")
        if prefix + ".ffn.gate.tid2eid" in keys:
            copy(moe.router.tid2eid, prefix + ".ffn.gate.tid2eid")
        shared = moe.shared_experts
        shared.linear_fc1 = _GateUp(
            fp8(prefix + ".ffn.shared_experts.w1"), fp8(prefix + ".ffn.shared_experts.w3")
        )
        shared.linear_fc2 = fp8(prefix + ".ffn.shared_experts.w2")

        def expert_matrix(n, prefix=prefix):
            codes = []
            scales = []
            for e in range(cfg.num_moe_experts):
                stem = f"{prefix}.ffn.experts.{e}.{n}"
                codes.append(get(stem + ".weight").view(torch.uint8))
                scales.append(get(stem + ".scale").view(torch.uint8))
            c = torch.stack(codes)
            s = torch.stack(scales)
            return PackedExpertWeight(c, s, (c.shape[0], c.shape[1], c.shape[2] * 2))

        gate = expert_matrix("w1")
        up = expert_matrix("w3")
        down = expert_matrix("w2")
        fused = PackedExpertWeight(
            torch.cat((gate.codes, up.codes), dim=1),
            torch.cat((gate.scales, up.scales), dim=1),
            (cfg.num_moe_experts, 2 * cfg.moe_ffn_hidden_size, cfg.hidden_size),
        )
        moe.experts = NativePackedExperts(fused, down, cfg)
        print("RLK_ORIGINAL_PACKED_LAYER_LOADED", i, len(loaded), flush=True)
    if loaded != keys:
        raise ValueError(f"unconsumed checkpoint tensors: {sorted(keys - loaded)}")
    print("RLK_ORIGINAL_PACKED_CHECKPOINT_LOADED", len(loaded), flush=True)
    return {
        "loaded_tensors": len(loaded),
        "fp4_profile": RocmPackedMXBackend.profile,
        "fp8_profile": CheckpointFP8Linear.profile,
    }


def export_native_checkpoint_parameters(model):
    for name, value in model._rlk_checkpoint_export:
        yield name, value() if callable(value) else value
