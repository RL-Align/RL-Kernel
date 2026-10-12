# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Lossless HF Qwen3 dense <-> Megatron TP checkpoint mapping.

Q/K/V are grouped by KV head before tensor-parallel slicing. Each SwiGLU FC1
shard stores its local gate rows followed by its local up rows. No model
layers are synthesized.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _dimensions(config: Mapping, tp: int):
    if config.get("model_type") != "qwen3":
        raise ValueError("only dense Qwen3 checkpoints are supported")
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    hidden, width = config["hidden_size"], config["head_dim"]
    if tp < 1 or heads % kv or kv % tp or config["intermediate_size"] % tp:
        raise ValueError("Qwen3 head/FFN dimensions must divide the TP layout")
    if config.get("attention_bias", False) or config.get("tie_word_embeddings", False):
        raise ValueError("this adapter requires bias-free attention and an untied LM head")
    return heads, kv, hidden, width


def hf_to_megatron(state: Mapping[str, torch.Tensor], config: Mapping, *, tp: int, rank: int):
    """Map a dense HF checkpoint to one TP shard, padding vocabulary rows with zeros."""
    heads, kv, hidden, width = _dimensions(config, tp)
    if rank not in range(tp):
        raise ValueError("TP rank out of range")
    out = {}
    vocab = config["vocab_size"]
    padded = ((vocab + tp - 1) // tp) * tp
    for src, dst in [
        ("model.embed_tokens.weight", "embedding.word_embeddings.weight"),
        ("lm_head.weight", "output_layer.weight"),
    ]:
        value = state[src]
        if value.shape != (vocab, hidden):
            raise ValueError(f"invalid shape: {src}")
        value = torch.nn.functional.pad(value, (0, 0, 0, padded - vocab))
        out[dst] = value.chunk(tp, dim=0)[rank].contiguous()
    out["decoder.final_layernorm.weight"] = state["model.norm.weight"].clone()
    for layer in range(config["num_hidden_layers"]):
        hf, mg = f"model.layers.{layer}.", f"decoder.layers.{layer}."
        q = state[hf + "self_attn.q_proj.weight"].view(kv, heads // kv * width, hidden)
        k = state[hf + "self_attn.k_proj.weight"].view(kv, width, hidden)
        v = state[hf + "self_attn.v_proj.weight"].view(kv, width, hidden)
        packed = torch.cat((q, k, v), dim=1).flatten(0, 1)
        out[mg + "self_attention.linear_qkv.weight"] = packed.chunk(tp, dim=0)[rank].contiguous()
        for src, dst in [
            ("input_layernorm.weight", "input_layernorm.weight"),
            ("post_attention_layernorm.weight", "pre_mlp_layernorm.weight"),
            ("self_attn.q_norm.weight", "self_attention.q_layernorm.weight"),
            ("self_attn.k_norm.weight", "self_attention.k_layernorm.weight"),
        ]:
            out[mg + dst] = state[hf + src].clone()
        for src, dst in [
            ("self_attn.o_proj.weight", "self_attention.linear_proj.weight"),
            ("mlp.down_proj.weight", "mlp.linear_fc2.weight"),
        ]:
            out[mg + dst] = state[hf + src].chunk(tp, dim=1)[rank].contiguous()
        gate = state[hf + "mlp.gate_proj.weight"].chunk(tp, dim=0)[rank]
        up = state[hf + "mlp.up_proj.weight"].chunk(tp, dim=0)[rank]
        out[mg + "mlp.linear_fc1.weight"] = torch.cat((gate, up), dim=0).contiguous()
    return out


def megatron_to_hf(shards: list[Mapping[str, torch.Tensor]], config: Mapping):
    """Restore HF tensors from TP-ordered local shards with matching replicated norms.

    Each FC1 shard must contain [gate_rank; up_rank]. A single full mapping must
    group its FC1 rows as [all_gate; all_up].
    """
    tp = len(shards)
    heads, kv, hidden, width = _dimensions(config, tp)
    out = {}

    def replicated(name):
        first = shards[0][name]
        if any(not torch.equal(first, shard[name]) for shard in shards[1:]):
            raise ValueError(f"replicated weights disagree: {name}")
        return first.clone()

    for src, dst in [
        ("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
        ("output_layer.weight", "lm_head.weight"),
    ]:
        out[dst] = torch.cat([s[src] for s in shards], dim=0)[: config["vocab_size"]].contiguous()
    out["model.norm.weight"] = replicated("decoder.final_layernorm.weight")
    for layer in range(config["num_hidden_layers"]):
        hf, mg = f"model.layers.{layer}.", f"decoder.layers.{layer}."
        packed = torch.cat([s[mg + "self_attention.linear_qkv.weight"] for s in shards], dim=0)
        packed = packed.view(kv, (heads // kv + 2) * width, hidden)
        q, k, v = packed.split([heads // kv * width, width, width], dim=1)
        for name, value in [("q", q), ("k", k), ("v", v)]:
            out[hf + f"self_attn.{name}_proj.weight"] = value.reshape(-1, hidden).contiguous()
        for src, dst in [
            ("input_layernorm.weight", "input_layernorm.weight"),
            ("pre_mlp_layernorm.weight", "post_attention_layernorm.weight"),
            ("self_attention.q_layernorm.weight", "self_attn.q_norm.weight"),
            ("self_attention.k_layernorm.weight", "self_attn.k_norm.weight"),
        ]:
            out[hf + dst] = replicated(mg + src)
        for src, dst in [
            ("self_attention.linear_proj.weight", "self_attn.o_proj.weight"),
            ("mlp.linear_fc2.weight", "mlp.down_proj.weight"),
        ]:
            out[hf + dst] = torch.cat([s[mg + src] for s in shards], dim=1)
        pairs = [s[mg + "mlp.linear_fc1.weight"].chunk(2, dim=0) for s in shards]
        out[hf + "mlp.gate_proj.weight"] = torch.cat([pair[0] for pair in pairs], dim=0)
        out[hf + "mlp.up_proj.weight"] = torch.cat([pair[1] for pair in pairs], dim=0)
    return out


def transformer_config(config, *, tp: int, dtype: torch.dtype):
    """Translate dense Qwen3 dimensions and precision to unfused Megatron settings."""
    from megatron.core.transformer.transformer_config import TransformerConfig

    _dimensions(config, tp)
    return TransformerConfig(
        num_layers=config["num_hidden_layers"],
        hidden_size=config["hidden_size"],
        num_attention_heads=config["num_attention_heads"],
        num_query_groups=config["num_key_value_heads"],
        kv_channels=config["head_dim"],
        ffn_hidden_size=config["intermediate_size"],
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=1,
        normalization="RMSNorm",
        layernorm_epsilon=config["rms_norm_eps"],
        qk_layernorm=True,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        masked_softmax_fusion=False,
        bias_dropout_fusion=False,
        bias_activation_fusion=False,
        attention_softmax_in_fp32=True,
    )


def load_pretrained(
    checkpoint,
    *,
    tp: int,
    policy: bool,
    dtype: torch.dtype,
    pg_collection=None,
    max_sequence_length=40960,
):
    """Build the native Megatron Qwen3 topology and load every real tensor strictly.

    The caller must verify the pinned checkpoint manifest before launching ranks.
    Scalar heads are separately initialized; their backbones still load pretrained
    final normalization, even though Megatron's post_process=False omits that norm.
    """
    import json
    from pathlib import Path

    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from safetensors.torch import load_file

    from rl_engine.integrations.engines.train.megatron.ppo import (
        MegatronPolicyAdapter,
        MegatronScalarAdapter,
    )

    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("rope_scaling") or config.get("use_sliding_window"):
        raise ValueError("this mapping requires unscaled full-context Qwen3 RoPE")
    group = pg_collection.tp if pg_collection else parallel_state.get_tensor_model_parallel_group()
    if dist.get_world_size(group) != tp:
        raise ValueError("checkpoint TP and active process group disagree")
    rank = dist.get_rank(group)
    model_parallel_cuda_manual_seed(
        44, tp_rank=rank, ep_rank=0, etp_rank=rank, force_reset_rng=True
    )
    native_config = transformer_config(config, tp=tp, dtype=dtype)
    # Keep post_process for scalar backbones until after loading: native Megatron
    # otherwise omits the pretrained final RMSNorm together with the LM head.
    model = GPTModel(
        native_config,
        get_gpt_layer_local_spec(qk_layernorm=True, normalization="RMSNorm"),
        vocab_size=((config["vocab_size"] + tp - 1) // tp) * tp,
        max_sequence_length=max_sequence_length,
        post_process=True,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=config["rope_theta"],
        rotary_percent=1.0,
        pg_collection=pg_collection,
    )
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    state = {}
    for filename in sorted(set(index["weight_map"].values())):
        state.update(load_file(str(checkpoint / filename), device="cpu"))
    if set(state) != set(index["weight_map"]):
        raise ValueError("checkpoint tensor inventory differs from shard index")
    mapped = hf_to_megatron(state, config, tp=tp, rank=rank)
    del state
    target = dict(model.named_parameters())
    if target.keys() != mapped.keys():
        raise ValueError(f"native Qwen3 parameter mismatch: {target.keys() ^ mapped.keys()}")
    with torch.no_grad():
        for name, parameter in target.items():
            if parameter.shape != mapped[name].shape:
                raise ValueError(f"native Qwen3 shape mismatch: {name}")
            parameter.copy_(mapped[name])
    del mapped
    model = model.to(device="cuda", dtype=dtype).eval()
    if policy:
        return MegatronPolicyAdapter(model, real_vocab_size=config["vocab_size"])
    model.post_process = False
    del model.output_layer
    head = torch.nn.Linear(config["hidden_size"] // tp, 1, bias=False, device="cuda", dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(
            torch.randn(
                1, config["hidden_size"], generator=torch.Generator().manual_seed(144)
            ).chunk(tp, dim=1)[rank]
            * 0.01
        )
    return MegatronScalarAdapter(
        model, head, group=group, leader=dist.get_process_group_ranks(group)[0]
    ).eval()
