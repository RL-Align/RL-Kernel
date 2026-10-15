# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Actual vLLM imports/loaders and constructor contracts, without GPU execution."""

from types import SimpleNamespace as Namespace

import pytest
import torch
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear

from rl_engine.integrations.engines.rollout.vllm import qwen3_next_vllm_model as adapter
from rl_engine.models.qwen3_next.qwen3_next_tp_blocks import shard_attention_weights
from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal as exact


def profile():
    config = Namespace(
        hidden_size=2048,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        vocab_size=151936,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        rms_norm_eps=1e-6,
        max_position_embeddings=262144,
        layer_types=[
            "full_attention" if index % 4 == 3 else "linear_attention" for index in range(48)
        ],
        rope_parameters={
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
    )
    return Namespace(
        model_config=Namespace(dtype=torch.bfloat16, enforce_eager=True, hf_text_config=config),
        parallel_config=Namespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            use_sequence_parallel_moe=False,
            enable_dbo=False,
        ),
        quant_config=None,
        speculative_config=None,
        lora_config=None,
        cache_config=Namespace(enable_prefix_caching=False),
        kernel_config=Namespace(moe_backend="triton"),
        attention_config=Namespace(backend=adapter.AttentionBackendEnum.CUSTOM),
    )


def test_profile_keeps_engine_and_model_acceptance_closed():
    adapter.validate_profile(profile())
    assert adapter.profile_capabilities()["scheduler_batch_invariance_verified"] is False
    assert adapter.profile_capabilities()["full_model_l2_verified"] is False


@pytest.mark.parametrize(
    "scope,name,value",
    [
        ("model_config", "dtype", torch.float32),
        ("model_config", "enforce_eager", False),
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "enable_expert_parallel", True),
        ("cache_config", "enable_prefix_caching", True),
        ("kernel_config", "moe_backend", "auto"),
        ("attention_config", "backend", None),
    ],
)
def test_real_wrapper_constructor_rejects_drift_before_allocating_model(scope, name, value):
    config = profile()
    setattr(getattr(config, scope), name, value)
    with pytest.raises(ValueError):
        adapter.RLKernelQwen3NextForCausalLM(vllm_config=config)


@pytest.mark.parametrize("rank", range(4))
def test_actual_native_qkv_loader_preserves_gate_layout_and_kv_replication(monkeypatch, rank):
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 4)
    # Construct the real native layer, then use its unchanged HF shard loader.
    layer = linear.QKVParallelLinear(
        2048,
        256,
        32,
        2,
        bias=False,
        params_dtype=torch.bfloat16,
    )
    weights = {}
    for name, rows in (("q_proj.weight", 8192), ("k_proj.weight", 512), ("v_proj.weight", 512)):
        weights[name] = (
            torch.arange(rows, dtype=torch.int32)
            .remainder(251)
            .to(torch.bfloat16)[:, None]
            .expand(rows, 2048)
            .contiguous()
        )
    weights.update(
        {
            "o_proj.weight": torch.zeros(2048, 4096, dtype=torch.bfloat16),
            "q_norm.weight": torch.zeros(256, dtype=torch.bfloat16),
            "k_norm.weight": torch.zeros(256, dtype=torch.bfloat16),
        }
    )
    expected = shard_attention_weights(weights, rank)
    for key, shard in (("q_proj.weight", "q"), ("k_proj.weight", "k"), ("v_proj.weight", "v")):
        layer.weight.weight_loader(layer.weight, weights[key], shard)
    for actual, key in zip(
        layer.weight.split((2048, 256, 256)), ("q_proj.weight", "k_proj.weight", "v_proj.weight")
    ):
        exact(actual, expected[key], name=f"native rank{rank} {key}")


def test_shared_moe_postload_preserves_parameter_identity_and_loader(monkeypatch):
    # Meta tensors verify 512-expert allocation shapes without allocating an
    # unnecessary CPU model. Real data movement is covered by the TP4 GPU gate.
    gate = torch.nn.Parameter(torch.empty(512, 256, 2048, device="meta", dtype=torch.bfloat16))
    down = torch.nn.Parameter(torch.empty(512, 2048, 128, device="meta", dtype=torch.bfloat16))
    loader = object()
    gate.weight_loader = down.weight_loader = loader
    layer = Namespace(w13_weight=gate, w2_weight=down)
    method = object.__new__(adapter.SharedPlainBF16MoEMethod)
    monkeypatch.setattr(
        adapter.SharedPlainBF16MoEMethod,
        "get_fused_moe_quant_config",
        lambda self, layer: "plain-bf16",
    )
    method.process_weights_after_loading(layer)
    assert layer.w13_weight is gate and layer.w2_weight is down
    assert gate.weight_loader is loader and down.weight_loader is loader
    assert method.moe_quant_config == "plain-bf16"
    with pytest.raises(RuntimeError, match="native MoE arithmetic"):
        method.apply()


def test_shared_model_rejects_rope_or_layer_layout_changes():
    config = profile()
    config.model_config.hf_text_config.rope_parameters["rope_theta"] = 10000
    with pytest.raises(ValueError, match="rotary dimension"):
        adapter.validate_profile(config)
    config = profile()
    config.model_config.hf_text_config.layer_types[0] = "full_attention"
    with pytest.raises(ValueError, match="36 GDN"):
        adapter.validate_profile(config)


@pytest.mark.parametrize(
    "decoder_class", [adapter.Qwen3NextDecoderLayer, adapter.SharedVLLMDecoderLayer]
)
def test_real_decoder_invokes_gdn_with_single_input_and_consumes_returned_tensor(
    monkeypatch, decoder_class
):
    from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm import SharedQwenGDN

    class ResidualNorm(torch.nn.Module):
        def forward(self, hidden, residual=None):
            if residual is None:
                return hidden
            return hidden + residual, hidden + residual

    gdn = object.__new__(SharedQwenGDN)
    torch.nn.Module.__init__(gdn)
    calls = []

    def into(self, hidden, output):
        assert output is not hidden and output.data_ptr() != hidden.data_ptr()
        assert output.dtype == hidden.dtype and output.shape == hidden.shape
        output.copy_(hidden + 2)
        calls.append(output)

    monkeypatch.setattr(SharedQwenGDN, "_forward_into", into)
    monkeypatch.setattr(adapter, "_rms", lambda value, module: value)
    decoder = object.__new__(decoder_class)
    torch.nn.Module.__init__(decoder)
    decoder.layer_type = "linear_attention"
    decoder.layer_scale = decoder.use_attn_reduce_scatter_for_moe = False
    decoder.linear_attn = gdn
    decoder.input_layernorm = decoder.post_attention_layernorm = ResidualNorm()
    decoder.mlp = torch.nn.Identity()
    hidden = torch.ones((2, 2048), dtype=torch.bfloat16)
    output, residual = decoder(hidden_states=hidden, residual=None, positions=torch.arange(2))
    assert len(calls) == 1
    exact(calls[0], torch.full_like(hidden, 3), name="public GDN return")
    exact(output, torch.full_like(hidden, 4), name="decoder consumed GDN output")
    exact(residual, torch.full_like(hidden, 4), name="decoder residual")


def test_binding_all48_layers_preserves_parameter_identity_and_native_loaders():
    from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm import SharedQwenGDN
    from rl_engine.integrations.engines.rollout.vllm.qwen3_next_vllm_attention import (
        SharedAttentionBackend,
    )

    def instance(cls):
        value = object.__new__(cls)
        torch.nn.Module.__init__(value)
        return value

    loader = object()

    def weight_module(shape):
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.empty(shape, device="meta", dtype=torch.bfloat16))
        module.weight.weight_loader = loader
        return module

    # The binding operation does not run a native constructor or GPU kernel.
    # Meta tensors make its 48-layer ownership contract affordable to test.
    model = instance(adapter.Qwen3NextModel)
    model.start_layer, model.end_layer = 0, 48
    model.embed_tokens = weight_module((37984, 2048))
    model.norm = weight_module((2048,))
    layers = []
    for index in range(48):
        layer = instance(adapter.Qwen3NextDecoderLayer)
        layer.layer_type = "full_attention" if index % 4 == 3 else "linear_attention"
        if layer.layer_type == "linear_attention":
            layer.linear_attn = instance(SharedQwenGDN)
        else:
            layer.self_attn = instance(adapter.Qwen3NextAttention)
            layer.self_attn.attn = Namespace(attn_backend=SharedAttentionBackend)
            layer.self_attn.qkv_proj = weight_module((2560, 2048))
            layer.self_attn.o_proj = weight_module((2048, 1024))
        layer.mlp = instance(adapter.Qwen3NextSparseMoeBlock)
        layer.mlp.is_fused_shared_expert_enabled = False
        layer.mlp.replicate_shared_expert = False
        layer.mlp.shared_expert = torch.nn.Module()
        layer.mlp.shared_expert.gate_up_proj = weight_module((256, 2048))
        layer.mlp.shared_expert.down_proj = weight_module((2048, 128))
        routed = torch.nn.Module()
        routed.quant_method = instance(adapter.UnquantizedFusedMoEMethod)
        routed.w13_weight = weight_module((512, 256, 2048)).weight
        routed.w2_weight = weight_module((512, 2048, 128)).weight
        layer.mlp.experts = torch.nn.Module()
        layer.mlp.experts.routed_experts = routed
        layers.append(layer)
    model.layers = torch.nn.ModuleList(layers)
    before = {name: id(value) for name, value in model.named_parameters()}
    adapter._bind_native_model(model, object())
    assert isinstance(model, adapter.SharedVLLMModel)
    assert {name: id(value) for name, value in model.named_parameters()} == before
    assert all(value.weight_loader is loader for value in model.parameters())
    assert all(isinstance(layer, adapter.SharedVLLMDecoderLayer) for layer in model.layers)
    assert all(isinstance(layer.mlp, adapter.SharedVLLMMoE) for layer in model.layers)


def test_spawn_worker_registers_model_before_native_worker_initialization(monkeypatch):
    from rl_engine.integrations.engines.rollout.vllm import qwen3_next_vllm_worker as worker

    calls = []
    monkeypatch.setattr(worker, "register_shared_model", lambda: calls.append("registered"))
    monkeypatch.setattr(
        worker,
        "imported_provider_identity",
        lambda: calls.append("native-binary") or {"test": True},
    )
    monkeypatch.setattr(
        worker.Worker, "__init__", lambda self, *args, **kwargs: calls.append("native-init")
    )
    worker.SharedQwenWorker()
    assert calls == ["registered", "native-binary", "native-init"]


def test_architecture_registration_is_idempotent_and_rejects_another_model(monkeypatch):
    from vllm.model_executor.models import ModelRegistry

    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    ModelRegistry.models.pop(adapter.ARCHITECTURE, None)
    assert adapter.register_shared_model() == adapter.ARCHITECTURE
    assert adapter.register_shared_model() == adapter.ARCHITECTURE
    ModelRegistry.models[adapter.ARCHITECTURE] = Namespace(load_model_cls=lambda: torch.nn.Linear)
    with pytest.raises(ValueError, match="different model"):
        adapter.register_shared_model()
