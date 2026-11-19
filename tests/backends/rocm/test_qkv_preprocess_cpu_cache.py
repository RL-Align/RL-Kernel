# SPDX-License-Identifier: Apache-2.0
"""CPU-cache construction and real-device delayed RoPE migration regressions."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

from rl_engine.backends.rocm.attention.qkv_preprocess import DSV4QKVPreprocess, _QKV


def test_cpu_cached_rope_does_not_query_gpu_properties(monkeypatch):
    # Isolate construction from optional vLLM/GPU dependencies. The GPU test
    # below exercises its real process-wide RoPE cache and fused codec.
    cache = torch.randn(8, 64, dtype=torch.float32)
    config = SimpleNamespace(max_position_embeddings=8, rms_norm_eps=1e-6)
    vllm_config = ModuleType("vllm.config")
    vllm_config.VllmConfig = object
    vllm_config.get_current_vllm_config_or_none = lambda: object()
    vllm_config.set_current_vllm_config = lambda _: nullcontext()
    rope_module = ModuleType("vllm.models.deepseek_v4.common.rope")

    def cached_rope(*args, **kwargs):
        return SimpleNamespace(cos_sin_cache=cache)

    rope_module.build_deepseek_v4_rope = cached_rope
    monkeypatch.setitem(sys.modules, vllm_config.__name__, vllm_config)
    monkeypatch.setitem(sys.modules, rope_module.__name__, rope_module)

    def unexpected_device_query(*args, **kwargs):
        pytest.fail("construction must not inspect the cached RoPE tensor's device")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected_device_query)
    op = DSV4QKVPreprocess(config, 1).bfloat16()
    assert op.cos_sin_cache is cache
    assert op.cos_sin_cache.dtype == torch.float32
    assert not hasattr(op, "_executed")


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)
@pytest.mark.parametrize("compression_ratio", [1, 4])
def test_real_cpu_cached_rope_moves_on_first_gpu_forward(monkeypatch, capsys, compression_ratio):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers import rotary_embedding
    from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope

    # A private real vLLM cache avoids test-order dependencies and restores the
    # prior process cache at teardown. Only table length is reduced for this
    # operator regression; no checkpoint weights or quantization are changed.
    monkeypatch.setattr(rotary_embedding, "_ROPE_DICT", {})
    config = SimpleNamespace(
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        compress_rope_theta=160000,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )
    with set_current_vllm_config(VllmConfig()):
        with torch.device("cpu"):
            rope = build_deepseek_v4_rope(
                config,
                head_dim=512,
                rope_head_dim=64,
                max_position_embeddings=config.max_position_embeddings,
                compress_ratio=compression_ratio,
            )
        assert rope.cos_sin_cache.device.type == "cpu"
        op = DSV4QKVPreprocess(config, compression_ratio).bfloat16()
    assert op.cos_sin_cache is rope.cos_sin_cache
    assert op.cos_sin_cache.dtype == torch.float32

    query = torch.randn(3, 1, 64, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(3, 1, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    positions = torch.arange(3, device="cuda", dtype=torch.int64)[:, None]
    reference_cache = rope.cos_sin_cache.to(query.device)
    actual = op(query, kv, positions)
    assert op.cos_sin_cache.device == query.device
    assert op.cos_sin_cache.dtype == torch.float32
    expected = _QKV.apply(
        query.detach(), kv.detach(), positions, reference_cache, config.rms_norm_eps
    )
    for result, reference in zip(actual, expected, strict=True):
        assert torch.equal(result.view(torch.uint8), reference.view(torch.uint8))
    sum(result.float().square().mean() for result in actual).backward()
    for grad in (query.grad, kv.grad):
        assert grad is not None and bool(grad.isfinite().all()) and bool(grad.ne(0).any())
    arch = torch.cuda.get_device_properties(query.device).gcnArchName.split(":")[0]
    assert op.profile == f"dsv4-vllm-qnorm-rope-fp8dsmla-{arch}-v1"
    assert op.profile in capsys.readouterr().out
    op(query.detach(), kv.detach(), positions)
    assert "RLK_QKV_PREPROCESS_EXECUTED" not in capsys.readouterr().out
