# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch contracts for the distinct native C128 numeric profiles."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from rl_engine.backends.rocm.attention import compressor_schedule, native_compressor as codec


@pytest.mark.parametrize("profile", [codec.C128_PROFILE, codec.PINNED_C128_PROFILE])
def test_c128_profile_selects_the_matching_writer(monkeypatch, profile):
    calls = []

    def single(**kwargs):
        calls.append(("single", kwargs))

    def two_stage(**kwargs):
        calls.append(("two_stage", kwargs))

    modules = {
        "vllm.models.deepseek_v4.common.ops.cache_utils": dict(
            dequantize_and_gather_k_cache=lambda out, *args, **kwargs: out.zero_()
        ),
        "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache": dict(
            compress_norm_rope_store_triton=single,
            compress_norm_rope_store_two_stage_triton=two_stage,
        ),
        "vllm.models.deepseek_v4.common.ops.save_partial_states": dict(
            _SAVE_PARTIAL_STATES_KERNEL=lambda **kwargs: None
        ),
        "vllm.models.deepseek_v4.compressor": dict(CompressorMetadata=SimpleNamespace),
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    pinned = []
    monkeypatch.setattr(
        compressor_schedule, "pin_c128_compressor_schedule", lambda: pinned.append(True)
    )
    kv = torch.zeros(128, 1, 512)
    result = codec._forward(
        kv,
        kv.clone(),
        torch.zeros(128, 512),
        torch.ones(512),
        torch.zeros(128, 64),
        128,
        1e-6,
        profile,
    )
    assert result.shape == (1, 1, 512) and result.dtype == torch.bfloat16
    assert len(calls) == 1
    name, kwargs = calls[0]
    if profile == codec.PINNED_C128_PROFILE:
        assert name == "two_stage" and pinned == [True]
        assert kwargs["num_decode_tokens"] == 0
        assert kwargs["compress_scratch"].shape == (128, 512)
    else:
        assert name == "single" and not pinned
        assert "num_decode_tokens" not in kwargs and "compress_scratch" not in kwargs


@pytest.mark.parametrize("profile", [codec.C128_PROFILE, codec.PINNED_C128_PROFILE])
def test_profile_argument_preserves_the_autograd_boundary(monkeypatch, profile):
    seen = []

    def forward(kv, score, ape, weight, cache, ratio, eps, selected_profile):
        seen.append(selected_profile)
        return codec._functional(kv, score, ape, weight, cache, ratio, eps).to(torch.bfloat16)

    monkeypatch.setattr(codec, "_forward", forward)
    torch.manual_seed(1)
    operands = [
        torch.randn(128, 1, 512, requires_grad=True),
        torch.randn(128, 1, 512, requires_grad=True),
        torch.randn(128, 512, requires_grad=True),
        torch.randn(512, requires_grad=True),
    ]
    cache = torch.randn(128, 64)
    codec.compressed_kv(*operands, cache, 128, c128_profile=profile).float().sum().backward()
    assert seen == [profile]
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in operands)
    with pytest.raises(ValueError, match="unsupported native C128"):
        codec.compressed_kv(*operands, cache, 128, c128_profile="unknown")
