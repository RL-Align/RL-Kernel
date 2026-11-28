# SPDX-License-Identifier: Apache-2.0
"""Declared C128 instance dispatch; no upstream module or operands replaced."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations.engines.rollout.vllm.c128_compressor import (
    bind_single_pass_c128,
)


def compress_norm_rope_store_triton(*args, **kwargs):
    return args, kwargs


def compress_norm_rope_store_two_stage_triton(*args, **kwargs):
    raise AssertionError("occupancy-selected prefill dispatch must not execute")


class Compressor:
    compress_ratio = 128
    head_dim = 512
    rope_head_dim = 64
    overlap = False
    use_fp4_cache = False
    _use_two_stage_fused_compressor = True
    norm = SimpleNamespace(weight=SimpleNamespace(device=torch.device("cuda:0")))

    def _compress_norm_rope_store(self, *args, **kwargs):
        if self._use_two_stage_fused_compressor:
            return compress_norm_rope_store_two_stage_triton(*args, **kwargs)
        return compress_norm_rope_store_triton(*args, **kwargs)


def test_declared_single_pass_preserves_native_operands(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "contract-test")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx942"),
    )
    compressor = Compressor()
    attention = SimpleNamespace(compressor=compressor)
    before = Compressor._compress_norm_rope_store
    assert bind_single_pass_c128(attention)
    state, metadata = object(), object()
    assert compressor._compress_norm_rope_store(state, metadata=metadata) == (
        (state,),
        {"metadata": metadata},
    )
    assert Compressor._compress_norm_rope_store is before
    assert compressor._use_two_stage_fused_compressor is False
    compressor._use_two_stage_fused_compressor = True
    with pytest.raises(RuntimeError, match="changed after binding"):
        compressor._compress_norm_rope_store(state)


def test_unsupported_c128_geometry_fails_before_mutation():
    compressor = Compressor()
    compressor.head_dim = 128
    with pytest.raises(ValueError, match="head512"):
        bind_single_pass_c128(SimpleNamespace(compressor=compressor))
    assert compressor._use_two_stage_fused_compressor is True


def test_c4_and_swa_are_not_changed():
    assert not bind_single_pass_c128(SimpleNamespace(compressor=None))
    compressor = SimpleNamespace(compress_ratio=4)
    assert not bind_single_pass_c128(SimpleNamespace(compressor=compressor))
    assert vars(compressor) == {"compress_ratio": 4}
