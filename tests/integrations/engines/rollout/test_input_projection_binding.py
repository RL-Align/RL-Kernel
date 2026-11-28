# SPDX-License-Identifier: Apache-2.0
"""Rollout fused wq_a|wkv must reproduce the actor's separate original projections."""

import types

import pytest
import torch
from safetensors.torch import save_file

from rl_engine.integrations.engines.rollout.vllm.group_projection import (
    bind_original_input_projection,
)

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)

HIDDEN, Q_LORA, HEAD = 4096, 1024, 512


def _original(rows, gen):
    codes = torch.randint(0, 0x78, (rows, HIDDEN), dtype=torch.uint8, generator=gen)
    codes |= torch.randint(0, 2, (rows, HIDDEN), dtype=torch.uint8, generator=gen) << 7
    scales = torch.randint(118, 124, (rows // 128, HIDDEN // 128), dtype=torch.uint8, generator=gen)
    return codes.view(torch.float8_e4m3fn), scales.view(torch.float8_e8m0fnu)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    gen = torch.Generator().manual_seed(4242)
    tensors = {}
    for key, rows in (("wq_a", Q_LORA), ("wkv", HEAD)):
        weight, scale = _original(rows, gen)
        tensors[f"layers.3.attn.{key}.weight"] = weight
        tensors[f"layers.3.attn.{key}.scale"] = scale
    path = tmp_path_factory.mktemp("ckpt") / "model.safetensors"
    save_file(tensors, str(path))
    return path, tensors


def _attention(q_lora=Q_LORA, head=HEAD):
    weight = torch.empty(q_lora + head, HIDDEN, device="cuda", dtype=torch.float8_e4m3fnuz)
    return types.SimpleNamespace(
        fused_wqa_wkv=types.SimpleNamespace(weight=weight), q_lora_rank=q_lora, head_dim=head
    )


@pytest.mark.parametrize("m", [1, 5, 8, 127, 130, 513, 2052])
def test_fused_gemm_equals_actor_projections(checkpoint, m):
    from rl_engine.backends.rocm.gemm.checkpoint_fp8 import CheckpointFP8Linear

    path, tensors = checkpoint
    attention = bind_original_input_projection(_attention(), path, 3)
    x = torch.randn(m, HIDDEN, generator=torch.Generator().manual_seed(m)).bfloat16().cuda()
    out = attention._fused_wqa_wkv_gemm(x)
    assert out.shape == (m, Q_LORA + HEAD) and out.dtype == torch.bfloat16
    for key, columns in (("wq_a", slice(0, Q_LORA)), ("wkv", slice(Q_LORA, None))):
        actor = CheckpointFP8Linear(
            tensors[f"layers.3.attn.{key}.weight"].cuda(),
            tensors[f"layers.3.attn.{key}.scale"].cuda(),
        )
        expected = actor(x.reshape(m, 1, HIDDEN))[0].reshape(m, -1)
        assert torch.equal(out[:, columns].view(torch.int16), expected.view(torch.int16))


def test_rows_are_batch_size_invariant(checkpoint):
    path, _ = checkpoint
    attention = bind_original_input_projection(_attention(), path, 3)
    x = torch.randn(2052, HIDDEN, generator=torch.Generator().manual_seed(7)).bfloat16().cuda()
    full = attention._fused_wqa_wkv_gemm(x)
    for m in (1, 5, 127, 513):
        part = attention._fused_wqa_wkv_gemm(x[:m].contiguous())
        assert torch.equal(part.view(torch.int16), full[:m].view(torch.int16))


def test_empty_batch_and_geometry_check(checkpoint):
    path, _ = checkpoint
    attention = bind_original_input_projection(_attention(), path, 3)
    empty = attention._fused_wqa_wkv_gemm(torch.empty(0, HIDDEN, device="cuda").bfloat16())
    assert empty.shape == (0, Q_LORA + HEAD)
    with pytest.raises(ValueError):
        bind_original_input_projection(_attention(head=256), path, 3)
