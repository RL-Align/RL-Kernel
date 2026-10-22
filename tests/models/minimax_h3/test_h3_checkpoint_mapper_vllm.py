# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 integration rows: checkpoint mapper and the vLLM-Omni rollout adapter."""

from __future__ import annotations

import pytest
import torch

from rl_engine.models.minimax_h3 import checkpoint_mapper as cm
from rl_engine.models.minimax_h3 import fingerprint as fp

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --- checkpoint mapper ----------------------------------------------------------------------


def _names():
    return list(fp.expected_tensor_table(fp.transformer_config()))


def test_every_checkpoint_tensor_maps_and_native_names_are_unique():
    names = _names()
    assert len(names) == 638
    state = {n: torch.zeros(6, 2) for n in names}
    native = cm.diffusers_to_native(state)
    # q/k/v of each of the 52 blocks (50 + 2 refiner) fuse per kind (weight): 3 -> 1
    assert len(native) == 638 - 2 * 52
    assert len(set(native)) == len(native)


def test_diffusers_native_roundtrip_is_bitwise():
    names = _names()
    gen = torch.Generator().manual_seed(0)
    state = {n: torch.randn(6, 2, generator=gen) for n in names}
    back = cm.native_to_diffusers(cm.diffusers_to_native(state))
    assert set(back) == set(state)
    assert all(torch.equal(back[n], state[n]) for n in state)


def test_fc1_halves_are_swapped_and_qkv_is_q_then_k_then_v():
    w = torch.arange(8.0).view(8, 1)
    state = {
        "transformer_blocks.3.ff.net.0.proj.weight": w,
        "transformer_blocks.3.attn.to_q.weight": torch.full((2, 1), 1.0),
        "transformer_blocks.3.attn.to_k.weight": torch.full((2, 1), 2.0),
        "transformer_blocks.3.attn.to_v.weight": torch.full((2, 1), 3.0),
    }
    native = cm.diffusers_to_native(state)
    assert torch.equal(native["blocks.3.mlp.fc1.weight"], torch.cat([w[4:], w[:4]]))
    assert native["blocks.3.attn.qkv_proj.weight"].flatten().tolist() == [1, 1, 2, 2, 3, 3]


def test_unmapped_names_fail_closed():
    with pytest.raises(ValueError, match="unmapped"):
        cm.diffusers_to_native({"transformer_blocks.0.surprise.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="unmapped"):
        cm.native_to_diffusers({"blocks.0.surprise.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="q, k and v"):
        cm.diffusers_to_native({"transformer_blocks.0.attn.to_q.weight": torch.zeros(1, 1)})


def test_grouped_checkpoint_layout_roundtrip():
    w = torch.randn(4 * 3 * 8, 5)
    fused = cm.grouped_qkv_to_qkv(w, heads=4, head_dim=8)
    assert torch.equal(cm.qkv_to_grouped(fused, heads=4, head_dim=8), w)


def test_tables_match_vllm_omni_when_importable():
    fasth3 = pytest.importorskip("vllm_omni.diffusion.models.minimax_h3.fasth3")
    assert dict(fasth3._MODEL_LEVEL_TARGETS) == cm.MODEL_LEVEL
    assert {k: v[0] for k, v in fasth3._BLOCK_TARGETS.items() if k in cm.BLOCK_TARGETS} == {
        k: v[0] for k, v in cm.BLOCK_TARGETS.items()
    }


class _Lin(torch.nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight = torch.nn.Parameter(
            (torch.randn(n, k) / 30).bfloat16().cuda(), requires_grad=False
        )

    def forward(self, x):
        raise AssertionError("must be replaced")


@needs_gpu
def test_vllm_adapter_patches_projections_and_restores():
    pytest.importorskip("triton")
    from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
    from rl_engine.integrations.engines.rollout.vllm.h3_omni import install_h3_rollout

    torch.manual_seed(0)
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    blk = model.blocks[0]
    blk.attn = torch.nn.Module()
    blk.attn.qkv_proj, blk.attn.out_proj = _Lin(3 * 1024, 640), _Lin(640, 1024)
    blk.mlp = torch.nn.Module()
    blk.mlp.fc1, blk.mlp.fc2 = _Lin(2 * 2048, 640), _Lin(640, 2048)
    x = torch.randn(2, 17, 640).bfloat16().cuda()
    with install_h3_rollout(model) as handle:
        qkv, bias = blk.attn.qkv_proj(x)
        assert bias is None and qkv.shape == (2, 17, 3 * 1024)
        q, k, v = (blk.attn.qkv_proj.weight[i * 1024 : (i + 1) * 1024] for i in range(3))
        sep = [par.g.gemm(x.reshape(-1, 640), w, b_t=True) for w in (q, k, v)]
        assert torch.equal(qkv.reshape(-1, 3072), torch.cat(sep, dim=1))  # fused == separate
        y, _ = blk.attn.out_proj(torch.randn(2, 17, 1024).bfloat16().cuda())
        assert y.shape == (2, 17, 640)
        assert sorted(handle.patched) == [
            "blocks.0.attn.out_proj",
            "blocks.0.attn.qkv_proj",
            "blocks.0.mlp.fc1",
            "blocks.0.mlp.fc2",
        ]
    with pytest.raises(AssertionError, match="must be replaced"):
        blk.attn.qkv_proj(x)


@needs_gpu
def test_vllm_adapter_strict_rejects_wrong_dtype_and_empty_models():
    from rl_engine.integrations.engines.rollout.vllm.h3_omni import install_h3_rollout

    with pytest.raises(RuntimeError, match="no H3 projection"):
        install_h3_rollout(torch.nn.Linear(2, 2).cuda())
    holder = torch.nn.Module()
    holder.qkv_proj = torch.nn.Linear(8, 8).cuda()  # float32
    with pytest.raises(RuntimeError, match="BF16"):
        install_h3_rollout(holder)
