# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``h3_sharded_load``: a rank holds only its projection shards, the bytes the harness cuts."""

from __future__ import annotations

import pytest
import torch

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
from rl_engine.validation.models import h3_parallel_model as pm
from rl_engine.validation.models import h3_sharded_load as sl

TINY = dict(
    num_attention_heads=4,
    attention_head_dim=128,  # inner width 512: 8 units of 64 (the GEMM K tile) for row-parallel
    hidden_size=256,
    num_layers=2,
    num_refiner_layers=2,
    ffn_dim=512,
    in_channels=24,
    audio_in_channels=32,
    patch_size=[1, 2, 2],
    text_dim=64,
    freq_dim=256,
    time_embed_hidden_dim=256,
    time_embed_dim=128,
    rope_freq_dim=16,
)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_cut_shard_is_the_parallel_slicing(tp):
    gen = torch.Generator().manual_seed(tp)
    rows = torch.randn(1024, 96, generator=gen).bfloat16()
    cols = torch.randn(40, 512, generator=gen).bfloat16()
    for r in range(tp):
        assert torch.equal(
            sl.cut_shard(rows, rows.shape, "rows", tp, r), par.shard_rows(rows, tp, r)
        )
        assert torch.equal(
            sl.cut_shard(cols, cols.shape, "cols", tp, r), par.shard_cols(cols, tp, r)
        )
        first = sl.cut_shard(rows, rows.shape, "rows_first_half", tp, r)
        second = sl.cut_shard(rows, rows.shape, "rows_second_half", tp, r)
        assert torch.equal(first, par.shard_rows(rows[:512], tp, r))
        assert torch.equal(second, par.shard_rows(rows[512:], tp, r))


def test_cut_shard_refuses_what_the_harness_refuses():
    with pytest.raises(ValueError):
        sl.cut_shard(torch.zeros(12, 4), (12, 4), "rows", 2, 0)  # 12 rows are not 8 units
    with pytest.raises(ValueError):
        sl.cut_shard(torch.zeros(4, 96), (4, 96), "cols", 2, 0)  # K unit 12 is not a K tile
    for how in ("rows_first_half", "rows_second_half"):  # an odd fused weight has no halves
        with pytest.raises(ValueError, match="no equal halves"):
            sl.cut_shard(torch.zeros(1025, 96), (1025, 96), how, 1, 0)
    with pytest.raises(ValueError):
        sl.cut_shard(torch.zeros(8, 8), (8, 8), "sideways", 1, 0)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    diffusers = pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")  # the loader and from_pretrained both use it
    model_cls = diffusers.MiniMaxH3Transformer3DModel
    torch.manual_seed(0)
    model = model_cls(**TINY).to(torch.bfloat16)
    for name in model_cls._keep_in_fp32_modules:  # stored as the release stores them
        module = getattr(model, name, None)
        if module is not None:
            module.float()
    root = tmp_path_factory.mktemp("tiny_h3")
    model.save_pretrained(root, safe_serialization=True, max_shard_size="400KB")
    assert (root / sl.INDEX).exists(), "the tiny checkpoint must be sharded like the release"
    reference = model_cls.from_pretrained(root, torch_dtype=torch.bfloat16)
    return root, reference


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_loader_matches_from_pretrained(tiny_checkpoint, tp):
    root, reference = tiny_checkpoint
    ref = dict(reference.named_parameters())
    for r in range(tp):
        model = sl.load_sharded_model(root, tp, r, "cpu")
        for name, p in model.named_parameters():
            local = ".".join(name.split(".")[2:])
            if name.startswith(sl.BLOCKS + ".") and local in sl.UNLOADED:
                assert p.is_meta, name
                continue
            assert p.dtype == ref[name].dtype and torch.equal(p, ref[name]), name
        bufs, ref_bufs = dict(model.named_buffers()), dict(reference.named_buffers())
        assert bufs.keys() == ref_bufs.keys()
        for name, buf in bufs.items():
            assert not buf.is_meta, name
            assert buf.dtype == ref_bufs[name].dtype and torch.equal(buf, ref_bufs[name]), name
        for i, block in enumerate(model.transformer_blocks):
            stored = getattr(block, pm.SHARDS_ATTR)
            assert (stored["tp"], stored["tp_rank"]) == (tp, r)
            full = {
                n[len(f"{sl.BLOCKS}.{i}.") :]: p
                for n, p in ref.items()
                if n.startswith(f"{sl.BLOCKS}.{i}.")
            }
            fused = full["ff.net.0.proj.weight"]
            half = fused.shape[0] // 2
            expected = {
                **{
                    f"attn.to_{n}.weight": par.shard_rows(full[f"attn.to_{n}.weight"], tp, r)
                    for n in "qkv"
                },
                "attn.to_out.0.weight": par.shard_cols(full["attn.to_out.0.weight"], tp, r),
                "ff.hidden": par.shard_rows(fused[:half], tp, r),
                "ff.gate": par.shard_rows(fused[half:], tp, r),
                "ff.net.2.weight": par.shard_cols(full["ff.net.2.weight"], tp, r),
            }
            assert set(stored["tensors"]) == set(expected)
            for name, shard in stored["tensors"].items():
                assert shard.dtype == expected[name].dtype and torch.equal(shard, expected[name]), (
                    i,
                    name,
                )


def test_a_sharded_model_does_not_run_outside_the_grid(tiny_checkpoint):
    """Its projection weights are on ``meta``, where ``F.linear`` returns a tensor of no values."""

    root, _ = tiny_checkpoint
    model = sl.load_sharded_model(root, 2, 0, "cpu")
    block = model.transformer_blocks[0]
    x = torch.zeros(1, 4, TINY["hidden_size"], dtype=torch.bfloat16)
    for module in (block.attn.to_q, block.attn.to_out[0], block.ff.net[0].proj, block.ff.net[2]):
        with pytest.raises(RuntimeError, match="parallel_model_execution"):
            module(x)  # refused before any shape check


def test_a_projection_of_the_wrong_shape_is_refused(tiny_checkpoint, tmp_path):
    """The shards are cut from the stored tensor, so its shape must be the model's."""

    import json
    import shutil

    from safetensors.torch import load_file, save_file

    root, _ = tiny_checkpoint
    bad = tmp_path / "bad"
    shutil.copytree(root, bad)
    name = "transformer_blocks.0.attn.to_q.weight"
    file = json.loads((bad / sl.INDEX).read_text())["weight_map"][name]
    tensors = load_file(str(bad / file))
    tensors[name] = tensors[name][:-2].clone()  # two rows short, still divisible by tp=2
    save_file(tensors, str(bad / file), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="to_q.weight: checkpoint shape"):
        sl.load_sharded_model(bad, 2, 0, "cpu")
