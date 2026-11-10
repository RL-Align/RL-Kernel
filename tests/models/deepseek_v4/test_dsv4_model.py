# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 decoder layers + model (forward): batch invariance, loading, RoPE, config.

The model is four layers of the real DeepSeek-V4-Flash shapes and types (two
sliding-window layers, C4 with the indexer, C128; hash routing on the first three)
with random weights and a small vocabulary.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest
import torch


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    if torch.cuda.get_device_capability()[0] != 9:
        return False
    try:
        import rl_engine._C as ext
        from rl_engine.kernels.ops.cuda.deepgemm import deepgemm_available
        from rl_engine.kernels.ops.cuda.flashmla import flashmla_available
    except ImportError:
        return False
    need = ("dsv4_mhc_pre", "dsv4_router_topk", "sm90_moe_fused_mlp_forward")
    return all(hasattr(ext, n) for n in need) and deepgemm_available() and flashmla_available()


gpu = pytest.mark.skipif(
    not _ready(), reason="SM90, DeepGEMM, FlashMLA and a KERNEL_ALIGN_MOE_SM90=1 build"
)

MAX_LEN = 4096
ROPE = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 65536,
    "type": "yarn",
}
QUANT = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
}


def _config(**kw):
    from rl_engine.models.deepseek_v4.config import Dsv4Config

    d = dict(
        vocab_size=2048,
        num_hidden_layers=4,
        compress_ratios=[0, 0, 4, 128],
        num_hash_layers=3,
        rope_scaling=ROPE,
        quantization_config=QUANT,
    )
    d.update(kw)
    return Dsv4Config.from_dict(d)


@pytest.fixture(scope="module")
def ckpt():
    from rl_engine.models.deepseek_v4.loader import random_checkpoint

    return random_checkpoint(_config(), seed=0)


@pytest.fixture(scope="module")
def model(ckpt):
    from rl_engine.models.deepseek_v4.loader import load_model

    return load_model(ckpt, _config(), max_model_len=MAX_LEN)


class Runner:
    """Feeds steps (lists of (request, lo, hi)) through the model with its own caches."""

    def __init__(self, model, ids, requests, seed):
        self.model, self.ids, self.requests = model, ids, requests
        self.caches = model.new_caches(len(requests), MAX_LEN, seed=seed)

    def step(self, parts):
        rows = torch.tensor([self.requests.index(r) for r, _, _ in parts], device="cuda")
        ids = torch.cat([self.ids[r][lo:hi] for r, lo, hi in parts])
        views = [
            type(c)(
                **{
                    k: (v[rows] if k.endswith("table") and v is not None else v)
                    for k, v in c.__dict__.items()
                }
            )
            for c in self.caches
        ]
        seq = [hi for _, _, hi in parts]
        qlen = [hi - lo for _, lo, hi in parts]
        return self.model.forward(ids, seq, qlen, views)


def _same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@gpu
def test_model_is_batch_invariant(model):
    lengths = {0: 200, 1: 300, 2: 1000}
    gen = torch.Generator().manual_seed(5)
    vocab = model.config.vocab_size
    ids = {r: torch.randint(vocab, (n,), generator=gen).cuda() for r, n in lengths.items()}

    full = Runner(model, ids, [0, 1, 2], seed=1).step([(r, 0, n) for r, n in lengths.items()])
    assert torch.isfinite(full.float()).all() and float(full.float().std()) > 0.1
    off = {0: 0, 1: lengths[0], 2: lengths[0] + lengths[1]}
    rows = {r: full[off[r] : off[r] + n] for r, n in lengths.items()}

    alone = Runner(model, ids, [1], seed=2).step([(1, 0, lengths[1])])
    assert _same(alone, rows[1])

    chunked = Runner(model, ids, [2], seed=3)
    a = chunked.step([(2, 0, 613)])
    b = chunked.step([(2, 613, lengths[2])])
    assert _same(torch.cat([a, b]), rows[2])

    dec = Runner(model, ids, [0, 2], seed=4)
    n0, n2 = lengths[0], lengths[2]
    p0 = dec.step([(0, 0, n0 - 3), (2, 0, n2 - 3)])
    assert _same(p0[: n0 - 3], rows[0][: n0 - 3])
    for k in range(3, 0, -1):
        out = dec.step([(0, n0 - k, n0 - k + 1), (2, n2 - k, n2 - k + 1)])
        assert _same(out[0:1], rows[0][n0 - k : n0 - k + 1]), k
        assert _same(out[1:2], rows[2][n2 - k : n2 - k + 1]), k


@gpu
def test_layers_are_wired(model):
    """Every layer changes the streams, and stream outputs are reproducible."""
    from rl_engine.models.deepseek_v4.attention import Dsv4Batch

    ids = torch.randint(model.config.vocab_size, (300,), generator=torch.Generator().manual_seed(1))
    ids = ids.cuda()
    batch = Dsv4Batch.from_lens([300], [300], model.config.sliding_window)
    out, streams = model.forward_batch(
        ids, batch, model.new_caches(1, MAX_LEN), return_streams=True
    )
    again, _ = model.forward_batch(
        ids, batch, model.new_caches(1, MAX_LEN, seed=9), return_streams=True
    )
    assert _same(out, again)
    prev = model.embed_streams(ids)
    for i, s in enumerate(streams):
        assert s.shape == prev.shape and torch.isfinite(s.float()).all(), i
        assert not torch.equal(s, prev), i
        prev = s


@gpu
def test_load_from_directory(tmp_path, ckpt):
    from safetensors.torch import save_file

    from rl_engine.models.deepseek_v4.attention import Dsv4Batch
    from rl_engine.models.deepseek_v4.loader import load_model

    cfg = _config(num_hidden_layers=3, compress_ratios=[0, 0, 4])
    shards: dict[str, dict] = {}
    for name, t in ckpt.items():
        if name.startswith("layers.3."):
            continue
        shard = name.split(".")[1] if name.startswith("layers.") else "top"
        shards.setdefault(f"model-{shard}.safetensors", {})[name] = t.cpu().contiguous()
    weight_map = {}
    for shard, tensors in shards.items():
        save_file(tensors, str(tmp_path / shard))
        weight_map.update({k: shard for k in tensors})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    d = {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}
    (tmp_path / "config.json").write_text(json.dumps(d))

    from_dir = load_model(tmp_path, max_model_len=MAX_LEN, load_head=False)
    from_map = load_model(ckpt, cfg, max_model_len=MAX_LEN, load_head=False)
    ids = torch.randint(cfg.vocab_size, (64,), generator=torch.Generator().manual_seed(2)).cuda()
    batch = Dsv4Batch.from_lens([64], [64], cfg.sliding_window)
    a = from_dir.forward_batch(ids, batch, from_dir.new_caches(1, MAX_LEN))
    b = from_map.forward_batch(ids, batch, from_map.new_caches(1, MAX_LEN))
    assert _same(a, b)


@gpu
def test_compressed_rope_matches_vllm():
    """C4 / C128 layers: YaRN, bit-equal to vLLM's cache."""
    pytest.importorskip("vllm")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekV4ScalingRotaryEmbedding,
    )

    from rl_engine.models.deepseek_v4.rope import rope_cos_sin

    cfg = _config()
    for ratio in (4, 128):
        base = cfg.compress_rope_theta
        with torch.device("cuda"), set_current_vllm_config(VllmConfig()):
            ref = DeepseekV4ScalingRotaryEmbedding(
                cfg.head_dim,
                cfg.qk_rope_head_dim,
                ROPE["original_max_position_embeddings"],
                base,
                False,
                float(ROPE["factor"]),
                torch.bfloat16,
                beta_fast=ROPE["beta_fast"],
                beta_slow=ROPE["beta_slow"],
                mscale=0,
                mscale_all_dim=0,
            ).cos_sin_cache
        ours = rope_cos_sin(cfg, ratio, MAX_LEN, "cuda")
        assert ours.dtype == ref.dtype == torch.float32
        assert torch.equal(ours.view(torch.int32), ref[:MAX_LEN].view(torch.int32)), ratio


@gpu
def test_window_rope_matches_deepseek():
    """Sliding-window-only layers: plain RoPE, bit-equal to ``precompute_freqs_cis`` of
    DeepSeek's inference ``model.py`` (``$DSV4_REF`` holds model.py and kernel.py)."""
    import sys

    from rl_engine.models.deepseek_v4.rope import rope_cos_sin

    ref_dir = pathlib.Path(os.environ.get("DSV4_REF", "/nonexistent"))
    if not (ref_dir / "model.py").exists():
        pytest.skip("set DSV4_REF to DeepSeek-V4's inference directory")
    pytest.importorskip("tilelang")  # model.py imports kernel.py
    sys.path.insert(0, str(ref_dir))
    try:
        import model as ref
    finally:
        sys.path.remove(str(ref_dir))
    cfg = _config()
    with torch.device("cuda"):
        cis = ref.precompute_freqs_cis(
            cfg.qk_rope_head_dim, MAX_LEN, 0, cfg.rope_theta, ROPE["factor"], 32, 1
        )
    want = torch.cat((cis.real, cis.imag), dim=-1)
    ours = rope_cos_sin(cfg, 1, MAX_LEN, "cuda")
    assert ours.dtype == want.dtype == torch.float32
    assert torch.equal(ours.view(torch.int32), want.view(torch.int32))


def test_checkpoint_spec_matches_flash_headers():
    """Names, dtypes and shapes against DeepSeek-V4-Flash's safetensors headers
    (``scripts/make_dsv4_toy_checkpoint.py --fetch`` writes them to ``$DSV4_META``)."""
    from rl_engine.models.deepseek_v4.config import Dsv4Config
    from rl_engine.models.deepseek_v4.loader import checkpoint_spec

    meta = pathlib.Path(os.environ.get("DSV4_META", "/nonexistent"))
    if not (meta / "headers.json").exists():
        pytest.skip("set DSV4_META to a directory with headers.json and config.json")
    headers = json.loads((meta / "headers.json").read_text())
    cfg = Dsv4Config.from_pretrained(meta)
    layers = 1 + max(int(k.split(".")[1]) for k in headers if k.startswith("layers."))
    names = {
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.float8_e4m3fn: "F8_E4M3",
        torch.float8_e8m0fnu: "F8_E8M0",
        torch.int8: "I8",
        torch.int64: "I64",
    }
    spec = {k: (list(s), names[d]) for k, (s, d) in checkpoint_spec(cfg, layers).items()}
    real = {k: (v["shape"], v["dtype"]) for k, v in headers.items()}
    assert spec == real


def test_config_rejects_unsupported():
    with pytest.raises(ValueError, match="hc_sinkhorn_iters"):
        _config(hc_sinkhorn_iters=10).check_supported()
    with pytest.raises(ValueError, match="compress_ratios"):
        _config(compress_ratios=[0, 0, 8, 128]).check_supported()
    with pytest.raises(ValueError, match="n_routed_experts"):
        _config(n_routed_experts=16).check_supported()
    with pytest.raises(ValueError, match="compress_ratios has"):
        _config(compress_ratios=[0, 0])
    _config().check_supported()
