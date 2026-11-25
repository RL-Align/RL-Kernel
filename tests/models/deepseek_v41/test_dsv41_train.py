# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4.1 training forward / backward:

- the KV group's gradient crosses layers: with the loss on a reuse layer only and only the
  KV source's input a leaf, dL/dx_source (which flows only through the shared compressed KV)
  matches finite differences, at ratio 2 and ratio 1;
- the model's forward_train equals forward_batch byte for byte, and every LoRA adapter
  (attention, routed and shared expert) gets a finite, nonzero gradient.
"""

from __future__ import annotations

import pytest
import torch


def _ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return hasattr(ext, "dsv41_compress_main") and hasattr(ext, "dsv4_compressor_backward")


pytestmark = pytest.mark.skipif(not _ready(), reason="CUDA rl_engine._C with DSv4.1 required")

T, MAX_LEN = 400, 1024


def _stack():
    from rl_engine.models.deepseek_v4.rope import rope_cos_sin
    from rl_engine.models.deepseek_v41 import Dsv41Config, toy_config
    from rl_engine.models.deepseek_v41.attention import (
        Dsv41Attention,
        attention_configs,
        new_group_caches,
        random_layer,
    )

    cfg = Dsv41Config.from_dict(toy_config())
    layers, caches = [], []
    for i, c in enumerate(attention_configs(cfg, MAX_LEN)):
        layers.append(
            Dsv41Attention(
                c, random_layer(c, seed=200 + i), rope_cos_sin(cfg, 2 if c.ratio else 1, MAX_LEN)
            )
        )
        group = caches[cfg.layer_role(i).kv_source] if c.ratio and not c.is_kv_source else None
        caches.append(new_group_caches(c, 1, MAX_LEN, seed=i, group=group))
    return cfg, layers, caches


@pytest.mark.parametrize("source,reuse", [(1, 2), (3, 4)])  # ratio 2, ratio 1
def test_group_gradient_reaches_the_kv_source(source, reuse):
    from rl_engine.models.deepseek_v41.attention import Dsv41Batch
    from rl_engine.models.deepseek_v41.attention_backward import attention_train

    _, layers, caches = _stack()
    batch = Dsv41Batch.from_lens([T], [T], 128)
    g = torch.Generator(device="cuda").manual_seed(0)

    def unit(*shape):
        z = torch.randn(*shape, generator=g, device="cuda")
        return z * torch.rsqrt(z.square().mean(-1, keepdim=True))

    x_src = unit(T, 5120).bfloat16()
    x_reuse = unit(T, 5120).bfloat16()
    r = torch.randn(T, 5120, generator=g, device="cuda")

    def loss(xs, train):
        shared: dict = {}
        if train:
            _, handle = attention_train(layers[source], xs, batch, caches[source], shared)
            y, _ = attention_train(layers[reuse], x_reuse, batch, caches[reuse], shared, handle)
        else:
            layers[source].forward_batch(xs, batch, caches[source], shared)
            y = layers[reuse].forward_batch(x_reuse, batch, caches[reuse], shared)
        return (y.float() * r).sum()

    leaf = x_src.clone().requires_grad_(True)
    loss(leaf, True).backward()
    grad = leaf.grad.float()
    assert torch.isfinite(grad).all() and float(grad.abs().max()) > 0
    pred, obs = [], []
    for _ in range(12):
        d = torch.randn(T, 5120, generator=g, device="cuda")
        step = 0.2 * d  # smaller steps drown in the FP8 KV's rounding
        with torch.no_grad():
            lp = float(loss((x_src.float() + step).bfloat16(), False))
            lm = float(loss((x_src.float() - step).bfloat16(), False))
        actual = (x_src.float() + step).bfloat16().float() - (
            x_src.float() - step
        ).bfloat16().float()
        pred.append(float((grad * actual).sum()))
        obs.append(lp - lm)
    corr = float(torch.corrcoef(torch.tensor([pred, obs]))[0, 1])
    print("predicted", [round(v, 2) for v in pred], "\nobserved ", [round(v, 2) for v in obs])
    scale = sum(abs(o) for o in obs) / sum(abs(p) for p in pred)
    assert corr > 0.95 and 0.8 < scale < 1.2, (corr, scale)


@pytest.fixture(scope="module")
def model():
    from rl_engine.models.deepseek_v41 import Dsv41Config, toy_config
    from rl_engine.models.deepseek_v41.loader import load_model, random_checkpoint

    cfg = Dsv41Config.from_dict({**toy_config(vocab_size=1024), "moe_intermediate_size": 256})
    return load_model(
        random_checkpoint(cfg, seed=1), cfg, max_model_len=MAX_LEN, token_map=list(range(1024))
    )


def test_forward_train_bytes_and_lora_gradients(model):
    from rl_engine.models.deepseek_v4.model import attach_lora
    from rl_engine.models.deepseek_v41.attention import Dsv41Batch
    from rl_engine.models.deepseek_v41.engram import ngram_windows

    attach_lora(model, rank=16, alpha=32, zero_b=False, seed=3)
    params = model.lora_parameters()
    assert params and all(p.requires_grad for p in params)
    ids = torch.randint(0, 1024, (T,), generator=torch.Generator().manual_seed(4)).cuda()
    pos = torch.arange(T, device="cuda")
    windows = ngram_windows([ids], [pos], model.config.engram_max_ngram_size)
    batch = Dsv41Batch.from_lens([T], [T], 128)
    out = model.forward_train(ids, batch, model.new_caches(1, seed=1), engram_windows=windows)
    ref = model.forward_batch(ids, batch, model.new_caches(1, seed=2), engram_windows=windows)
    assert torch.equal(out.detach().view(torch.int16), ref.view(torch.int16))
    r = torch.randn(
        out.shape, generator=torch.Generator(device="cuda").manual_seed(5), device="cuda"
    )
    (out.float() * r).sum().backward()
    names = []
    for layer in model.layers:
        names += [f"layer {layer.index} attn {n}" for n in layer.attn.w.lora for _ in (0, 1)]
        names += [
            f"layer {layer.index} moe {n}" for n, _ in layer.moe.lora.adapters() for _ in (0, 1)
        ]
    for name, p in zip(names, params):
        assert p.grad is not None, name
        assert torch.isfinite(p.grad.float()).all(), name
        assert float(p.grad.float().abs().max()) > 0, name


def test_activation_recompute_same_bytes_same_gradients_less_memory(model):
    from rl_engine.models.deepseek_v4.model import attach_lora
    from rl_engine.models.deepseek_v41.attention import Dsv41Batch
    from rl_engine.models.deepseek_v41.engram import ngram_windows

    attach_lora(model, rank=16, alpha=32, zero_b=False, seed=9)
    params = model.lora_parameters()
    ids = torch.randint(0, 1024, (T,), generator=torch.Generator().manual_seed(6)).cuda()
    pos = torch.arange(T, device="cuda")
    windows = ngram_windows([ids], [pos], model.config.engram_max_ngram_size)
    batch = Dsv41Batch.from_lens([T], [T], 128)
    r = torch.randn(T, 5120, generator=torch.Generator(device="cuda").manual_seed(7), device="cuda")

    def run(recompute):
        for p in params:
            p.grad = None
        caches = model.new_caches(1, seed=3)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        out = model.forward_train(ids, batch, caches, engram_windows=windows, recompute=recompute)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - before
        (out.float() * r).sum().backward()
        return out.detach(), [p.grad.float().clone() for p in params], held

    out_a, grads_a, held_a = run(False)
    out_b, grads_b, held_b = run(True)
    assert torch.equal(out_a.view(torch.int16), out_b.view(torch.int16))
    for i, (a, b) in enumerate(zip(grads_a, grads_b)):
        cos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
        rel = float((a - b).norm() / a.norm().clamp_min(1e-30))
        assert cos > 0.999 and rel < 5e-2, (i, cos, rel)
    print(f"held activations: {held_a / 2**20:.1f} MiB without, {held_b / 2**20:.1f} MiB with")
    assert held_b < 0.5 * held_a, (held_a, held_b)
