# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 model training forward / backward with LoRA (full model, end to end).

- ``forward_train`` gives the bytes of ``forward_batch`` (the rollout / recompute forward).
- The selected-token log-prob loss reaches every LoRA adapter with a finite, nonzero grad.
- The gradient agrees with a finite difference along a random direction of all adapters.
"""

from __future__ import annotations

import pytest
import torch

from tests.models.deepseek_v4.test_dsv4_model import MAX_LEN, _config, _ready

pytestmark = pytest.mark.skipif(
    not _ready(), reason="SM90, DeepGEMM, FlashMLA and a KERNEL_ALIGN_MOE_SM90=1 build"
)


def _cudnn():
    from rl_engine.kernels.ops.cuda.dsa_backward import cudnn_dsa_available

    return cudnn_dsa_available()


@pytest.fixture(scope="module")
def model():
    from rl_engine.models.deepseek_v4.loader import load_model, random_checkpoint

    cfg = _config()
    return load_model(random_checkpoint(cfg, seed=0), cfg, max_model_len=MAX_LEN)


@pytest.fixture(scope="module")
def hash_model():
    """All four layers hash-routed: no top-k routing can flip under a parameter step
    (hash routing depends only on the token ids). With <= 260 tokens the C4 indexer
    selects every entry, so it cannot flip either."""
    from rl_engine.models.deepseek_v4.loader import load_model, random_checkpoint

    cfg = _config(num_hash_layers=4)
    return load_model(random_checkpoint(cfg, seed=0), cfg, max_model_len=MAX_LEN)


LENS = [150, 260]


def _inputs(model, seed=1):
    gen = torch.Generator().manual_seed(seed)
    ids = [torch.randint(model.config.vocab_size, (n + 1,), generator=gen) for n in LENS]
    inp = torch.cat([x[:-1] for x in ids]).cuda()
    tgt = torch.cat([x[1:] for x in ids]).cuda()
    return inp, tgt


def _batch(model):
    from rl_engine.models.deepseek_v4.attention import Dsv4Batch

    return Dsv4Batch.from_lens(LENS, LENS, model.config.sliding_window)


def _loss(model, inp, tgt, train: bool):
    batch = _batch(model)
    caches = model.new_caches(len(LENS), MAX_LEN, seed=3)
    if train:
        hidden = model.forward_train(inp, batch, caches)
        return model.selected_logprob(hidden, tgt)[0].sum(), hidden
    with torch.no_grad():
        hidden = model.forward_batch(inp, batch, caches)
        return model.selected_logprob(hidden, tgt)[0].double().sum(), hidden


def _clear(model):
    for layer in model.layers:
        layer.attn.w.lora = {}
        layer.moe.lora = None


@pytest.mark.skipif(not _cudnn(), reason="cudnn-frontend DSA backward")
def test_train_forward_equals_rollout_forward(model):
    from rl_engine.models.deepseek_v4.model import attach_lora

    inp, tgt = _inputs(model)
    try:
        _, plain = _loss(model, inp, tgt, train=False)
        attach_lora(model, rank=16, alpha=32, zero_b=True)
        loss, train = _loss(model, inp, tgt, train=True)
        assert torch.equal(train.view(torch.int16), plain.view(torch.int16))
        loss.backward()  # zero B: gradients reach A only through B, which they do not
    finally:
        _clear(model)


@pytest.mark.skipif(not _cudnn(), reason="cudnn-frontend DSA backward")
def test_lora_gradients_reach_every_adapter_and_agree_with_finite_differences(hash_model):
    """Finite differences on this model are noisy: the FP8 / MX activation quantization makes
    the loss piecewise constant, and with scored routing a step can flip top-k choices
    (several units on a loss of about -3300). This model is hash-routed, so only the
    quantization noise remains. The check is statistical, over 12 parameter groups (each
    layer's attention, routed and shared-expert adapters):
    - predicted (g . step) and observed (L+ - L-) changes must correlate;
    - the strong-signal groups must agree in sign.
    The components are checked exactly against FP32 autograd in their own tests."""
    from rl_engine.models.deepseek_v4.model import attach_lora

    model = hash_model
    inp, tgt = _inputs(model, seed=2)
    try:
        attach_lora(model, rank=16, alpha=32, zero_b=False, seed=7)
        params = model.lora_parameters()
        loss, train = _loss(model, inp, tgt, train=True)
        _, rollout = _loss(model, inp, tgt, train=False)
        assert torch.equal(train.view(torch.int16), rollout.view(torch.int16))
        loss.backward()
        for i, p in enumerate(params):
            assert torch.isfinite(p.grad.float()).all(), i
            assert float(p.grad.float().abs().max()) > 0, i

        groups = []
        for layer in model.layers:
            groups.append([t for ad in layer.attn.w.lora.values() for t in (ad.a, ad.b)])
            for prefix in ("fc", "shared"):
                groups.append(
                    [
                        t
                        for n, ad in layer.moe.lora.adapters()
                        if n.startswith(prefix)
                        for t in (ad.a, ad.b)
                    ]
                )
        gen = torch.Generator(device="cuda").manual_seed(5)
        eps = 3e-2
        pred, obs = [], []
        for ps in groups:
            base = [p.detach().clone() for p in ps]
            dirs = [torch.randn(p.shape, generator=gen, device="cuda") for p in ps]
            vals, steps = [], []
            for sign in (1.0, -1.0):
                with torch.no_grad():
                    for p, b, d in zip(ps, base, dirs):
                        p.copy_((b.float() + sign * eps * d * b.float().abs().mean()).to(p.dtype))
                vals.append(float(_loss(model, inp, tgt, train=False)[0]))
                steps.append([p.detach().float().clone() for p in ps])
            with torch.no_grad():
                for p, b in zip(ps, base):
                    p.copy_(b)
            pred.append(sum(float((p.grad.float() * (a - b)).sum()) for p, a, b in zip(ps, *steps)))
            obs.append(vals[0] - vals[1])
        pt, ot = torch.tensor(pred), torch.tensor(obs)
        corr = float(torch.corrcoef(torch.stack([pt, ot]))[0, 1])
        print("predicted", [round(v, 3) for v in pred])
        print("observed ", [round(v, 3) for v in obs])
        print(f"correlation {corr:.3f}")
        assert corr > 0.7, corr
        for a, b in zip(pred, obs):
            if abs(a) > 1.0:
                assert a * b > 0, (a, b)
    finally:
        _clear(model)
