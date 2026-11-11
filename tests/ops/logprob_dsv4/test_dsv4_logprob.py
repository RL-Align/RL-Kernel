# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 selected-token logprob (dsv4-logprob-v1): CUDA vs golden bytes, TP-shard and
batch invariance, accuracy, backward."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch import logprob_dsv4 as gold


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        import rl_engine._C as ext
        from rl_engine.kernels.ops.cuda.deepgemm import deepgemm_available
    except ImportError:
        return False
    return hasattr(ext, "dsv4_logprob_tile_stats") and deepgemm_available()


pytestmark = pytest.mark.skipif(not _ready(), reason="CUDA + DeepGEMM + dsv4_logprob build")

H, V = 4096, 129280


def _same(a, b) -> bool:
    if a.dtype == torch.bfloat16:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def _logits(n, v, scale, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn(n, v, generator=g, device="cuda") * scale
    t = torch.randint(v, (n,), generator=g, device="cuda")
    return z, t


@pytest.mark.parametrize("v", [64 * 40, 64 * 33, V])
@pytest.mark.parametrize("scale", [1.0, 30.0])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_matches_golden_bytes(v, scale, temperature):
    from rl_engine.kernels.ops.cuda.logprob_dsv4 import (
        logprob_backward_cuda,
        logprob_from_logits_cuda,
        tile_stats_cuda,
    )

    z, t = _logits(37, v, scale, seed=v + int(scale))
    m, s = tile_stats_cuda(z, v // 64, temperature)
    gm, gs = gold.tile_stats(z, temperature)
    assert _same(m, gm) and _same(s, gs)
    logp, lse = logprob_from_logits_cuda(z, t, temperature)
    glogp, glse = gold.selected_logprob(z, t, temperature)
    assert _same(logp, glogp) and _same(lse, glse)
    g = torch.randn(37, generator=torch.Generator(device="cuda").manual_seed(3), device="cuda")
    dz = logprob_backward_cuda(z, t, lse, g, temperature)
    assert _same(dz, gold.selected_logprob_backward(z, t, glse, g, temperature))


@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_golden_accuracy(temperature):
    z, t = _logits(64, 64 * 200, 8.0, seed=1)
    logp, lse = gold.selected_logprob(z, t, temperature)
    ref = torch.log_softmax(z.double() / temperature, dim=-1)
    want = ref.gather(1, t.view(-1, 1)).view(-1)
    assert torch.allclose(logp.double(), want, atol=2e-5, rtol=0)
    want_lse = torch.logsumexp(z.double() / temperature, dim=-1)
    assert torch.allclose(lse.double(), want_lse, atol=2e-5, rtol=0)


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_vocab_shards_give_same_bytes(tp):
    """Each TP rank computes its tiles' partials; gathered, they equal the full stats."""
    from rl_engine.kernels.ops.cuda.logprob_dsv4 import (
        logprob_backward_cuda,
        logprob_from_logits_cuda,
        merge_cuda,
        tile_stats_cuda,
    )

    z, t = _logits(19, V, 4.0, seed=tp)
    full_logp, full_lse = logprob_from_logits_cuda(z, t)
    shard, tile = V // tp, V // 64
    parts = [tile_stats_cuda(z[:, r * shard : (r + 1) * shard], tile) for r in range(tp)]
    m = torch.cat([p[0] for p in parts], 1)
    s = torch.cat([p[1] for p in parts], 1)
    logp, lse = merge_cuda(m, s, z.gather(1, t.view(-1, 1)).view(-1))
    assert _same(logp, full_logp) and _same(lse, full_lse)
    g = torch.randn(19, device="cuda")
    full_dz = logprob_backward_cuda(z, t, lse, g)
    dz = torch.cat(
        [
            logprob_backward_cuda(
                z[:, r * shard : (r + 1) * shard], t, lse, g, col_offset=r * shard
            )
            for r in range(tp)
        ],
        1,
    )
    assert _same(dz, full_dz)


@pytest.fixture(scope="module")
def head():
    g = torch.Generator(device="cuda").manual_seed(0)
    return (torch.randn(V, H, generator=g, device="cuda") / H**0.5).bfloat16()


def _hidden(n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h = (torch.randn(n, H, generator=g, device="cuda") * 2).bfloat16()
    return h, torch.randint(V, (n,), generator=g, device="cuda")


def test_selected_logprob_batch_invariant(head):
    from rl_engine.kernels.ops.cuda.logprob_dsv4 import selected_logprob

    h, t = _hidden(300, 1)
    logp, lse = selected_logprob(h, head, t)
    for lo, hi in ((0, 1), (5, 6), (17, 81), (299, 300)):
        a, b = selected_logprob(h[lo:hi].clone(), head, t[lo:hi].clone())
        assert _same(a, logp[lo:hi]) and _same(b, lse[lo:hi])
    a, b = selected_logprob(h, head, t, chunk_rows=7)
    assert _same(a, logp) and _same(b, lse)
    a, _ = selected_logprob(h, head, t)
    assert _same(a, logp)


def test_selected_logprob_accuracy(head):
    from rl_engine.kernels.ops.cuda.logprob_dsv4 import selected_logprob

    h, t = _hidden(64, 2)
    logp, _ = selected_logprob(h, head, t, temperature=0.8)
    ref = torch.log_softmax((h.double() @ head.double().t()) / 0.8, dim=-1)
    want = ref.gather(1, t.view(-1, 1)).view(-1)
    assert torch.allclose(logp.double(), want, atol=1e-3, rtol=0)


@pytest.mark.parametrize("temperature", [1.0, 0.8])
def test_backward(head, temperature):
    from rl_engine.kernels.ops.cuda.logprob_dsv4 import selected_logprob

    h, t = _hidden(96, 3)
    g = torch.randn(96, generator=torch.Generator(device="cuda").manual_seed(4), device="cuda")

    def grad(x, targets, gg, chunk=4096):
        x = x.clone().requires_grad_(True)
        logp, _ = selected_logprob(x, head, targets, temperature, chunk_rows=chunk)
        (logp * gg).sum().backward()
        return x.grad

    dh = grad(h, t, g)
    assert dh.dtype == torch.bfloat16
    assert _same(grad(h, t, g), dh)  # run to run
    assert _same(grad(h, t, g, chunk=13), dh)  # chunking
    assert _same(grad(h[40:41], t[40:41], g[40:41]), dh[40:41])  # batch invariance

    x = h.double().requires_grad_(True)
    ref = torch.log_softmax((x @ head.double().t()) / temperature, dim=-1)
    (ref.gather(1, t.view(-1, 1)).view(-1) * g.double()).sum().backward()
    rel = float((dh.double() - x.grad).norm() / x.grad.norm())
    assert rel < 1e-2, rel


def test_rejects_bad_inputs(head):
    from rl_engine.kernels.ops.cuda.logprob_dsv4 import logprob_from_logits_cuda, selected_logprob

    h, t = _hidden(4, 5)
    with pytest.raises(ValueError, match="temperature"):
        selected_logprob(h, head, t, temperature=0.0)
    with pytest.raises(ValueError, match="multiple of 64"):
        selected_logprob(h, head[:100], t)
    with pytest.raises(ValueError, match="one target per row"):
        selected_logprob(h, head, t[:3])
    with pytest.raises(TypeError):
        selected_logprob(h.float(), head, t)
    with pytest.raises(RuntimeError, match="FP32"):
        logprob_from_logits_cuda(torch.zeros(2, 128, device="cuda", dtype=torch.bfloat16), t[:2])


def test_empty(head):
    from rl_engine.kernels.ops.cuda.logprob_dsv4 import selected_logprob

    h = torch.empty(0, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    logp, lse = selected_logprob(h, head, torch.empty(0, dtype=torch.long, device="cuda"))
    assert logp.shape == (0,) and lse.shape == (0,)
    logp.sum().backward()
    assert h.grad.shape == (0, H)
