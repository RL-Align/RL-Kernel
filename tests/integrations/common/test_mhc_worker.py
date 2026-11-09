# SPDX-License-Identifier: Apache-2.0
"""Exercise the two worker calling conventions, independent of framework installs."""
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from rl_engine.integrations.common.mhc_worker import P1Worker
from rl_engine.integrations.orchestrators.verl.mhc import bind_mhc_worker as verl_bind
from rl_engine.integrations.orchestrators.vime.mhc import bind_mhc_worker as vime_bind


def bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def stages(c, device="cpu"):
    rng = torch.Generator().manual_seed(75)

    def coef(*dims):
        return nn.Parameter(torch.randn(dims, generator=rng).to(device) * 0.2)

    def hc():
        return NS(fn=coef(24, 4 * c), scale=coef(3), base=coef(24))

    def norm():
        return NS(weight=coef(c), variance_epsilon=1e-6)

    a, f = hc(), hc()
    an, fn = norm(), norm()
    cfg = NS(rms_norm_eps=1e-6, hc_eps=1e-6, hc_mult=4, hc_sinkhorn_iters=2)
    h = NS(hc_fn=coef(4, 4 * c), hc_scale=coef(1), hc_base=coef(4))
    tl = NS(
        config=cfg,
        attn_hc=a,
        ffn_hc=f,
        _mhc_pre=None,
        _mhc_post=None,
        input_layernorm=an,
        post_attention_layernorm=fn,
        self_attn=lambda x, metadata: x + metadata,
        mlp=lambda x, input_ids: x * input_ids[:, None],
    )
    train = NS(rlk_p1_head_hook=True, layers={"0": tl}, hc_head=h, norm=norm())

    def copy(v):
        return nn.Parameter(v.detach().clone())

    rl = NS(
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=2.0,
        hc_sinkhorn_iters=2,
        attn=lambda pos, x, _: x + pos[:, None],
        ffn=lambda x, ids: x * ids[:, None],
        attn_norm=NS(weight=copy(an.weight), variance_epsilon=1e-6),
        ffn_norm=NS(weight=copy(fn.weight), variance_epsilon=1e-6),
        use_fused_mhc=True,
    )
    for name, source in (("attn", a), ("ffn", f)):
        for k in ("fn", "scale", "base"):
            setattr(rl, f"hc_{name}_{k}", copy(getattr(source, k)))
    rollout = NS(
        start_layer=0,
        end_layer=1,
        layers=[rl],
        hc_head_op=nn.Identity(),
        norm=NS(weight=copy(train.norm.weight), variance_epsilon=1e-6),
    )
    return train, rollout, NS(**{n: copy(v) for n, v in vars(h).items()})


@pytest.mark.parametrize("bind", [verl_bind, vime_bind])
def test_train_and_rollout_abis_across_updates(bind):
    train, rollout, h = stages(8)
    traces = [[], []]

    def observer(i):
        return lambda fw, role, version, backend, profile, name, v: traces[i].append(
            (version, name, v.clone())
        )

    tw = bind(train, "training", backend="reference", observer=observer(0))
    rw = bind(rollout, "rollout", backend="reference", observer=observer(1))
    assert not rollout.layers[0].use_fused_mhc
    assert tw.framework == rw.framework
    previous = None
    for version in range(3):
        if version:
            for worker, weight in ((tw, train.hc_head.hc_base), (rw, h.hc_base)):
                with worker.updating(version, lambda: None), torch.no_grad():
                    weight[0].add_(0.73)
        single = None
        for m in (1, 16, 17):
            gen = torch.Generator().manual_seed(5)
            x = torch.randn(m, 8, generator=gen).bfloat16()
            x[-1] = torch.arange(8, dtype=x.dtype) / 8
            ids = torch.ones(m, dtype=x.dtype)
            pos = ids / 8
            traces[0].clear()
            traces[1].clear()
            with tw.request(version):
                z = train.layers["0"].forward(
                    x, position_ids=pos, attention_metadata=pos[:, None], input_ids=ids
                )
                a = tw.norm(
                    tw.head(z, train.hc_head.hc_fn, train.hc_head.hc_scale, train.hc_head.hc_base),
                    train.norm.weight,
                )
            with rw.request(version), torch.no_grad():
                z, *tail = rollout.layers[0].forward(x[:, None, :].expand(-1, 4, -1), pos, ids)
                assert tail == [None, None, None]
                out = rollout.hc_head_op(z, h.hc_fn, h.hc_scale, h.hc_base, 1e-6, 1e-6)
                b = rollout.norm.forward(out)
            bits(a, b)
            assert len(traces[0]) == len(traces[1]) > 6
            for (ver, name, v), (ver2, name2, v2) in zip(*traces, strict=True):
                assert (ver, name) == (ver2, name2) and ver == version
                bits(v, v2)
            if single is None:
                single = a[-1].detach().clone()
            bits(single, a[-1])
            assert torch.isfinite(torch.autograd.grad(a.sum(), train.hc_head.hc_fn)[0]).all()
        if previous is not None:
            assert not torch.equal(previous, single)
        previous = single


def test_failed_reload_and_nested_update_do_not_publish_weights():
    w = P1Worker("vime", "rollout", backend="reference")
    with w.request(0), pytest.raises(RuntimeError, match="nest"), w.updating(1, lambda: None):
        pass
    with pytest.raises(OSError), w.updating(1, lambda: None):
        raise OSError("interrupted reload")
    with pytest.raises(RuntimeError, match="stale"), w.request(0):
        pass
    calls = []
    with w.updating(1, lambda: calls.append(1)):
        pass
    assert calls == [1, 1]
    with pytest.raises(RuntimeError, match="stale"), w.request(0):
        pass
    with w.request(1):
        pass
    with pytest.raises(RuntimeError, match="request"):
        w.head(None, None, None, None)


def test_reject_unpatched_training_worker_before_modification():
    m, _, _ = stages(8)
    del m.rlk_p1_head_hook
    with pytest.raises(ValueError, match="patch"):
        verl_bind(m, "training", backend="reference")
    assert not hasattr(m, "_rlk_p1")
