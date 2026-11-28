# SPDX-License-Identifier: Apache-2.0
"""packed_w4a16 N-slices concatenated equal the full projection, bit for bit."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)

E, K, I, H = 8, 512, 256, 512


def _weights(n, k, gen):
    codes = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, generator=gen)
    scales = torch.randint(118, 130, (E, n, k // 32), dtype=torch.uint8, generator=gen)
    return codes.cuda(), scales.cuda()


def _routing(rows):
    experts = torch.sort(torch.randint(0, E, (rows,))).values
    counts = torch.bincount(experts, minlength=E)
    return torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int64).cuda()


@pytest.mark.parametrize("size", [2, 4, 8])
def test_n_slices_equal_full(size):
    from rl_engine.backends.rocm.moe.packed_w4a16 import packed_w4a16

    gen = torch.Generator().manual_seed(1)
    rows = 77
    x = torch.randn(rows, K, generator=gen).bfloat16().cuda()
    offsets = _routing(rows)
    gammas = torch.rand(rows, generator=gen).float().cuda()
    codes, scales = _weights(H, K, gen)
    full = packed_w4a16(x, codes, scales, offsets, gammas)
    w = H // size
    parts = [
        packed_w4a16(
            x,
            codes[:, r * w : (r + 1) * w].contiguous(),
            scales[:, r * w : (r + 1) * w].contiguous(),
            offsets,
            gammas,
        )
        for r in range(size)
    ]
    assert torch.equal(torch.cat(parts, dim=1).view(torch.int16), full.view(torch.int16))


@pytest.mark.parametrize("size", [2, 4, 8])
def test_i_slices_gate_up_activation_equal_full(size):
    from rl_engine.backends.rocm.moe.packed_w4a16 import packed_w4a16
    from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func

    gen = torch.Generator().manual_seed(2)
    rows = 53
    x = torch.randn(rows, K, generator=gen).bfloat16().cuda()
    offsets = _routing(rows)
    gcodes, gscales = _weights(I, K, gen)
    ucodes, uscales = _weights(I, K, gen)
    fused_c = torch.cat((gcodes, ucodes), dim=1).contiguous()
    fused_s = torch.cat((gscales, uscales), dim=1).contiguous()
    z = packed_w4a16(x, fused_c, fused_s, offsets)
    ref = torch.empty(rows, I, device="cuda", dtype=torch.bfloat16)
    swiglu_limit_func(ref, z, 10.0)
    w = I // size
    outs = []
    for r in range(size):
        sl = slice(r * w, (r + 1) * w)
        c = torch.cat((gcodes[:, sl], ucodes[:, sl]), dim=1).contiguous()
        s = torch.cat((gscales[:, sl], uscales[:, sl]), dim=1).contiguous()
        zs = packed_w4a16(x, c, s, offsets)
        o = torch.empty(rows, w, device="cuda", dtype=torch.bfloat16)
        swiglu_limit_func(o, zs, 10.0)
        outs.append(o)
    assert torch.equal(torch.cat(outs, dim=1).view(torch.int16), ref.view(torch.int16))
