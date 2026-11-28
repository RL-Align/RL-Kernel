# SPDX-License-Identifier: Apache-2.0
"""Shared-expert TP partition math (CPU) and N-slice invariance of the actor ops (ROCm GPU)."""

import pytest
import torch

from rl_engine.integrations.engines.rollout.vllm.shared_expert_tp import shared_expert_partition

I, H = 2048, 4096


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_partition_covers_rows_once_in_rank_order(tp):
    parts = [shared_expert_partition(I, H, tp, r) for r in range(tp)]
    for key, total in (("gate_up", I), ("down", H)):
        spans = [p[key] for p in parts]
        assert spans[0][0] == 0 and spans[-1][1] == total
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
        for r0, r1, c0, c1 in spans:
            assert (r1 - r0) == total // tp and r0 == c0 * 128 and r1 == c1 * 128


def test_partition_fails_closed():
    with pytest.raises(ValueError):
        shared_expert_partition(I, H, 3, 0)
    with pytest.raises(ValueError):
        shared_expert_partition(I, H, 2, 2)
    with pytest.raises(ValueError):
        shared_expert_partition(I, H, 32, 0)  # 64 rows < block128


gpu = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def _operands(n, k, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.5).to(torch.float8_e4m3fn)
    # E8M0 bytes near 1.0 (127) with spread; valid exponents only.
    s = torch.randint(122, 130, (n // 128, k // 128), generator=g, dtype=torch.uint8)
    return w.cuda(), s.view(torch.float8_e8m0fnu).cuda()


def _bits(t):
    return t.contiguous().view(torch.uint8)


@gpu
@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("m", [1, 7, 33])
@pytest.mark.parametrize("name", ["gate_up", "down"])
def test_actor_linear_n_slice_invariance(tp, m, name):
    from rl_engine.backends.rocm.gemm.checkpoint_fp8 import CheckpointFP8Linear

    n, k = (I, H) if name == "gate_up" else (H, I)
    w, s = _operands(n, k, seed=n + tp)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        full = CheckpointFP8Linear(w, s)(x)[0]
        parts = []
        for r in range(tp):
            r0, r1, c0, c1 = shared_expert_partition(I, H, tp, r)[name]
            parts.append(
                CheckpointFP8Linear(w[r0:r1].contiguous(), s[c0:c1].contiguous())(x)[0]
            )
    assert all(p.shape == (m, n // tp) for p in parts)
    assert torch.equal(_bits(torch.cat(parts, dim=-1)), _bits(full))
