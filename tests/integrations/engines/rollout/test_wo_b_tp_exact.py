# SPDX-License-Identifier: Apache-2.0
"""TP-invariant wo_b: row/scale slice math (CPU) and N-slice bit invariance (ROCm GPU)."""

import pytest
import torch

from rl_engine.integrations.engines.rollout.vllm.group_projection import wo_b_row_partition

H, K = 4096, 8192


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_row_and_scale_slices_tile_the_original(tp):
    spans = [wo_b_row_partition(H, K, tp, r) for r in range(tp)]
    assert spans[0][0] == 0 and spans[-1][1] == H
    for r, (b0, b1, s0, s1) in enumerate(spans):
        assert (b0, b1) == (r * H // tp, (r + 1) * H // tp)
        assert (s0, s1) == (r * H // (128 * tp), (r + 1) * H // (128 * tp))
        assert (b1 - b0) == 128 * (s1 - s0)
        if r:
            assert spans[r - 1][1] == b0 and spans[r - 1][3] == s0


@pytest.mark.parametrize(
    "hidden,columns,tp,rank",
    [(4096, 8192, 3, 0), (512, 8192, 8, 0), (4096, 8000, 2, 0), (4096, 8192, 2, 2),
     (4096, 8192 - 128, 2, 0)],
)
def test_misaligned_fails_closed(hidden, columns, tp, rank):
    with pytest.raises(ValueError):
        wo_b_row_partition(hidden, columns, tp, rank)


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)
@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("m", [1, 7, 64])
def test_n_slice_outputs_concat_equals_full(tp, m):
    from rl_engine.backends.rocm.gemm.checkpoint_fp8 import CheckpointFP8Linear

    gen = torch.Generator().manual_seed(1234 + tp)
    # Finite E4M3 bytes (exclude NaN 0x7f/0xff) and moderate E8M0 exponents.
    codes = torch.randint(0, 0x78, (H, K), dtype=torch.uint8, generator=gen)
    codes |= torch.randint(0, 2, (H, K), dtype=torch.uint8, generator=gen) << 7
    scales = torch.randint(118, 124, (H // 128, K // 128), dtype=torch.uint8, generator=gen)
    weight = codes.view(torch.float8_e4m3fn).cuda()
    scale = scales.view(torch.float8_e8m0fnu).cuda()
    x = torch.randn(m, K, device="cuda").bfloat16()
    full = CheckpointFP8Linear(weight, scale)(x)[0]
    parts = []
    for r in range(tp):
        b0, b1, s0, s1 = wo_b_row_partition(H, K, tp, r)
        part = CheckpointFP8Linear(weight[b0:b1].contiguous(), scale[s0:s1].contiguous())
        parts.append(part(x)[0])
    out = torch.cat(parts, dim=-1)
    assert torch.equal(out.view(torch.int16), full.view(torch.int16))
