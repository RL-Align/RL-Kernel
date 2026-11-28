# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the TP-exact gather permutation and int-bit embedding combine."""

import pytest
import torch

from rl_engine.integrations.engines.rollout.vllm.tp_exact import (
    bits_to_dtype,
    capacity_bytes,
    combine_owner_bits,
    gather_wire_view,
    int_bits,
    last_dim_from_gathered,
)


def _simulate_gather(shards):
    # Collective semantics: dim-0 rank-ordered concat of each rank's [M, C] view.
    cols = shards[0].shape[-1]
    return torch.cat([s.reshape(-1, cols) for s in shards], dim=0)


@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("lead", [(5,), (3, 4), ()])
def test_last_dim_gather_matches_concat(size, lead):
    torch.manual_seed(0)
    shards = [torch.randn(*lead, 6) for _ in range(size)]
    out = last_dim_from_gathered(_simulate_gather(shards), size, lead)
    assert out.shape == (*lead, 6 * size)
    assert torch.equal(out, torch.cat(shards, dim=-1))


def test_last_dim_gather_rank_order_marked():
    shards = [torch.full((2, 3), float(r)) for r in range(4)]
    out = last_dim_from_gathered(_simulate_gather(shards), 4, (2,))
    assert out[0].tolist() == [0.0] * 3 + [1.0] * 3 + [2.0] * 3 + [3.0] * 3


def test_last_dim_gather_rejects_ragged_rows():
    with pytest.raises(ValueError):
        last_dim_from_gathered(torch.zeros(5, 2), 2, (2,))


def test_fp8_and_uint8_wire_view_roundtrip():
    fp8 = torch.randn(4, 8).to(torch.float8_e4m3fn)
    wire = gather_wire_view(fp8)
    assert wire.dtype == torch.uint8 and wire.shape == fp8.shape
    shards = [fp8, fp8.flip(0)]
    wires = [gather_wire_view(s) for s in shards]
    out = last_dim_from_gathered(_simulate_gather(wires), 2, (4,)).view(torch.float8_e4m3fn)
    assert torch.equal(out.view(torch.uint8), torch.cat(shards, dim=-1).view(torch.uint8))
    f32 = torch.randn(3, 5)
    assert gather_wire_view(f32) is f32
    with pytest.raises(TypeError):
        gather_wire_view(torch.zeros(2, dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_embedding_bit_combine_preserves_negative_zero(dtype):
    rows = torch.tensor([[-0.0, 1.5, -2.0, 0.0], [3.0, -0.0, -0.0, -7.25]], dtype=dtype)
    # Rank 0 owns row 0, rank 1 owns row 1; each zero-fills the other.
    zero = torch.zeros(2, 4, dtype=dtype)
    r0, r1 = zero.clone(), zero.clone()
    r0[0], r1[1] = rows[0], rows[1]
    out = combine_owner_bits([int_bits(r0), int_bits(r1)], dtype)
    assert out.dtype == dtype
    assert torch.equal(out.view(_same_int(dtype)), rows.view(_same_int(dtype)))
    # The float sum would flip -0.0 to +0.0; the bit sum must not.
    assert torch.signbit(out[0, 0]) and torch.signbit(out[1, 1])
    assert not torch.signbit((r0 + r1)[0, 0])


def _same_int(dtype):
    return {2: torch.int16, 4: torch.int32}[torch.empty((), dtype=dtype).element_size()]


def test_embedding_bit_combine_owners_never_overlap():
    torch.manual_seed(1)
    size, tokens, dim = 4, 64, 8
    table = torch.randn(tokens, dim).to(torch.bfloat16)
    table[3] = -0.0
    ids = torch.randint(0, tokens, (50,))
    ids[0] = 3
    shard = tokens // size
    parts = []
    owners = torch.zeros(len(ids), dtype=torch.int64)
    for rank in range(size):
        lo, hi = rank * shard, (rank + 1) * shard
        mine = (ids >= lo) & (ids < hi)
        owners += mine.long()
        local = table[(ids - lo).clamp(0, shard - 1) + lo]
        bits = local.view(torch.int16)
        bits = bits.masked_fill(~mine.unsqueeze(-1), 0)
        parts.append(bits.to(torch.int32))
    assert bool((owners == 1).all())
    out = combine_owner_bits(parts, torch.bfloat16)
    assert torch.equal(out.view(torch.int16), table[ids].view(torch.int16))


def test_fp8_bits_roundtrip():
    x = torch.randn(3, 4).to(torch.float8_e4m3fn)
    back = bits_to_dtype(int_bits(x), torch.float8_e4m3fn)
    assert torch.equal(back.view(torch.uint8), x.view(torch.uint8))


def test_capacity_bytes():
    assert capacity_bytes(16384, 8) == 16384 * 4096 * 2 * 8
