# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``cp_full_attention`` (ring): every rank equals the unsharded H3 attention bitwise.

The multi-process cases run the ring on real NCCL groups and compare each rank's rows of the
output and of dQ / dK / dV with the unsharded op computed in the same process. They need the
strict H3 CUDA extension (``RL_KERNEL_H3_STRICT=1`` build) and one GPU per rank.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
from rl_engine.validation.distributed.h3_ws2 import run_world


def _world(world, target, inputs, **kwargs):
    """Plain NCCL ranks: no fixed-tree collective, so any world size (3, 5, ...) works."""

    return run_world(world, target, inputs, deterministic_collective=False, **kwargs)


needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def gpus(n):
    return pytest.mark.skipif(torch.cuda.device_count() < n, reason=f"needs {n} GPUs")


def _strict_extension():
    """Skip unless the strict H3 attention extension is built."""
    from rl_engine.backends.cuda.attention import h3_full_attention as fa

    try:
        fa._extension()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _qkvg(length, heads, seed=7):
    gen = torch.Generator().manual_seed(seed)

    def draw():
        return torch.randn(1, length, heads, 128, generator=gen).bfloat16().cuda()

    return draw(), draw(), draw(), draw()


def _ring_target(collective, dummy, *, length, heads, chunk_rows, mode):
    """One rank of the ring / Ulysses check against the unsharded attention."""
    import torch.distributed as dist

    from rl_engine.backends.cuda.model_specific.minimax_h3 import ring_attention as ring

    del collective, dummy
    cp, rank = dist.get_world_size(), dist.get_rank()
    shard = par.SeqShard(length, cp, rank)
    q, k, v, grad = _qkvg(length, heads)
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
    ref = ring.local_attention(*leaves, chunk_rows)
    ref.backward(grad)
    expected = {
        "out": ref.detach(),
        "dq": leaves[0].grad,
        "dk": leaves[1].grad,
        "dv": leaves[2].grad,
    }

    mine = [par.scatter_packed(t, shard, dim=1).clone().requires_grad_(True) for t in (q, k, v)]
    if mode == "ring":
        out = ring.ring_attention(*mine, shard, dist.group.WORLD, chunk_rows)
    else:  # Ulysses, for the layouts it can express
        out = par.ulysses_attention(
            *mine,
            lambda a, b, c: ring.local_attention(a, b, c, chunk_rows),
            shard,
            dist.group.WORLD,
        )
    out.backward(par.scatter_packed(grad, shard, dim=1))
    got = {"out": out.detach(), "dq": mine[0].grad, "dk": mine[1].grad, "dv": mine[2].grad}
    result = {"rank": torch.tensor(rank), "rows": torch.tensor([shard.lo, shard.hi])}
    for name in got:
        want = par.scatter_packed(expected[name], shard, dim=1)
        result[f"equal_{name}"] = torch.tensor(int(torch.equal(got[name], want)))
    return result


def _check(ranks):
    """Every rank's output and dQ / dK / dV equal the unsharded ones."""
    for r in ranks:
        for name in ("out", "dq", "dk", "dv"):
            assert int(r[f"equal_{name}"]) == 1, f"rank {int(r['rank'])}: {name} differs"


@needs_gpu
class TestSharding:
    def test_ring_validates_its_inputs(self):
        """Refuses non-BF16 inputs, wrong local rows, unequal shapes and a bad ``chunk_rows``."""
        from rl_engine.backends.cuda.model_specific.minimax_h3 import ring_attention as ring

        shard = par.SeqShard(4, 1, 0)
        q = torch.zeros(1, 4, 7, 128, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="BF16"):
            ring.ring_attention(q.float(), q.float(), q.float(), shard)
        with pytest.raises(ValueError, match="local rows"):
            ring.ring_attention(q, q, q, par.SeqShard(5, 1, 0))
        with pytest.raises(ValueError, match="equal"):
            ring.ring_attention(q, q[:, :, :3], q, shard)
        with pytest.raises(ValueError, match="chunk_rows"):
            ring.ring_attention(q, q, q, shard, chunk_rows=0)


@needs_gpu
@pytest.mark.parametrize("cp", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("length", [1, 2, 7, 130, 261])
def test_ring_equals_unsharded_attention(cp, length):
    """Ring attention on ``cp`` ranks equals the unsharded op bitwise."""
    if torch.cuda.device_count() < cp:
        pytest.skip(f"needs {cp} GPUs")
    _strict_extension()
    _check(
        _world(
            cp,
            _ring_target,
            {"dummy": torch.zeros(1)},
            length=length,
            heads=7,
            chunk_rows=128,
            mode="ring",
        )
    )


@needs_gpu
@gpus(3)
@pytest.mark.parametrize("chunk_rows", [1, 5, 33])
def test_ring_is_independent_of_the_chunk_boundaries(chunk_rows):
    """Shard edges rarely fall on chunk edges; the dK/dV chain must not care."""

    _strict_extension()
    _check(
        _world(
            3,
            _ring_target,
            {"dummy": torch.zeros(1)},
            length=130,
            heads=7,
            chunk_rows=chunk_rows,
            mode="ring",
        )
    )


@needs_gpu
@gpus(2)
def test_ring_handles_the_full_head_count_and_a_rank_with_no_rows():
    _strict_extension()
    _check(
        _world(
            2,
            _ring_target,
            {"dummy": torch.zeros(1)},
            length=1,
            heads=56,
            chunk_rows=128,
            mode="ring",
        )
    )


@needs_gpu
@gpus(2)
def test_ring_and_ulysses_agree_where_ulysses_applies():
    """Where Ulysses applies, ring and Ulysses both equal the unsharded op."""
    _strict_extension()
    for mode in ("ring", "ulysses"):
        _check(
            _world(
                2,
                _ring_target,
                {"dummy": torch.zeros(1)},
                length=261,
                heads=14,
                chunk_rows=128,
                mode=mode,
            )
        )


@needs_gpu
@gpus(3)
def test_ulysses_cannot_express_cp3_but_the_ring_can():
    """56 heads (and 7 per TP8 rank) do not split over 3: only the ring covers CP3."""

    with pytest.raises(Exception, match="heads"):
        _world(
            3,
            _ring_target,
            {"dummy": torch.zeros(1)},
            length=9,
            heads=7,
            chunk_rows=128,
            mode="ulysses",
        )


# --- The whole block (block 0) with the H3 attention kernels and a ring or Ulysses CP -------------


def _assemble(ranks, key="out"):
    """Join the ranks' row spans into the full tensor; ranks holding a span must agree."""
    pieces: dict[tuple[int, int], torch.Tensor] = {}
    for r in ranks:
        span = tuple(int(v) for v in r["rows"])
        if span in pieces:
            assert torch.equal(pieces[span], r[key]), f"ranks disagree on rows {span}"
        pieces[span] = r[key]
    return torch.cat([pieces[s] for s in sorted(pieces)], dim=1)


BLOCK_GRIDS = [
    pytest.param(1, 2, False, "ring", 261, id="cp2_ring_len261", marks=gpus(2)),
    pytest.param(1, 3, False, "ring", 261, id="cp3_ring_len261", marks=gpus(3)),
    pytest.param(1, 4, False, "ring", 263, id="cp4_ring_len263", marks=gpus(4)),
    pytest.param(2, 2, False, "ring", 261, id="tp2_cp2_ring", marks=gpus(4)),
    pytest.param(2, 2, True, "ring", 0, id="tp2_cp2_sp_ring", marks=gpus(4)),
    pytest.param(1, 2, False, "ulysses", 261, id="cp2_ulysses_h3", marks=gpus(2)),
    pytest.param(2, 3, False, "ring", 261, id="tp2_cp3_ring", marks=gpus(6)),
    pytest.param(4, 2, False, "ring", 0, id="tp4_cp2_ring", marks=gpus(8)),
    pytest.param(4, 2, True, "ring", 0, id="tp4_cp2_sp_ring", marks=gpus(8)),
    pytest.param(1, 7, False, "ring", 261, id="cp7_ring", marks=gpus(7)),
]


@needs_gpu
@pytest.mark.parametrize(("tp", "cp", "sp", "mode", "length"), BLOCK_GRIDS)
def test_block_with_h3_attention_equals_single_gpu_bitwise(tp, cp, sp, mode, length):
    """A block on TP / CP / SP with the H3 attention equals the single-GPU block bitwise."""
    pytest.importorskip("triton")
    _strict_extension()
    from rl_engine.validation.models import h3_tp
    from rl_engine.validation.models.h3_weights import load_h3_block_weights

    ranks = _world(
        tp * cp,
        h3_tp.block_target,
        {"dummy": torch.zeros(1)},
        tp=tp,
        cp=cp,
        sp=sp,
        seq_len=length,
        backward=True,
        attention="h3_full",
        cp_mode=mode,
    )
    ref = h3_tp.block_case(
        load_h3_block_weights("cuda"),
        h3_tp.Grid(1, 1, 0, 0, attention="h3_full"),
        seq_len=length,
        backward=True,
    )
    assert torch.equal(_assemble(ranks).cpu(), ref["out"].cpu()), "forward differs"
    assert torch.equal(_assemble(ranks, "dx").cpu(), ref["dx"].cpu()), "input gradient differs"
    if cp == 1 and not sp:
        for name, axis in (
            ("attn.to_q.weight", 0),
            ("attn.to_k.weight", 0),
            ("attn.to_v.weight", 0),
        ):
            full = ref["dw." + name].cpu()
            shards = [r["dw." + name].cpu() for r in sorted(ranks, key=lambda r: r["rank"])]
            assert torch.equal(torch.cat(shards, dim=axis), full), f"dW {name} differs"
