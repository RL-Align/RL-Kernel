# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The pieces of the full-model WS2 harness: row spans, exact slice / gather, first drift, chaining.

``h3_parallel_model.parallel_model_execution`` itself needs the 33B checkpoint and is exercised by
``tools/validation/models/h3_parallel_full_model.py``. Here the same slicing, gathering and block
body are chained over a few blocks with block 0's real weights, on real NCCL ranks, and compared
with the canonical single-GPU chain bit for bit, forward and backward.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
from rl_engine.validation.models import h3_parallel_model as pm
from rl_engine.validation.models.h3_ws2 import run_world

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def gpus(n):
    return pytest.mark.skipif(torch.cuda.device_count() < n, reason=f"needs {n} GPUs")


def _world(world, target, inputs, **kwargs):
    return run_world(world, target, inputs, deterministic_collective=False, **kwargs)


class TestSpans:
    @pytest.mark.parametrize(("tp", "cp", "length"), [(1, 1, 5), (2, 3, 40), (4, 2, 261)])
    def test_spans_tile_the_sequence_without_sp(self, tp, cp, length):
        spans = pm.rank_spans(tp, cp, length, sp=False)
        covered = sorted({s for s in spans})
        assert covered[0][0] == 0 and covered[-1][1] == length
        assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))
        assert len(spans) == tp * cp

    def test_sp_spans_are_disjoint_shares(self):
        spans = pm.rank_spans(4, 2, 40, sp=True)
        assert sorted(spans) == [(i * 5, i * 5 + 5) for i in range(8)]

    def test_sp_needs_divisible_rows(self):
        with pytest.raises(ValueError, match="divisible"):
            pm.rank_spans(4, 2, 12, sp=True)


class TestFirstDrift:
    def _case(self):
        ref = {f"block.{i}": torch.zeros(1, 8, 4) for i in range(4)}
        spans = [(0, 4), (4, 8)]
        tags = torch.tensor([1, 1, 0, 0, 0, 0, 2, 2])
        return ref, spans, tags

    def test_no_drift(self):
        ref, spans, tags = self._case()
        assert pm.first_drift(ref, {k: v.clone() for k, v in ref.items()}, spans, tags) is None

    def test_names_block_row_owner_and_modality(self):
        ref, spans, tags = self._case()
        seen = {k: v.clone() for k, v in ref.items()}
        seen["block.2"][0, 6, 1] = 1.0  # and everything after it differs too
        seen["block.3"][0, 5:, :] = 2.0
        report = pm.first_drift(ref, seen, spans, tags)
        assert report == {
            "node": "block.2",
            "first_row": 6,
            "rows_differing": 1,
            "owner_ranks": [1],
            "modality": "audio",
        }

    def test_replicated_rows_name_every_holder(self):
        ref, _, tags = self._case()
        seen = {k: v.clone() for k, v in ref.items()}
        seen["block.0"][0, 1, 0] = 1.0
        assert pm.first_drift(ref, seen, [(0, 4), (0, 4), (4, 8)], tags)["owner_ranks"] == [0, 1]

    def test_order_decides_which_node_is_first(self):
        ref, spans, tags = self._case()
        seen = {k: v.clone() for k, v in ref.items()}
        seen["block.1"][0, 0, 0] = 1.0
        seen["block.3"][0, 7, 0] = 1.0
        order = ["block.3", "block.1"]  # backward order
        assert pm.first_drift(ref, seen, spans, tags, order)["node"] == "block.3"


@needs_gpu
class TestDigest:
    def test_equal_tensors_agree_and_one_flipped_bit_disagrees(self):
        gen = torch.Generator().manual_seed(1)
        x = torch.randn(300, 7, generator=gen).bfloat16().cuda()
        assert pm.exact_digest(x) == pm.exact_digest(x.clone())
        y = x.clone().view(torch.int16)
        y[123, 3] ^= 1
        assert pm.exact_digest(x) != pm.exact_digest(y.view(torch.bfloat16))

    def test_a_swap_of_two_values_is_seen(self):
        x = torch.arange(64, dtype=torch.float32).bfloat16().cuda()
        y = x.clone()
        y[[3, 40]] = y[[40, 3]]
        assert pm.exact_digest(x) != pm.exact_digest(y)


def _slice_gather_target(collective, dummy, *, tp, cp, sp, length):
    import torch.distributed as dist

    del collective, dummy
    rank = dist.get_rank()
    spans = pm.rank_spans(tp, cp, length, sp)
    gen = torch.Generator().manual_seed(4)
    full = torch.randn(1, length, 6, generator=gen).bfloat16().cuda().requires_grad_(True)
    rows = pm._RowsSlice.apply(full, spans, rank)
    back = pm._RowsGather.apply(rows, spans, rank, length)
    weight = torch.randn(1, length, 6, generator=torch.Generator().manual_seed(5)).bfloat16().cuda()
    (back.float() * weight.float()).sum().backward()
    lo, hi = spans[rank]
    return {
        "forward": torch.tensor(int(torch.equal(back, full.detach()))),
        "slice": torch.tensor(int(torch.equal(rows, full.detach()[:, lo:hi]))),
        # every rank's slice receives its rows' share, gathered back: the identity
        "grad": torch.tensor(int(torch.equal(full.grad, weight))),
    }


@needs_gpu
@pytest.mark.parametrize(
    ("tp", "cp", "sp", "length"),
    [
        pytest.param(2, 1, False, 12, marks=gpus(2)),
        pytest.param(2, 1, True, 12, marks=gpus(2)),
        pytest.param(1, 3, False, 13, marks=gpus(3)),
        pytest.param(2, 2, True, 40, marks=gpus(4)),
        pytest.param(2, 2, False, 261, marks=gpus(4)),
    ],
)
def test_slice_then_gather_is_the_identity_with_an_identity_adjoint(tp, cp, sp, length):
    ranks = _world(
        tp * cp, _slice_gather_target, {"dummy": torch.zeros(1)}, tp=tp, cp=cp, sp=sp, length=length
    )
    for r in ranks:
        assert all(int(v) == 1 for k, v in r.items() if k != "rank"), r


def _chain_target(collective, dummy, *, tp, cp, sp, cp_mode, depth):
    """``depth`` copies of block 0 chained through slice / gather, canonical grid vs this grid."""

    import torch.distributed as dist

    from rl_engine.validation.models import h3_block as hb
    from rl_engine.validation.models import h3_tp
    from rl_engine.validation.models.h3_weights import load_h3_block_weights

    del collective, dummy
    rank = dist.get_rank()
    weights = load_h3_block_weights("cuda")
    ctx = hb.make_block_context(weights, layout="tiny", seed=0)
    length = int(ctx["seq_len"])
    ops = hb.CandidateOps()
    grad = torch.randn(1, length, hb.HIDDEN, generator=torch.Generator().manual_seed(8))
    grad = grad.bfloat16().cuda()

    def run(grid, with_sp, world_rank):
        spans = pm.rank_spans(grid.tp, grid.cp, length, with_sp)
        hidden = ctx["hidden"].clone().requires_grad_(True)
        x, leaves_all = hidden, []
        shard = grid.seq_shard(length)
        lo, hi = spans[world_rank]
        for _ in range(depth):
            h_res = pm._RowsSlice.apply(x, spans, world_rank)
            cos, sin = par.coshard(shard, ctx["rope"][0], ctx["rope"][1], dim=0)
            idx = par.coshard(shard, ctx["index"], dim=0)[0]
            leaves: dict[str, torch.Tensor] = {}
            out = h3_tp.run_tp_block_rows(
                weights,
                h_res,
                idx[lo - shard.lo : hi - shard.lo],
                cos,
                sin,
                ctx["temb"],
                ops,
                grid,
                sp=with_sp,
                shard=shard,
                leaves=leaves,
            )
            x = pm._RowsGather.apply(out, spans, world_rank, length)
            leaves_all.append(leaves)
        x.backward(grad)
        return x.detach(), hidden.grad, [{n: lf.grad for n, lf in lv.items()} for lv in leaves_all]

    grid = h3_tp.make_grid(tp, cp, attention="h3_full", cp_mode=cp_mode)
    ref_out, ref_dx, ref_leaves = run(h3_tp.Grid(1, 1, 0, 0, attention="h3_full"), False, 0)
    out, dx, leaves = run(grid, sp, rank)
    flags = {
        "forward": int(torch.equal(out, ref_out)),
        "input_grad": int(torch.equal(dx, ref_dx)),
    }
    # weight shards of the TP-sharded projections: slices of the unsharded gradient, also with CP
    ok = True
    for d in range(depth):
        for name in pm.SHARDED_WEIGHTS:
            want = pm.shard_of(ref_leaves[d][name], name, tp, grid.tp_rank)
            ok &= bool(torch.equal(leaves[d][name], want))
    flags["weight_shards"] = int(ok)
    return {k: torch.tensor(v) for k, v in flags.items()}


@needs_gpu
@pytest.mark.parametrize(
    ("tp", "cp", "sp", "mode"),
    [
        pytest.param(2, 1, False, "ring", marks=gpus(2)),
        pytest.param(2, 1, True, "ring", marks=gpus(2)),
        pytest.param(1, 3, False, "ring", marks=gpus(3)),
        pytest.param(2, 2, False, "ring", marks=gpus(4)),
        pytest.param(2, 2, True, "ring", marks=gpus(4)),
        pytest.param(2, 2, True, "ulysses", marks=gpus(4)),
        pytest.param(4, 2, True, "ring", marks=gpus(8)),
    ],
)
def test_chained_blocks_equal_the_canonical_chain_bitwise(tp, cp, sp, mode):
    pytest.importorskip("triton")
    from rl_engine.backends.cuda.attention import h3_full_attention as fa

    try:
        fa._extension()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    ranks = _world(
        tp * cp,
        _chain_target,
        {"dummy": torch.zeros(1)},
        tp=tp,
        cp=cp,
        sp=sp,
        cp_mode=mode,
        depth=3,
    )
    for r in ranks:
        bad = [k for k, v in r.items() if k != "rank" and int(v) != 1]
        assert not bad, f"rank {int(r['rank'])}: {bad}"
