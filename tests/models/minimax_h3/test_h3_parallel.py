# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 WS2 rows: TP attention/FFN, Ulysses CP, metadata co-shard, arbitrary length, packed IO.

Single-process tests pin the canonical 8-unit reduction (summing per-rank partials in tree
order reproduces the TP=1 result for TP 2/4/8) and the sharding helpers. Multi-process tests
(2-4 GPUs) run the whole block 0 under TP / CP / SP grids and require every rank's rows to be
bitwise equal to the single-GPU canonical block, forward and backward.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
from rl_engine.distributed.algorithms.hierarchical import tree_add
from rl_engine.validation.models import h3_block as hb
from rl_engine.validation.models import h3_tp
from rl_engine.validation.models.h3_ws2 import run_world

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def gpus(n):
    return pytest.mark.skipif(torch.cuda.device_count() < n, reason=f"needs {n} GPUs")


def _triton():
    pytest.importorskip("triton")


class TestSharding:
    @pytest.mark.parametrize("tp", [1, 2, 4, 8])
    def test_row_and_column_shards_tile_once(self, tp):
        w = torch.arange(56 * 128 * 16, dtype=torch.float32).view(56 * 128, 16)
        assert torch.equal(torch.cat([par.shard_rows(w, tp, r) for r in range(tp)]), w)
        w2 = torch.arange(8 * 64 * 4, dtype=torch.float32).view(4, 8 * 64)
        assert torch.equal(torch.cat([par.shard_cols(w2, tp, r) for r in range(tp)], 1), w2)

    @pytest.mark.parametrize("tp", [3, 5, 6, 16])
    def test_illegal_tp_is_rejected(self, tp):
        with pytest.raises(ValueError):
            par.shard_rows(torch.zeros(56 * 128, 4), tp, 0)

    def test_shards_are_whole_heads(self):
        for tp in (1, 2, 4, 8):
            assert (par.HEADS // tp) * par.HEAD_DIM == par.INNER // tp

    def test_gate_up_shards_keep_both_halves(self):
        w = torch.arange(2 * 64 * 3, dtype=torch.float32).view(128, 3)
        s = par.shard_gate_up(w, 4, 1)
        hidden, gate = s.chunk(2)
        assert torch.equal(hidden, w[:64][16:32]) and torch.equal(gate, w[64:][16:32])

    @pytest.mark.parametrize("length", [1, 2, 3, 7, 261, 264, 925])
    @pytest.mark.parametrize("cp", [1, 2, 3, 4])
    def test_seq_shards_tile_any_length(self, length, cp):
        spans = [(par.SeqShard(length, cp, r).lo, par.SeqShard(length, cp, r).hi) for r in range(cp)]
        assert spans[0][0] == 0 and spans[-1][1] == length
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
        sizes = [hi - lo for lo, hi in spans]
        assert max(sizes) - min(sizes) <= 1

    def test_coshard_slices_every_per_row_tensor_alike(self):
        n = 11
        idx, tags, rows = torch.arange(n), torch.arange(n) * 3, torch.arange(n * 2).view(n, 2)
        for r in range(3):
            s = par.SeqShard(n, 3, r)
            a, b, c = par.coshard(s, idx, tags, rows)
            assert torch.equal(a * 3, b) and torch.equal(c[:, 0] // 2, a)

    def test_coshard_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="rows"):
            par.coshard(par.SeqShard(5, 2, 0), torch.zeros(4))


@needs_gpu
class TestCanonicalReduction:
    """Per-rank partials added in tree order are bitwise the TP=1 result (no collective needed)."""

    @pytest.mark.parametrize("k", [7168, 14336])
    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_row_parallel_partials_fold_to_tp1(self, k, tp):
        _triton()
        gen = torch.Generator().manual_seed(1)
        x = torch.randn(33, k, generator=gen).bfloat16().cuda()
        w = (torch.randn(5376, k, generator=gen) / k**0.5).bfloat16().cuda()
        full = par._units_partial(x, w, 1)
        parts = [
            par._units_partial(par.shard_heads(x, tp, r, 1), par.shard_cols(w, tp, r), tp)
            for r in range(tp)
        ]
        assert torch.equal(tree_add(parts), full)
        assert torch.equal(
            par.row_parallel_linear(x, w), full.bfloat16()
        )

    def test_row_parallel_close_to_fp64(self):
        _triton()
        gen = torch.Generator().manual_seed(2)
        x = torch.randn(64, 7168, generator=gen).bfloat16().cuda()
        w = (torch.randn(5376, 7168, generator=gen) / 7168**0.5).bfloat16().cuda()
        got = par.row_parallel_linear(x, w).double()
        ref = x.double() @ w.double().t()
        provider = torch.nn.functional.linear(x, w).double()
        err = (got - ref).norm() / ref.norm()
        assert err <= 1.25 * (provider - ref).norm() / ref.norm() + 1e-3

    def test_each_row_is_independent_of_the_others(self):
        _triton()
        gen = torch.Generator().manual_seed(3)
        x = torch.randn(70, 7168, generator=gen).bfloat16().cuda()
        w = (torch.randn(5376, 7168, generator=gen) / 7168**0.5).bfloat16().cuda()
        whole = par.row_parallel_linear(x, w)
        for rows in (slice(0, 1), slice(5, 6), slice(30, 31), slice(0, 33)):
            assert torch.equal(par.row_parallel_linear(x[rows].contiguous(), w), whole[rows])

    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_column_parallel_backward_folds_to_tp1(self, tp):
        _triton()
        gen = torch.Generator().manual_seed(4)
        k, n = 5376, 7168
        x = torch.randn(40, k, generator=gen).bfloat16().cuda()
        ws = [(torch.randn(n, k, generator=gen) / k**0.5).bfloat16().cuda() for _ in range(3)]
        grads = [torch.randn(40, n, generator=gen).bfloat16().cuda() for _ in range(3)]

        def unit_partials(weights, gs, local):
            units = []
            for j in range(local):
                acc = None
                for grad, w in zip(gs, weights):
                    width = grad.shape[1] // local
                    piece = par.g.gemm(
                        grad[:, j * width : (j + 1) * width], w[j * width : (j + 1) * width],
                        out_dtype=torch.float32,
                    )
                    acc = piece if acc is None else acc + piece
                units.append(acc)
            return tree_add(units)

        full = unit_partials(ws, grads, 8)
        ranks = [
            unit_partials(
                [par.shard_rows(w, tp, r) for w in ws],
                [par.shard_heads(g_, tp, r, 1) for g_ in grads],
                8 // tp,
            )
            for r in range(tp)
        ]
        assert torch.equal(tree_add(ranks), full)

    def test_column_parallel_forward_is_the_unsharded_columns(self):
        _triton()
        gen = torch.Generator().manual_seed(5)
        x = torch.randn(40, 5376, generator=gen).bfloat16().cuda()
        w = (torch.randn(7168, 5376, generator=gen) / 5376**0.5).bfloat16().cuda()
        (full,) = par.column_parallel_linear(x, [w])
        for tp in (2, 4, 8):
            for r in range(tp):
                (part,) = par.column_parallel_linear(x, [par.shard_rows(w, tp, r)])
                assert torch.equal(part, par.shard_heads(full, tp, r, 1))


@needs_gpu
class TestPackedIO:
    @pytest.mark.parametrize("length", [1, 5, 261])
    def test_scatter_then_gather_is_identity_single_rank(self, length):
        t = torch.randn(length, 7).cuda()
        shard = par.SeqShard(length, 1, 0)
        assert torch.equal(par.gather_packed(par.scatter_packed(t, shard), shard), t)


# --- Multi-process grids --------------------------------------------------------------------


def _reference(seq_len=0, backward=False):
    from rl_engine.validation.models.h3_weights import load_h3_block_weights

    return h3_tp.block_case(
        load_h3_block_weights("cuda"), h3_tp.Grid(1, 1, 0, 0), seq_len=seq_len, backward=backward
    )


def _assemble(ranks, tp, cp, key="out"):
    """Global rows from the ranks; replicated ranks must agree bitwise."""

    pieces: dict[tuple[int, int], torch.Tensor] = {}
    for r in ranks:
        span = tuple(int(v) for v in r["rows"])
        if span in pieces:
            assert torch.equal(pieces[span], r[key]), f"ranks disagree on rows {span}"
        pieces[span] = r[key]
    return torch.cat([pieces[s] for s in sorted(pieces)], dim=1), sorted(pieces)


GRIDS = [
    pytest.param(2, 1, False, 0, id="tp2", marks=gpus(2)),
    pytest.param(4, 1, False, 0, id="tp4", marks=gpus(4)),
    pytest.param(1, 2, False, 0, id="cp2", marks=gpus(2)),
    pytest.param(2, 2, False, 0, id="tp2_cp2", marks=gpus(4)),
    pytest.param(2, 1, True, 0, id="tp2_sp", marks=gpus(2)),
    pytest.param(4, 1, True, 0, id="tp4_sp", marks=gpus(4)),
    pytest.param(2, 2, True, 0, id="tp2_cp2_sp", marks=gpus(4)),
    pytest.param(1, 2, False, 261, id="cp2_len261", marks=gpus(2)),
    pytest.param(2, 2, False, 261, id="tp2_cp2_len261", marks=gpus(4)),
    pytest.param(4, 1, False, 263, id="tp4_len263", marks=gpus(4)),
]


@needs_gpu
@pytest.mark.parametrize(("tp", "cp", "sp", "length"), GRIDS)
def test_parallel_block_equals_single_gpu_bitwise(tp, cp, sp, length):
    _triton()
    ranks = run_world(
        tp * cp, h3_tp.block_target, {"dummy": torch.zeros(1)},
        tp=tp, cp=cp, sp=sp, seq_len=length, backward=True,
    )
    ref = _reference(length, backward=True)
    out, spans = _assemble(ranks, tp, cp)
    assert out.shape == ref["out"].shape, (spans, out.shape)
    assert torch.equal(out.cpu(), ref["out"].cpu()), "forward differs from the single-GPU block"
    dx, _ = _assemble(ranks, tp, cp, "dx")
    assert torch.equal(dx.cpu(), ref["dx"].cpu()), "input gradient differs"
    if cp == 1:  # weight gradients: shards of the unsharded gradient
        for name, axis in (("attn.to_out.0.weight", 1), ("ff.net.2.weight", 1),
                           ("attn.to_q.weight", 0), ("attn.to_k.weight", 0), ("attn.to_v.weight", 0),
                           ("ff.hidden", 0), ("ff.gate", 0)):
            if sp:
                continue  # SP shards sequence rows; dW is reduced over rows below
            full = ref["dw." + name].cpu()
            shards = [r["dw." + name].cpu() for r in sorted(ranks, key=lambda r: r["rank"])]
            assert torch.equal(torch.cat(shards, dim=axis), full), f"dW {name} differs"


@needs_gpu
@gpus(2)
def test_ulysses_matches_unsharded_attention_for_ragged_lengths():
    _triton()
    for length in (1, 2, 7, 130):
        ranks = run_world(2, _ulysses_target, {"dummy": torch.zeros(1)}, length=length)
        gathered = torch.cat([r["out"] for r in ranks], dim=1)
        assert torch.equal(gathered.cpu(), ranks[0]["ref"].cpu()), f"length {length}"


def _ulysses_target(collective, dummy, *, length):
    import torch.distributed as dist

    from rl_engine.backends.cuda.attention.deterministic_attn import DeterministicAttentionOp

    del collective, dummy
    rank = dist.get_rank()
    gen = torch.Generator().manual_seed(7)
    q, k, v = (torch.randn(1, length, 8, 128, generator=gen).bfloat16().cuda() for _ in range(3))
    op = DeterministicAttentionOp()

    def attend(q_, k_, v_):
        t = lambda x: x.transpose(1, 2).contiguous()  # noqa: E731
        return op(t(q_), t(k_), t(v_), causal=False).transpose(1, 2).contiguous()

    shard = par.SeqShard(length, 2, rank)
    local = [par.scatter_packed(t, shard, dim=1) for t in (q, k, v)]
    out = par.ulysses_attention(*local, attend, shard, dist.group.WORLD)
    return {"out": out, "ref": attend(q, k, v)}


# --- Hierarchical (cross-node design) collective, exercised inside one node ----------------


def _hier_target(collective, dummy, *, local_size, dtype):
    import torch.distributed as dist

    from rl_engine.distributed.algorithms.hierarchical import HierarchicalCollective

    del dummy
    h = HierarchicalCollective(local_size, device=torch.cuda.current_device(), max_size_bytes=64 << 20)
    gen = torch.Generator().manual_seed(100 + dist.get_rank())
    x = (torch.randn(257, 129, generator=gen) * 10 ** torch.randint(-3, 4, (257, 1), generator=gen)).to(
        dtype
    ).cuda()
    flat = collective.all_reduce(x.clone())
    mine = h.all_reduce(x)
    rs = h.reduce_scatter(x[: 256].contiguous())
    ag = h.all_gather(x[:2].contiguous())
    return {"flat": flat, "hier": mine, "rs": rs, "ag": ag, "x": x}


@needs_gpu
@gpus(4)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("local_size", [1, 2])
def test_hierarchical_equals_single_node_fixed_tree(local_size, dtype):
    ranks = run_world(4, _hier_target, {"dummy": torch.zeros(1)}, local_size=local_size, dtype=dtype)
    for r in ranks:
        assert torch.equal(r["flat"], r["hier"]), f"rank {r['rank']}: hierarchical != fixed tree"
    assert torch.equal(ranks[0]["rs"], ranks[0]["hier"][:64])
    assert torch.equal(ranks[3]["rs"], ranks[3]["hier"][192:256])
    assert torch.equal(ranks[1]["ag"], torch.cat([r["x"][:2] for r in ranks]))
