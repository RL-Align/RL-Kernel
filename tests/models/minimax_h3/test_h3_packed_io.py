# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``distributed_packed_io``: scatter from one rank, gather video/audio, row identity.

The packed sequence interleaves text, video and audio rows by ``token_tags``; the model reads
the video and audio outputs back by index. These cases run on real NCCL groups (one GPU per
rank), use a non-zero root, ragged lengths (including fewer rows than ranks) and shuffled
selections, and compare with the single-process ``index_select`` bit for bit.
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


def _layout(length, seed=3):
    """Shuffled text(1) / video(0) / audio(2) tags, as legal packed permutations."""

    gen = torch.Generator().manual_seed(seed)
    tags = torch.zeros(length, dtype=torch.long)
    order = torch.randperm(length, generator=gen)
    tags[order[: length // 5]] = 1
    tags[order[length // 5 : length // 5 + length // 4]] = 2
    video = (tags == 0).nonzero().flatten()
    audio = (tags == 2).nonzero().flatten()
    shuffled = video[torch.randperm(video.numel(), generator=gen)]
    return tags, video, audio, shuffled


def _io_target(collective, dummy, *, length, root):
    """Per-rank checks of scatter, gather, selected-row gather and the row-identity check."""
    import torch.distributed as dist

    del collective, dummy
    cp, rank = dist.get_world_size(), dist.get_rank()
    shard = par.SeqShard(length, cp, rank)
    gen = torch.Generator().manual_seed(11)
    packed = torch.randn(1, length, 9, generator=gen).bfloat16().cuda()
    ids = torch.arange(length).cuda()
    tags, video, audio, shuffled = (t.cuda() for t in _layout(length))
    flags: dict[str, int] = {}

    # input scatter from the root only (the sequence axis is the leading one here)
    rows = par.scatter_from_root(
        packed[0] if rank == root else None,
        shard,
        tail_shape=(9,),
        dtype=torch.bfloat16,
        root=root,
    )
    flags["scatter"] = int(torch.equal(rows, packed[0, shard.lo : shard.hi]))
    local_ids = par.scatter_from_root(
        ids if rank == root else None, shard, tail_shape=(), dtype=torch.long, root=root
    )
    flags["scatter_ids"] = int(torch.equal(local_ids, ids[shard.lo : shard.hi]))
    par.check_row_identity(local_ids, shard)  # raises on drift
    flags["identity_ok"] = 1
    # negative control: rank 0 (always holds rows) declares ids that are not its rows. The check
    # gathers every rank's ids, so every rank, not only rank 0, must detect the drift.
    corrupt = local_ids.clone()
    if rank == 0:
        if corrupt.numel() >= 2:
            corrupt[[0, -1]] = corrupt[[-1, 0]]
        else:
            corrupt = corrupt + 1
    try:
        par.check_row_identity(corrupt, shard)
        flags["identity_drift_detected"] = 0
    except ValueError:
        flags["identity_drift_detected"] = 1
    # negative control: rank 0 declares no rows where it holds some. The gather pads a missing
    # row with zero, so only the count check, run on every rank, can see it (length 1: [0]).
    try:
        par.check_row_identity(local_ids[:0] if rank == 0 else local_ids, shard)
        flags["missing_rows_detected"] = 0
    except ValueError:
        flags["missing_rows_detected"] = 1

    # final gathers (video, audio, a shuffled order, nothing)
    local = packed[:, shard.lo : shard.hi]
    for name, sel in (("video", video), ("audio", audio), ("shuffled", shuffled)):
        got_rows = par.gather_selected(local, shard, sel, dim=1)
        flags[name] = int(torch.equal(got_rows, packed.index_select(1, sel)))
    empty = torch.empty(0, dtype=torch.long, device="cuda")
    flags["empty"] = int(par.gather_selected(local, shard, empty, dim=1).shape[1] == 0)

    # backward: each rank gets its rows of the replicated output gradient, zero elsewhere
    leaf = local.detach().clone().float().requires_grad_(True)
    out = par.gather_selected(leaf, shard, shuffled, dim=1)
    weight = torch.randn(out.shape, generator=torch.Generator().manual_seed(5)).cuda()
    (out * weight).sum().backward()
    want = torch.zeros(1, length, 9, device="cuda")
    want.index_copy_(1, shuffled, weight)
    flags["backward"] = int(torch.equal(leaf.grad, want[:, shard.lo : shard.hi]))
    return {"rank": torch.tensor(rank), **{k: torch.tensor(v) for k, v in flags.items()}}


def _check(ranks):
    """Every rank reports every check as passed."""
    for r in ranks:
        bad = [k for k, v in r.items() if k != "rank" and int(v) != 1]
        assert not bad, f"rank {int(r['rank'])}: {bad}"


@needs_gpu
class TestSingleRank:
    def test_gather_selected_matches_index_select(self):
        """On one rank, the selected-row gather equals ``index_select``."""
        local = torch.randn(1, 20, 4).cuda()
        shard = par.SeqShard(20, 1, 0)
        sel = torch.tensor([7, 2, 19, 0]).cuda()
        assert torch.equal(par.gather_selected(local, shard, sel, dim=1), local[:, sel])

    @pytest.mark.parametrize(
        ("selected", "match"),
        [
            (torch.tensor([1, 1]), "unique"),
            (torch.tensor([0, 20]), "lie in"),
            (torch.tensor([[0, 1]]), "1-D"),
            (torch.tensor([0.5]), "integer"),
        ],
    )
    def test_rejects_bad_selections(self, selected, match):
        """Out-of-range, repeated or non-integer selections are refused."""
        shard = par.SeqShard(20, 1, 0)
        with pytest.raises(ValueError, match=match):
            par.gather_selected(torch.zeros(1, 20, 4).cuda(), shard, selected.cuda(), dim=1)

    def test_wrong_local_row_count_is_rejected(self):
        """A local tensor with the wrong number of rows for its shard is refused."""
        with pytest.raises(ValueError, match="local rows"):
            par.gather_selected(
                torch.zeros(1, 3, 4).cuda(), par.SeqShard(20, 2, 0), torch.tensor([0]).cuda(), dim=1
            )

    def test_scatter_roles_are_checked(self):
        """Only the root may hold the input, and the root must be a group member."""
        shard = par.SeqShard(4, 1, 0)
        with pytest.raises(ValueError, match="root must hold"):
            par.scatter_from_root(None, shard, tail_shape=(2,), dtype=torch.float32)
        with pytest.raises(ValueError, match="not a member"):
            par.scatter_from_root(
                torch.zeros(4, 2), shard, tail_shape=(2,), dtype=torch.float32, root=3
            )


def test_a_group_of_another_size_than_the_shard_is_refused(monkeypatch):
    """A process group that does not have ``shard.cp`` ranks (e.g. the default world group of a
    TP x CP grid) is refused before any collective."""
    monkeypatch.setattr(par.dist, "get_world_size", lambda group=None: 8)
    shard = par.SeqShard(20, 2, 0)
    local = torch.zeros(1, shard.sizes[0], 4)
    with pytest.raises(ValueError, match="group has 8 ranks, but the rows are split over 2"):
        par.gather_selected(local, shard, torch.tensor([0, 1]), dim=1)
    with pytest.raises(ValueError, match="group has 8 ranks"):
        par.check_row_identity(torch.arange(shard.sizes[0]), shard)
    monkeypatch.setattr(par.torch.cuda, "current_device", lambda: 0)
    with pytest.raises(ValueError, match="group has 8 ranks"):
        par.scatter_from_root(
            None, par.SeqShard(20, 2, 1), tail_shape=(4,), dtype=torch.float32, root=0
        )


def _one_rank_of(monkeypatch, cp):
    """Run this process as one rank of a ``cp``-rank group whose ranks all send what it sends."""
    monkeypatch.setattr(par.dist, "get_world_size", lambda group=None: cp)

    def all_gather(out, tensor, group=None):
        for o in out:
            o.copy_(tensor)

    monkeypatch.setattr(par.dist, "all_gather", all_gather)


def test_a_missing_row_is_not_read_as_row_zero(monkeypatch):
    """One row over two ranks, and rank 0 declares none: the gather would pad it to ``[0]``, the
    one valid identity. The row counts are compared first, so the drift is reported."""
    _one_rank_of(monkeypatch, 2)
    with pytest.raises(
        ValueError, match=r"the ranks hold \[0, 0\] rows, the shard splits \[1, 0\]"
    ):
        par.check_row_identity(torch.empty(0, dtype=torch.long), par.SeqShard(1, 2, 0))


def test_a_negative_dim_gives_the_gradient_of_its_axis(monkeypatch):
    """``dim=-1`` selects along the last axis and its backward returns the local gradient, like
    the same call with the axis as a non-negative index."""
    _one_rank_of(monkeypatch, 2)
    shard = par.SeqShard(4, 2, 0)  # rank 0 holds rows 0 and 1: the selection is all its own
    selected = torch.tensor([1, 0])
    weight = torch.randn(3, 2, generator=torch.Generator().manual_seed(1))
    grads = []
    for dim in (-1, 1):
        leaf = torch.randn(3, 2, generator=torch.Generator().manual_seed(0)).requires_grad_(True)
        out = par.gather_selected(leaf, shard, selected, dim=dim)
        assert torch.equal(out, leaf.detach()[:, [1, 0]])
        (out * weight).sum().backward()
        grads.append(leaf.grad)
    assert torch.equal(grads[0], grads[1])
    assert torch.equal(grads[0], weight[:, [1, 0]])
    with pytest.raises(ValueError, match="out of range"):
        par.gather_selected(torch.zeros(3, 2), shard, selected, dim=2)


def test_the_root_tensor_must_be_on_the_current_device(monkeypatch):
    """The root sends from its packed tensor, the receivers receive on their current device: a
    root tensor on another device is refused."""
    monkeypatch.setattr(par.torch.cuda, "current_device", lambda: 0)
    with pytest.raises(ValueError, match="packed tensor is on cpu, not on cuda:0"):
        par.scatter_from_root(
            torch.zeros(20, 4), par.SeqShard(20, 2, 0), tail_shape=(4,), dtype=torch.float32
        )


@needs_gpu
@pytest.mark.parametrize("cp", [2, 3, 4])
@pytest.mark.parametrize("length", [1, 5, 261])
@pytest.mark.parametrize("root", [0, 1])
def test_packed_io_matches_single_process_bitwise(cp, length, root):
    """Distributed packed IO equals the single-process result bitwise on ``cp`` ranks."""
    if torch.cuda.device_count() < cp:
        pytest.skip(f"needs {cp} GPUs")
    _check(_world(cp, _io_target, {"dummy": torch.zeros(1)}, length=length, root=min(root, cp - 1)))
