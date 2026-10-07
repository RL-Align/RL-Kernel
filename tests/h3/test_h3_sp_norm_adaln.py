# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``sp_norm_adaln``: sequence-parallel norm / modulation / gated residual.

* ownership: contiguous position shards per batch item; bad layouts fail closed;
* backward (one GPU, ranks as threads around the plain backward functions): every
  rank's row gradients equal WS1's rows, and its ``dweight``/``d_shift``/``d_scale``/
  ``d_gate`` equal WS1's, for SP 2..8, odd S, B > 1, interleaved modality rows and
  bf16/fp32;
* end to end (real NCCL, autograd): the MLP-side block region
  ``norm2(residual + gate_msa * y)`` with modulation is byte-equal to WS1 on every rank.
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import pytest
import torch

from rl_engine.kernels.ops.cuda.h3.sp_norm_adaln import SPRowLayout, sp_row_layout
from rl_engine.testing.h3_ws2 import (
    run_world,
    sp_case,
    sp_matches_ws1,
    sp_region_rank,
    ws1_sp_region,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
ALL_TRUE = dict.fromkeys(("out", "d_residual", "d_y", "d_weight", "d_table"), True)


def _require_ext():
    from rl_engine.kernels.ops.cuda.h3.sp_norm_adaln import sp_norm_adaln_available

    if not sp_norm_adaln_available():
        pytest.skip("rl_engine._C lacks the H3 SP symbols")


class _ThreadGroup:
    """Ranks as threads on one GPU; ``all_gather`` is the rank-ordered concatenation.

    Only for the plain backward functions: autograd runs every backward of a device
    on one worker thread, so autograd-driven ranks need real processes.
    """

    def __init__(self, world: int) -> None:
        self.barrier, self.slots = threading.Barrier(world), [None] * world

    def handle(self, rank: int):
        def all_gather(t):
            self.slots[rank] = t
            self.barrier.wait()
            out = torch.cat(self.slots)
            self.barrier.wait()
            return out

        return SimpleNamespace(rank=rank, world_size=len(self.slots), all_gather=all_gather)

    def run(self, fn):
        results, errors = [None] * len(self.slots), []

        def body(rank):
            try:
                results[rank] = fn(self.handle(rank))
            except BaseException as exc:  # re-raised below
                errors.append(exc)
                self.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(len(self.slots))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise next(
                (e for e in errors if not isinstance(e, threading.BrokenBarrierError)), errors[0]
            )
        return results


class TestLayout:
    @pytest.mark.parametrize("seq, sp", [(8, 8), (257, 2), (1000, 3), (4097, 8)])
    def test_shards_cover_every_position_once(self, seq, sp):
        bounds = [SPRowLayout(seq, 1, sp, r).bounds(r) for r in range(sp)]
        assert bounds[0][0] == 0 and bounds[-1][1] == seq
        assert all(a[1] == b[0] and a[0] < a[1] for a, b in zip(bounds, bounds[1:]))

    @pytest.mark.parametrize("seq, batch, sp, rank", [(3, 1, 4, 0), (8, 0, 2, 0), (8, 1, 2, 2)])
    def test_rejects(self, seq, batch, sp, rank):
        with pytest.raises(ValueError):
            sp_row_layout(seq, batch, sp, rank)


@requires_cuda
class TestBackwardRanks:
    @pytest.mark.parametrize(
        "sp, batch, seq, hidden, layout, dtype",
        [
            (2, 1, 4097, 5376, "block", torch.bfloat16),
            (2, 1, 4097, 5376, "interleaved", torch.bfloat16),
            (3, 2, 1000, 256, "interleaved", torch.bfloat16),
            (4, 2, 777, 512, "block", torch.float32),
            (5, 1, 1301, 128, "interleaved", torch.bfloat16),
            (8, 2, 2049, 256, "block", torch.bfloat16),
            (8, 1, 40, 64, "interleaved", torch.float32),  # shards far smaller than a tile
        ],
    )
    def test_ranks_reproduce_ws1(self, sp, batch, seq, hidden, layout, dtype):
        _require_ext()
        from rl_engine import _C
        from rl_engine.kernels.ops.cuda.h3.adaln_row_gather import _segment_tiles
        from rl_engine.kernels.ops.cuda.h3.sp_norm_adaln import (
            sp_gate_residual_backward,
            sp_plan,
            sp_rmsnorm_backward,
        )

        c = sp_case(batch, seq, hidden, layout=layout, dtype=dtype, seed=sp * 100 + seq)
        chunks = c["table"].chunk(6, dim=-1)
        shift, scale, gate = chunks[3], chunks[4], chunks[2]  # shift/scale_mlp, gate_msa
        x2, g2, y2 = (c[k].view(-1, hidden) for k in ("residual", "grad", "y"))
        idx, rows = c["index"], shift.shape[0]
        tiles = _segment_tiles(idx.repeat(batch), rows)
        _, rstd = _C.h3_rmsnorm_forward(x2, c["weight"], 1e-5, shift, scale, idx)
        ws1_norm = _C.h3_rmsnorm_backward(g2, x2, c["weight"], rstd, shift, scale, idx, *tiles)
        ws1_gate = _C.h3_gate_residual_backward(g2, y2, gate, idx, *tiles)

        def rank(coll):
            lay = sp_row_layout(seq, batch, sp, coll.rank)
            part = slice(lay.lo, lay.hi)
            local = {
                k: c[k][:, part].reshape(-1, hidden).contiguous() for k in ("residual", "grad", "y")
            }
            rs = rstd.view(batch, seq)[:, part].reshape(-1).contiguous()
            plan = sp_plan(lay, idx, rows)
            norm = sp_rmsnorm_backward(
                local["grad"], local["residual"], c["weight"], rs, shift, scale, plan, coll
            )
            gate_grads = sp_gate_residual_backward(local["grad"], local["y"], gate, plan, coll)
            return lay, norm, gate_grads

        for lay, (dx, dw, dsh, dsc), (dy, dgate) in _ThreadGroup(sp).run(rank):
            part = slice(lay.lo, lay.hi)
            assert torch.equal(
                dx.view(batch, -1, hidden), ws1_norm[0].view(batch, seq, hidden)[:, part]
            )
            assert torch.equal(dw, ws1_norm[1])
            assert torch.equal(dsh, ws1_norm[2].to(dtype)) and torch.equal(
                dsc, ws1_norm[3].to(dtype)
            )
            assert torch.equal(
                dy.view(batch, -1, hidden), ws1_gate[0].view(batch, seq, hidden)[:, part]
            )
            assert torch.equal(dgate, ws1_gate[1])

    def test_plain_norm_ranks_reproduce_ws1(self):
        _require_ext()
        from rl_engine import _C
        from rl_engine.kernels.ops.cuda.h3.sp_norm_adaln import sp_plan, sp_rmsnorm_backward

        c = sp_case(2, 1500, 256, seed=5)
        x2, g2 = c["residual"].view(-1, 256), c["grad"].view(-1, 256)
        _, rstd = _C.h3_rmsnorm_forward(x2, c["weight"], 1e-5)
        ws1 = _C.h3_rmsnorm_backward(g2, x2, c["weight"], rstd)

        def rank(coll):
            lay = sp_row_layout(1500, 2, 4, coll.rank)
            part = slice(lay.lo, lay.hi)
            loc = [c[k][:, part].reshape(-1, 256).contiguous() for k in ("grad", "residual")]
            rs = rstd.view(2, 1500)[:, part].reshape(-1).contiguous()
            return sp_rmsnorm_backward(*loc, c["weight"], rs, None, None, sp_plan(lay), coll)

        assert all(torch.equal(dw, ws1[1]) for _, dw, _, _ in _ThreadGroup(4).run(rank))

    def test_fails_closed(self):
        _require_ext()
        from rl_engine.kernels.ops.cuda.h3.sp_norm_adaln import H3SPNormAdaLNCudaOp

        c = sp_case(1, 100, 64)
        shift, scale = c["table"].chunk(6, dim=-1)[3:5]
        op = H3SPNormAdaLNCudaOp(SimpleNamespace(rank=1, world_size=2), seq_len=100, batch=1)
        local = c["residual"][:, 50:]
        with pytest.raises(ValueError):  # the full sequence instead of this rank's rows
            op.norm_modulated(c["residual"], c["weight"], shift, scale, c["index"])
        with pytest.raises(ValueError):  # a shard of the index instead of the full index
            op.norm_modulated(local, c["weight"], shift, scale, c["index"][50:])
        with pytest.raises(IndexError):
            op.gate_residual(local, c["y"][:, 50:], shift, c["index"] + 9)
        with pytest.raises(ValueError):
            op.norm(local.cpu(), c["weight"].cpu())
        assert op.readback()["positions"] == [50, 100]


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 1, reason="owns its processes")
def test_nccl_region_byte_equal_to_ws1(world):
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    _require_ext()
    case = sp_case(2, 4097, layout="interleaved", seed=world)
    ws1 = {k: v.cpu() for k, v in ws1_sp_region(**case).items()}
    torch.cuda.empty_cache()
    ranks = run_world(world, sp_region_rank, {k: v.cpu() for k, v in case.items()})
    assert sp_matches_ws1(ws1, ranks) == ALL_TRUE
    assert {r["readback"]["collective_backend"] for r in ranks} == {"cuda_ipc_fixed_tree"}
