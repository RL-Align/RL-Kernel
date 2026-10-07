# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``tp_adaln_3mod``: column-parallel AdaLN projection, byte-equal to WS1.

* ownership: contiguous 64-aligned column shards cover the 3 x 6 x H table
  exactly once; unsupported TP sizes and wrong shards fail closed;
* shard arithmetic (one GPU): a rank's forward columns and ``d_input`` chunk
  partials are bitwise the matching slice of the full WS1 call, and folding
  the rank-ordered partials reproduces WS1's ``d_input``;
* real NCCL (2/4/8 GPUs): every rank's table and ``d_temb`` and the
  concatenated ``dW``/``db`` shards are bitwise equal to WS1.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.cuda.h3.tp_adaln_projection import adaln_column_shard
from rl_engine.testing.h3_ws2 import run_world, tp_matches_ws1, tp_projection_rank, ws1_projection

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
WEIGHT = "transformer_blocks.0.adaln_proj.linear.weight"
BIAS = "transformer_blocks.0.adaln_proj.linear.bias"
N_H3 = 3 * 6 * 5376
ALL_TRUE = dict.fromkeys(("table", "d_temb", "d_weight", "d_bias"), True)


def _require_ext():
    from rl_engine.kernels.ops.cuda.h3.det_linear import det_linear_available

    if not det_linear_available():
        pytest.skip("rl_engine._C lacks h3_det_linear_*")


def _inputs(num_t, n_out, k, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    temb = (torch.randn(num_t, k, generator=g) * 2).cuda()
    weight = (torch.randn(n_out, k, generator=g) / k**0.5).to(dtype).cuda()
    bias = (torch.randn(n_out, generator=g) * 0.1).to(dtype).cuda()
    grad = torch.randn(num_t, n_out, generator=g).to(dtype).cuda()
    return temb, weight, bias, grad


class TestOwnership:
    @pytest.mark.parametrize("tp", [1, 2, 3, 4, 6, 7, 8])
    def test_shards_tile_the_table_once(self, tp):
        shards = [adaln_column_shard(N_H3, tp, r) for r in range(tp)]
        assert [s.begin for s in shards] == [0, *[s.end for s in shards[:-1]]]
        assert shards[-1].end == N_H3
        assert sum(h1 - h0 for s in shards for (_, _, h0, h1) in s.slots()) == N_H3

    def test_tp2_slots(self):
        # rank 0: video's six chunks and text's first three; rank 1: the rest.
        assert adaln_column_shard(N_H3, 2, 0).slots() == [
            *[(0, c, 0, 5376) for c in range(6)],
            *[(1, c, 0, 5376) for c in range(3)],
        ]

    @pytest.mark.parametrize("tp, rank", [(5, 0), (16, 0), (0, 0), (2, 2), (2, -1)])
    def test_rejects_unsupported(self, tp, rank):
        with pytest.raises(ValueError):
            adaln_column_shard(N_H3, tp, rank)


@requires_cuda
class TestShardArithmetic:
    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_pinned_shards_reproduce_ws1(self, h3_weights_cpu, tp):
        _require_ext()
        from rl_engine.kernels.ops.cuda.h3 import det_linear as dl

        weight, bias = h3_weights_cpu[WEIGHT].cuda(), h3_weights_cpu[BIAS].cuda()
        g = torch.Generator(device="cpu").manual_seed(tp)
        act = F.silu(torch.randn(3, weight.shape[1], generator=g).cuda() * 2).bfloat16()
        grad = torch.randn(3, weight.shape[0], generator=g).cuda()
        (full,) = dl.det_linear_forward(act, weight, bias)
        full_partials = dl.det_linear_backward_input_partials(grad, weight)
        pieces = []
        for rank in range(tp):
            s = adaln_column_shard(weight.shape[0], tp, rank)
            (cols,) = dl.det_linear_forward(act, weight[s.begin : s.end], bias[s.begin : s.end])
            assert torch.equal(cols, full[:, s.begin : s.end])
            part = dl.det_linear_backward_input_partials(
                grad[:, s.begin : s.end].contiguous(), weight[s.begin : s.end]
            )
            c0 = s.begin // dl.DINPUT_CHUNK
            assert torch.equal(part, full_partials[c0 : c0 + part.shape[0]])
            pieces.append(part)
        folded = dl.det_linear_fold_chunks(torch.cat(pieces), torch.float32)
        assert torch.equal(folded, dl.det_linear_backward_input(grad, weight, torch.float32))

    def test_tp1_op_is_ws1(self):
        _require_ext()
        one = SimpleNamespace(rank=0, world_size=1, backend_id="none", all_gather=lambda t: t)
        temb, weight, bias, grad = _inputs(3, 3 * 6 * 256, 136)
        ws1 = ws1_projection(temb, weight, bias, grad)
        assert tp_matches_ws1(ws1, [tp_projection_rank(one, temb, weight, bias, grad)]) == ALL_TRUE

    def test_fails_closed(self):
        _require_ext()
        from rl_engine.kernels.ops.cuda.h3.tp_adaln_projection import (
            H3TPAdaLNProjectionCudaOp,
            shard_adaln_projection,
        )

        temb, weight, bias, _ = _inputs(2, 3 * 6 * 256, 136)
        rank1 = SimpleNamespace(rank=1, world_size=2, backend_id="unused", all_gather=None)
        op = H3TPAdaLNProjectionCudaOp(rank1, weight.shape[0])
        w, b = shard_adaln_projection(weight, bias, 2, 1)
        with pytest.raises(TypeError):  # RFC probe H7: cast before the SiLU
            op(temb.bfloat16(), w, b)
        with pytest.raises(ValueError):  # the full weight instead of this rank's rows
            op(temb, weight, bias)
        with pytest.raises(ValueError):
            op(temb.cpu(), w, b)
        with pytest.raises(ValueError):  # 4608 / 16 is not a multiple of 64
            H3TPAdaLNProjectionCudaOp(SimpleNamespace(rank=0, world_size=16), weight.shape[0])
        assert op.readback()["columns"] == [2304, 4608]


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 1, reason="owns its processes")
def test_nccl_byte_equal_to_ws1(h3_weights_cpu, world):
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    _require_ext()
    weight, bias = h3_weights_cpu[WEIGHT], h3_weights_cpu[BIAS]
    g = torch.Generator(device="cpu").manual_seed(world)
    inputs = {
        "temb": torch.randn(3, weight.shape[1], generator=g) * 2,
        "weight": weight,
        "bias": bias,
        "grad": torch.randn(3, weight.shape[0], generator=g).bfloat16(),
    }
    ws1 = {
        k: v.cpu() for k, v in ws1_projection(**{k: v.cuda() for k, v in inputs.items()}).items()
    }
    torch.cuda.empty_cache()
    ranks = run_world(world, tp_projection_rank, inputs)
    assert tp_matches_ws1(ws1, ranks) == ALL_TRUE
    assert [r["readback"]["rank"] for r in ranks] == list(range(world))
    assert {r["readback"]["collective_backend"] for r in ranks} == {"cuda_ipc_fixed_tree"}
