# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the TP weight-slicing index math for TP-exact routed experts."""

import pytest
import torch

from rl_engine.integrations.engines.rollout.vllm.routed_experts_tp import (
    expert_key,
    expert_offsets,
    load_tp_expert_slices,
    slice_bounds,
)

E, I, H, LAYER = 4, 128, 256, 3


class FakeSource:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.t = {}
        for e in range(E):
            for n, rows, cols in (("w1", I, H), ("w3", I, H), ("w2", H, I)):
                self.t[expert_key(LAYER, e, n + ".weight")] = torch.randint(
                    0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g
                )
                self.t[expert_key(LAYER, e, n + ".scale")] = torch.randint(
                    100, 140, (rows, cols // 32), dtype=torch.uint8, generator=g
                )

    def shape(self, name):
        return tuple(self.t[name].shape)

    def rows(self, name, a, b):
        return self.t[name][a:b]


def _load(rank, size):
    return load_tp_expert_slices(FakeSource_, LAYER, rank, size, "cpu", num_experts=E)


FakeSource_ = FakeSource()


def test_slice_bounds_partition_and_errors():
    assert [slice_bounds(2048, 4, r) for r in range(4)] == [
        (0, 512), (512, 1024), (1024, 1536), (1536, 2048)
    ]
    with pytest.raises(ValueError):
        slice_bounds(2048, 3, 0)
    with pytest.raises(ValueError):
        slice_bounds(2048, 128, 0)  # 16 columns: not a 32-column tile multiple
    with pytest.raises(ValueError):
        slice_bounds(64, 2, 2)


@pytest.mark.parametrize("size", [2, 4])
def test_slices_reassemble_to_original_fused_weights(size):
    src = FakeSource_
    parts = [_load(r, size) for r in range(size)]
    w = I // size
    for e in range(E):
        full_gate = src.t[expert_key(LAYER, e, "w1.weight")]
        full_up = src.t[expert_key(LAYER, e, "w3.weight")]
        # gate|up order inside each rank's slice, rank-ordered concat of each half.
        gate = torch.cat([p.gate_up_codes[e, :w] for p in parts])
        up = torch.cat([p.gate_up_codes[e, w:] for p in parts])
        assert torch.equal(gate, full_gate) and torch.equal(up, full_up)
        sg = torch.cat([p.gate_up_scales[e, :w] for p in parts])
        assert torch.equal(sg, src.t[expert_key(LAYER, e, "w1.scale")])
        su = torch.cat([p.gate_up_scales[e, w:] for p in parts])
        assert torch.equal(su, src.t[expert_key(LAYER, e, "w3.scale")])
        down = torch.cat([p.down_codes[e] for p in parts])
        assert torch.equal(down, src.t[expert_key(LAYER, e, "w2.weight")])
        ds = torch.cat([p.down_scales[e] for p in parts])
        assert torch.equal(ds, src.t[expert_key(LAYER, e, "w2.scale")])
    p = parts[0]
    assert tuple(p.gate_up_codes.shape) == (E, 2 * w, H // 2)
    assert tuple(p.down_codes.shape) == (E, H // size, I // 2)
    assert (p.intermediate, p.hidden) == (I, H)


def test_activation_pairing_is_per_column_under_slicing():
    # silu(gate_j) * up_j on a [gate | up] slice equals the same columns of the full op.
    full = torch.randn(5, 2 * I)
    gate, up = full[:, :I], full[:, I:]
    ref = torch.nn.functional.silu(gate) * up
    size = 4
    w = I // size
    out = []
    for r in range(size):
        zs = torch.cat((gate[:, r * w : (r + 1) * w], up[:, r * w : (r + 1) * w]), dim=1)
        out.append(torch.nn.functional.silu(zs[:, :w]) * zs[:, w:])
    assert torch.equal(torch.cat(out, dim=1), ref)


def test_expert_offsets_tile_rows():
    experts = torch.tensor([0, 0, 2, 2, 2, 5], dtype=torch.int64)
    off = expert_offsets(experts, 8)
    assert off.dtype == torch.int64
    assert off.tolist() == [0, 2, 2, 5, 5, 5, 6, 6, 6]
