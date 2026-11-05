# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""One real MiniMax-H3 block, forward and backward (RFC #420 ``ws1_one_h3_block``).

GPU tests run block 0 of the pinned checkpoint on packed FL2VA layouts and
check the block-level promises of ``rl_engine.validation.models.h3_block``:

* every node is bitwise repeatable, and batch row 0 of a two-row batch has the
  bytes of the single-row run (forward, and the gradient reaching every node);
* nodes that promise it are bitwise equal to diffusers on diffusers' inputs;
* every node, the block output and every gradient are about as close to the
  FP32 golden as diffusers itself.

CPU tests pin the packed layouts to the pinned diffusers pipeline and check
the graph's structure.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from rl_engine.reference.minimax_h3 import block as gold
from rl_engine.validation.models import h3_block
from rl_engine.validation.models import h3_provider as prov
from rl_engine.validation.models.h3_cases import H3_BLOCK_LAYOUTS, h3_fl2va_layout

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="needs an SM80+ CUDA device",
)

# sha256(position_ids float64 bytes + token_tags int64 bytes) of diffusers@f53d552's
# ``MiniMaxH3PrepareLayoutStep.build_packed_sequence`` for each layout (text tags 1, patch
# (1, 2, 2), two audio channels, audio tag 2, video tag 0, no keyframe anchors).
PINNED_LAYOUTS = {
    "tiny": "5329a0a13a9996c4fea7a0e676b65aa24302d65bea2e2248ba68d12184cc7c37",
    "small": "64e4e79d8fb91d5a68ae7d15bb29049e08add38da2639030e089c6ef80968931",
    "medium": "16e448800695eedfb85e0aacbca9bd5415ed53fbbe25f701bcb60df5f4754a31",
}

# Accuracy: the candidate's error against the FP32 golden may exceed diffusers' by this
# factor plus an absolute floor (relative to the golden's max magnitude).
FORWARD_FACTOR, FORWARD_FLOOR = 1.25, 1e-3
BACKWARD_FACTOR, BACKWARD_FLOOR = 1.5, 5e-3
# A BF16 leaf gradient may instead exceed diffusers' error by this many ulps of its largest
# golden element. Its max|err| / max|golden| is set by one rounded element, so it moves in
# whole ulps (0.004 relative for attn.norm_q.weight), and the relative rule alone failed on
# seed 2 with torch 2.13 and on seed 0 with torch 2.9.1. Sized from
# tools/validation/models/h3_block_ulp_sweep.py over tiny/small x seeds 0-9 on both torch
# versions (max excess +4.15 ulp) and 34 single-node candidate/provider swaps (max +5.15 ulp);
# reports in reports/experiments/h3-one-block-b200/ulp-sweep-torch*.json.
BACKWARD_BF16_EXCESS_ULPS = 6.0


@pytest.mark.parametrize("layout", sorted(PINNED_LAYOUTS))
def test_layout_matches_pinned_pipeline(layout):
    packed = h3_fl2va_layout(**H3_BLOCK_LAYOUTS[layout], device="cpu")
    data = packed["position_ids"].numpy().tobytes() + packed["token_tags"].numpy().tobytes()
    assert hashlib.sha256(data).hexdigest() == PINNED_LAYOUTS[layout]


def test_graph_is_well_formed():
    seen = {"hidden", "temb"}
    for node in h3_block.NODES:
        assert set(node.inputs) <= seen, node.name
        assert node.name not in seen, node.name
        assert node.binding.status in ("row", "interim", "reference"), node.name
        seen.add(node.name)
    assert h3_block.OUTPUT == "residual_mlp"
    rows = {n.binding.rfc_row for n in h3_block.NODES if n.binding.status == "row"}
    assert rows == {"adaln_projection_3mod", "h3_rmsnorm", "adaln_gate_residual"}


def test_partial_rope_rotates_96_of_128_channels():
    packed = h3_fl2va_layout(**H3_BLOCK_LAYOUTS["tiny"], device="cpu")
    cos, sin = prov.provider_rope(packed["position_ids"])
    assert cos.shape == (packed["position_ids"].shape[0], 96) and cos.dtype == torch.float32
    gold_cos, gold_sin = gold.golden_rope_tables(packed["position_ids"])
    assert torch.equal(cos, gold_cos) and torch.equal(sin, gold_sin)
    x = torch.randn(1, cos.shape[0], 2, 128).bfloat16()
    for out in (prov.provider_apply_rotary(x, cos, sin), gold.golden_apply_rotary(x, cos, sin)):
        assert torch.equal(out[..., 96:].float(), x[..., 96:].float())
        assert not torch.equal(out[..., :96].float(), x[..., :96].float())


@pytest.fixture(scope="module")
def ops():
    return h3_block.CandidateOps()


@needs_gpu
def test_stack_rows_dispatch_to_cuda(require_h3_kernels, ops):
    backends = ops.backends()
    for name in ("projection", "norm", "gate"):
        assert backends[name].endswith("CudaOp"), backends


@needs_gpu
@pytest.mark.parametrize("layout", ["tiny", "small"])
def test_block_forward(h3_block_weights_cuda, ops, layout):
    report = h3_block.run_block_case(h3_block_weights_cuda, layout=layout, ops=ops)
    nodes = {node.name: node for node in h3_block.NODES}
    for entry in report["nodes"]:
        name = entry["node"]
        assert entry["repeat_bitwise_equal"], name
        assert entry["batch_row_bitwise_equal"], name
        if nodes[name].provider_bitwise_isolated:
            assert entry["isolated_vs_provider"]["bitwise_equal"], name
        bound = FORWARD_FACTOR * entry["provider_rel_err_vs_golden"] + FORWARD_FLOOR
        assert entry["candidate_rel_err_vs_golden"] <= bound, entry
    assert report["first_repeat_drift"] is None
    assert report["first_batch_drift"] is None
    drift = report["first_isolated_drift"]
    assert drift is None or not nodes[drift].provider_bitwise_isolated
    out = report["output"]
    assert out["candidate_rel_err_vs_golden"] <= (
        FORWARD_FACTOR * out["provider_rel_err_vs_golden"] + FORWARD_FLOOR
    )


@needs_gpu
def test_block_backward(h3_block_weights_cuda, ops):
    report = h3_block.run_block_backward(h3_block_weights_cuda, layout="tiny", ops=ops)
    for name, entry in report["leaves"].items():
        assert entry["repeat_bitwise_equal"], name
        bound = BACKWARD_FACTOR * entry["provider_rel_err_vs_golden"] + BACKWARD_FLOOR
        within_rel = entry["candidate_rel_err_vs_golden"] <= bound
        within_ulps = entry["storage_dtype"] == "bfloat16" and (
            entry["candidate_ulps_vs_golden"]
            <= entry["provider_ulps_vs_golden"] + BACKWARD_BF16_EXCESS_ULPS
        )
        assert within_rel or within_ulps, (name, entry)
    assert report["leaves"]["hidden"]["batch_row_bitwise_equal"]
    assert report["first_grad_repeat_drift"] is None
    assert report["first_grad_batch_drift"] is None
