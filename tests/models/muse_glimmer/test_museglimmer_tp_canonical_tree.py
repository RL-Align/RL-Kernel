# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer canonical TP tree (RFC #501, WS2 ``tp_canonical_tree``).

These tests run on CPU. Cross-TP equality uses one thread per TP rank and the production
balanced-tree reduction (see ``tools/validation/models/museglimmer_tp_canonical_tree.py``);
the 8-GPU run of the same script is the hardware evidence.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import pytest
import torch

from rl_engine.models.muse_glimmer import tp_canonical_tree as tct
from rl_engine.models.muse_glimmer import workload as w

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools/validation/models/museglimmer_tp_canonical_tree.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("museglimmer_tp_canonical_tree_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _canonical_env(monkeypatch):
    monkeypatch.delenv(tct.CANONICAL_TP_ENV, raising=False)


class _FakeCollective:
    def __init__(self, backend_id: str, world_size: int = 2, rank: int = 0) -> None:
        self.backend_id, self.world_size, self.rank = backend_id, world_size, rank

    def all_reduce(self, value):  # pragma: no cover - must never be reached
        raise AssertionError("a rejected collective was used")


# --- frozen layout --------------------------------------------------------------------------


def test_official_shards_match_fingerprint():
    shards = w.OFFICIAL_PARALLEL_LAYOUT["row_parallel_shards"]
    assert tct.muse_projection("self_attn.o_proj").canonical_width == shards["o_proj"] == 512
    assert tct.muse_projection("mlp.down_proj").canonical_width == shards["down_proj"] == 2496
    assert tct.muse_projection("self_attn.q_proj").canonical_width == 512
    assert tct.muse_projection("mlp.up_proj").canonical_width == 2496
    # 512 = 4 heads of 128 per canonical shard.
    assert 512 // int(w.OFFICIAL_FINGERPRINT["head_dim"]) == 4


def test_owned_leaves_are_contiguous_subtrees():
    assert tct.owned_leaves(0, 1) == list(range(8))
    assert tct.owned_leaves(1, 2) == [4, 5, 6, 7]
    assert tct.owned_leaves(2, 4) == [4, 5]
    assert tct.owned_leaves(7, 8) == [7]


# --- fail closed ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["self_attn.qkv_proj", "mlp.gate_up_proj", "linear_qkv", "linear_fc1", "qkvg_proj"]
)
def test_fused_projection_fails_closed(name):
    with pytest.raises(tct.CanonicalTreeError, match="fused projection"):
        tct.muse_projection(name)


def test_unknown_projection_fails_closed():
    with pytest.raises(tct.CanonicalTreeError, match="not on the Muse Glimmer canonical TP tree"):
        tct.muse_projection("self_attn.k_proj")


@pytest.mark.parametrize("value", ["4", "16", "1"])
def test_canonical_tp_env_other_than_8_fails_closed(monkeypatch, value):
    monkeypatch.setenv(tct.CANONICAL_TP_ENV, value)
    with pytest.raises(tct.CanonicalTreeError, match="freezes canonical TP = 8"):
        tct.leaves_per_rank(2)


def test_canonical_tp_env_8_is_accepted(monkeypatch):
    monkeypatch.setenv(tct.CANONICAL_TP_ENV, "8")
    assert tct.leaves_per_rank(4) == 2


@pytest.mark.parametrize("tp", [3, 6, 16])
def test_unsupported_physical_tp_fails_closed(tp):
    with pytest.raises(tct.CanonicalTreeError, match="physical TP"):
        tct.leaves_per_rank(tp)


@pytest.mark.parametrize("backend", ["nccl", "rccl", None])
def test_native_reducing_collective_fails_closed(backend):
    spec = tct.muse_projection("self_attn.o_proj")
    x = torch.zeros(2, 2048, dtype=torch.bfloat16)
    weight = torch.zeros(6656, 2048, dtype=torch.bfloat16)
    with pytest.raises(tct.CanonicalTreeError, match="not a fixed-tree collective"):
        tct.row_parallel_forward(x, weight, spec, collective=_FakeCollective(backend))


def test_bf16_leaf_gemm_fails_closed():
    spec = tct.ProjectionSpec("p", tct.ROW_PARALLEL, 16, 4)

    def bf16_gemm(a, b):
        return a @ b.t()

    x = torch.ones(2, 16, dtype=torch.bfloat16)
    with pytest.raises(tct.CanonicalTreeError, match="partials must be FP32"):
        tct.row_parallel_forward(x, torch.ones(4, 16, dtype=torch.bfloat16), spec, gemm=bf16_gemm)


def test_shard_shape_mismatch_fails_closed():
    spec = tct.muse_projection("mlp.down_proj")
    x = torch.zeros(2, 19968 // 2, dtype=torch.bfloat16)  # a TP2 shard used as TP1
    weight = torch.zeros(6656, 19968 // 2, dtype=torch.bfloat16)
    with pytest.raises(tct.CanonicalTreeError, match="input shard width"):
        tct.row_parallel_forward(x, weight, spec)


def test_reduced_dimension_must_split_into_8_leaves():
    with pytest.raises(tct.CanonicalTreeError, match="canonical shards"):
        tct.ProjectionSpec("p", tct.ROW_PARALLEL, 12, 4).validate()


def test_balanced_tree_rejects_non_power_of_two():
    with pytest.raises(tct.CanonicalTreeError, match="2\\^k"):
        tct.balanced_tree_sum([torch.zeros(1)] * 3)


def test_balanced_tree_parenthesization():
    values = [torch.tensor([float(v)]) for v in (1e8, 1.0, -1e8, 1.0)]
    # (1e8 + 1) + (-1e8 + 1) in FP32 = 0, while a left fold gives 1.
    assert tct.balanced_tree_sum(values).item() == 0.0
    assert (((values[0] + values[1]) + values[2]) + values[3]).item() == 1.0


# --- numerics -------------------------------------------------------------------------------


def _tiny(tool):
    return tool.Inputs(tool.TINY, 12, torch.device("cpu"), seed=7)


def test_tp1_matches_fp64_and_repeats(tool):
    inp = _tiny(tool)
    reference = tool.tp1_reference(inp)
    failed = [c for c in tool.tp1_checks(inp, reference) if not c["pass"]]
    assert not failed, failed


def test_row_forward_rounds_once(tool):
    inp = _tiny(tool)
    spec = inp.specs["mlp.down_proj"]
    x, weight = inp.inputs["mlp.down_proj"], inp.weights["mlp.down_proj"]
    width = spec.canonical_width
    leaves = [
        x[:, i * width : (i + 1) * width].float()
        @ weight[:, i * width : (i + 1) * width].float().t()
        for i in range(8)
    ]
    expected = tct.balanced_tree_sum(leaves).to(torch.bfloat16)
    assert torch.equal(tct.row_parallel_forward(x, weight, spec), expected)


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_cross_tp_equals_tp1(tool, tp):
    inp = _tiny(tool)
    reference = tool.tp1_reference(inp)
    tct_allowed = tct.FIXED_TREE_COLLECTIVES
    tct.FIXED_TREE_COLLECTIVES = tct_allowed | {tool.SIMULATED_BACKEND_ID}
    try:
        shared = {"world": tp, "barrier": threading.Barrier(tp), "slots": {}}
        results = {}

        def worker(rank):
            try:
                collective = tool.ThreadedBalancedTreeCollective(shared, rank)
                results[rank] = tool.check_tp(inp, tp, rank, collective, reference)
            except BaseException as exc:
                shared["barrier"].abort()
                results[rank] = exc

        threads = [threading.Thread(target=worker, args=(r,)) for r in range(tp)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        tct.FIXED_TREE_COLLECTIVES = tct_allowed
    for rank in range(tp):
        assert not isinstance(results[rank], BaseException), results[rank]
        failed = [c for c in results[rank] if not c["pass"]]
        assert not failed, (rank, failed)
        # 2 row projections x 3 tensors + 4 column outputs + 2 group input grads + 4 weight grads
        assert len(results[rank]) == 16


def test_left_fold_collective_is_detected(tool, monkeypatch):
    """Negative control: a native-style left fold over ranks must not match TP1."""

    class LeftFold(tool.ThreadedBalancedTreeCollective):
        def all_reduce(self, input):
            barrier, slots = self.shared["barrier"], self.shared["slots"]
            slots[self.rank] = input.detach().clone()
            barrier.wait()
            total = slots[0].clone()
            for r in range(1, self.world_size):
                total = total + slots[r]
            barrier.wait()
            return total

    monkeypatch.setattr(
        tct, "FIXED_TREE_COLLECTIVES", tct.FIXED_TREE_COLLECTIVES | {tool.SIMULATED_BACKEND_ID}
    )
    spec = tct.ProjectionSpec("p", tct.ROW_PARALLEL, 8 * 64, 256)
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(32, spec.in_features, generator=gen)  # FP32 end to end: no final rounding
    weight = torch.randn(spec.out_features, spec.in_features, generator=gen)
    tp1 = tct.row_parallel_forward(x, weight, spec)
    shared = {"world": 8, "barrier": threading.Barrier(8), "slots": {}}
    outs = {}

    def worker(rank):
        outs[rank] = tct.row_parallel_forward(
            tct.shard_activation(x, spec, rank, 8),
            tct.shard_weight(weight, spec, rank, 8),
            spec,
            collective=LeftFold(shared, rank),
        )

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not torch.equal(outs[0], tp1)


def test_validation_tool_threads_end_to_end(tool, tmp_path):
    report = tmp_path / "report.json"
    code = tool.main(
        [
            "--device",
            "cpu",
            "--launcher",
            "threads",
            "--shapes",
            "tiny",
            "--tokens",
            "8",
            "--tp",
            "2",
            "4",
            "8",
            "--report",
            str(report),
        ]
    )
    assert code == 0
    assert report.exists()


def test_provenance_records_topology(tool):
    shared = {"world": 4, "barrier": threading.Barrier(4), "slots": {}}
    allowed = tct.FIXED_TREE_COLLECTIVES
    tct.FIXED_TREE_COLLECTIVES = allowed | {tool.SIMULATED_BACKEND_ID}
    try:
        prov = tct.tp_canonical_provenance(tool.ThreadedBalancedTreeCollective(shared, 3))
    finally:
        tct.FIXED_TREE_COLLECTIVES = allowed
    assert prov["physical_tp"] == 4
    assert prov["canonical_tp"] == 8
    assert prov["owned_canonical_leaves"] == [6, 7]
    assert prov["partial_dtype"] == "float32"
    assert prov["rounding"] == "once_after_tree"
    assert prov["canonical_shards"]["mlp.down_proj"] == ["row", 2496]
    assert prov["collective_backend"] == tool.SIMULATED_BACKEND_ID
