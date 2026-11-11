# SPDX-License-Identifier: Apache-2.0
"""Real-kernel Indexer score acceptance: byte-level BI and accuracy are checked separately."""

from dataclasses import replace
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from rl_engine.backends.rocm.attention.indexer_score import AiterIndexerScoreBackend
from rl_engine.contracts.operators.indexer_score import IndexerScoreContract, IndexerScoreError
from rl_engine.models.deepseek_v4.indexer_score import IndexerScore
from tests.models.deepseek_v4.score_helpers import (
    byte_equal,
    head_partials_fp64,
    normalized_error,
    score_fp64,
    vjp_fp64,
)

pytestmark = [
    pytest.mark.ws1,
    pytest.mark.skipif(
        torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU"
    ),
]


@pytest.fixture(scope="module")
def backend():
    return AiterIndexerScoreBackend()


@pytest.fixture(scope="module")
def op(backend):
    return IndexerScore(IndexerScoreContract(num_heads=64, head_dim=128), backend=backend)


def inputs(m, n, *, h=64, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    icv = torch.randn(m, h, n, generator=g, device="cuda") * 4
    return icv, torch.randn(m, h, generator=g, device="cuda")


@pytest.mark.parametrize("m, n", [(1, 1), (7, 129), (64, 2048), (256, 4096)])
def test_full_geometry_matches_fp64(op, m, n):
    icv, w = inputs(m, n)
    scores = op.forward(icv, w).scores
    assert scores.shape == (m, n) and scores.dtype == torch.float32
    assert normalized_error(scores, score_fp64(icv, w, op.contract)) < 1e-6


def test_bytes_match_pr631_composition(op, backend):
    icv, w = inputs(9, 333)
    relu = torch.relu(icv * (128**-0.5))
    weighted = w * (64**-0.5)
    expected = backend.row_dot(
        relu.transpose(1, 2).contiguous(), weighted[:, None, :].expand(-1, icv.shape[2], -1)
    )
    byte_equal(op.forward(icv, w).scores, expected)


def test_head_partials_match_fp64_and_are_batch_invariant(op):
    icv, w = inputs(9, 333, seed=9)
    record = op.forward(icv, w)
    partials = record.head_partials()
    expected = head_partials_fp64(icv, w, op.contract)
    assert normalized_error(partials.flatten(1), expected.flatten(1)) < 1e-6
    torch.testing.assert_close(partials.sum(1), record.scores, rtol=1e-5, atol=1e-5)
    byte_equal(op.forward(icv[3:5], w[3:5]).head_partials(), partials[3:5])


def test_batch_composition_and_key_count_do_not_change_bytes(op):
    icv, w = inputs(96, 1500, seed=3)
    full = op.forward(icv, w).scores
    for rows in (slice(0, 1), slice(5, 6), slice(17, 49), slice(95, 96)):
        byte_equal(op.forward(icv[rows], w[rows]).scores, full[rows])
    order = torch.randperm(96, device="cuda")
    byte_equal(op.forward(icv[order], w[order]).scores, full[order])
    for n in (1, 31, 32, 33, 1024, 1499):
        byte_equal(op.forward(icv[:, :, :n].contiguous(), w).scores, full[:, :n])
    padded = torch.cat((icv, torch.randn(8, 64, 1500, device="cuda")))
    padded_w = torch.cat((w, torch.randn(8, 64, device="cuda")))
    byte_equal(op.forward(padded, padded_w).scores[:96], full)


def test_noncontiguous_icv_gives_same_bytes(op):
    icv, w = inputs(12, 200, seed=4)
    strided = icv.transpose(0, 2).contiguous().transpose(0, 2)
    assert not strided.is_contiguous()
    byte_equal(op.forward(strided, w).scores, op.forward(icv, w).scores)


def test_repeated_calls_are_identical(op):
    icv, w = inputs(32, 777, seed=5)
    first = op.forward(icv, w).scores
    for _ in range(5):
        byte_equal(op.forward(icv, w).scores, first)


def test_backward_matches_fp64_and_is_batch_invariant(op):
    icv, w = inputs(40, 512, seed=6)
    grad = torch.randn(40, 512, device="cuda")
    d_icv, d_w = op.backward(op.forward(icv, w), grad)
    ref_icv, ref_w = vjp_fp64(icv, w, grad, op.contract)
    assert normalized_error(d_icv.flatten(1), ref_icv.flatten(1)) < 1e-6
    assert normalized_error(d_w, ref_w) < 1e-6
    for rows in (slice(0, 1), slice(13, 29)):
        part_icv, part_w = op.backward(op.forward(icv[rows], w[rows]), grad[rows])
        byte_equal(part_icv, d_icv[rows])
        byte_equal(part_w, d_w[rows])


def test_relu_zero_has_zero_gradient(op):
    icv, w = inputs(3, 64, seed=7)
    icv[:, :, ::4] = 0
    d_icv, _ = op.backward(op.forward(icv, w), torch.ones(3, 64, device="cuda"))
    assert d_icv[:, :, ::4].eq(0).all()


def test_tp_shards_use_global_head_scale(backend, op):
    shard = IndexerScore(replace(op.contract, tp_size=2), backend=backend)
    icv, w = inputs(16, 300, seed=8)
    partial = shard.forward(icv[:, :32].contiguous(), w[:, :32].contiguous()).scores
    rest = shard.forward(icv[:, 32:].contiguous(), w[:, 32:].contiguous()).scores
    torch.testing.assert_close(partial + rest, op.forward(icv, w).scores, rtol=1e-5, atol=1e-5)
    with pytest.raises(IndexerScoreError, match="local_heads"):
        shard.forward(icv, w)


def test_empty_inputs(op):
    assert op.forward(*inputs(0, 5)).scores.shape == (0, 5)
    record = op.forward(*inputs(3, 0))
    assert record.scores.shape == (3, 0)
    d_icv, d_w = op.backward(record, torch.zeros(3, 0, device="cuda"))
    assert d_icv.shape == (3, 64, 0) and d_w.eq(0).all()


def test_rejects_cpu_tensors_and_graph_capture(op):
    icv, w = inputs(2, 8)
    with pytest.raises(IndexerScoreError, match="device"):
        op.forward(icv.cpu(), w.cpu())
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(IndexerScoreError, match="Graph"):
        with torch.cuda.graph(graph):
            op.forward(icv, w)


def test_unvalidated_architecture_is_rejected(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *_: SimpleNamespace(gcnArchName="gfx942")
    )
    with pytest.raises(IndexerScoreError, match="unvalidated architecture gfx942"):
        AiterIndexerScoreBackend()


def test_fresh_processes_with_distinct_jit_caches_produce_identical_bytes(tmp_path):
    outputs = []
    for run in range(2):
        output = tmp_path / f"run{run}.json"
        env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / f"triton{run}"))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.models.deepseek_v4.score_process_probe",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(json.loads(output.read_text()))
    assert outputs[0] == outputs[1]
