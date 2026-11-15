# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from rl_engine.backends.rocm.attention.indexer_topk import TritonIndexerTopKBackend
from rl_engine.contracts.operators.indexer_topk import IndexerTopKContract, IndexerTopKError
from rl_engine.models.deepseek_v4.indexer_topk import IndexerTopK
from tests.models.deepseek_v4.rope_mxfp4_helpers import byte_equal
from tests.models.deepseek_v4.topk_helpers import reference_topk

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU"
)


@pytest.fixture(scope="module")
def op():
    return IndexerTopK()


def scores(m, n, *, seed=0, ties=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(m, n, generator=g, device="cuda")
    return torch.round(s * ties) / ties if ties else s


def check(op, s, lengths=None):
    if lengths is None:
        lengths = torch.full((s.shape[0],), s.shape[1], dtype=torch.int32, device="cuda")
    out = op(s, lengths)
    byte_equal(out.cpu(), reference_topk(s, lengths, 512))
    return out


@pytest.mark.parametrize("n", [0, 1, 511, 512, 513, 4096, 70000])
@pytest.mark.parametrize("ties", [0, 2])
def test_matches_reference(op, n, ties):
    check(op, scores(3, n, seed=n, ties=ties))


def test_tie_rules(op):
    s = torch.full((3, 2000), -1.0, device="cuda")
    s[0, 500::3] = 1.0
    s[0, :500] = 2.0
    s[1, :1024] = 1.0
    s[1, 1023] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)).item()
    s[2, ::2] = -0.0
    s[2, 1::2] = 0.0
    s[2, 700] = torch.finfo(torch.float32).tiny / 4
    out = check(op, s).tolist()
    assert out[0][0] == 500 + 3 * 11 and 500 + 3 * 12 not in out[0]
    assert 1023 in out[1] and 511 not in out[1] and 700 in out[2]


def test_neg_inf_and_lengths(op):
    s = scores(4, 3000, seed=2)
    s[0] = float("-inf")
    s[1, 10:] = float("-inf")
    s[2, 7:] = float("nan")
    out = check(op, s, torch.tensor([3000, 3000, 7, 0], device="cuda"))
    assert out[0].eq(-1).all() and out[3].eq(-1).all()
    assert out[1, :10].tolist() == list(range(9, -1, -1))
    assert out[2, :7].tolist() == list(range(6, -1, -1))


def test_batch_invariance(op):
    s = scores(64, 5000, seed=3, ties=4)
    lengths = torch.randint(0, 5001, (64,), device="cuda", dtype=torch.int32)
    full = check(op, s, lengths)
    for rows in (slice(0, 1), slice(10, 40), slice(63, 64)):
        byte_equal(op(s[rows].contiguous(), lengths[rows].contiguous()), full[rows])
    order = torch.randperm(64, device="cuda")
    byte_equal(op(s[order].contiguous(), lengths[order].contiguous()), full[order])
    byte_equal(op(torch.cat((s, s), 1)[:, :5000], lengths), full)
    byte_equal(op(s, lengths), full)


def test_smaller_k():
    s = scores(3, 1000, seed=7, ties=2)
    out = IndexerTopK(IndexerTopKContract(topk=64))(s)
    byte_equal(out.cpu(), reference_topk(s, torch.full((3,), 1000), 64))


def test_feeds_auxiliary_metadata(op):
    from rl_engine.contracts.operators.indexer_training import IndexerAuxContract
    from rl_engine.models.deepseek_v4.indexer_training import IndexerAuxMetadata

    c = IndexerAuxContract(2, 0.01, 8, 8, 128, selected_capacity=512)
    queries = torch.tensor([[0, 2799], [0, 1199]], device="cuda")
    keys = torch.stack((torch.zeros(700, dtype=torch.int64), torch.arange(700)), 1).cuda()
    ids = op(scores(2, 700, seed=6, ties=2), (queries[:, 1] + 1) // 4)
    meta = IndexerAuxMetadata.bind(c, queries, keys, ids, version="v1")
    byte_equal(meta.selected_group_ids.int(), ids)


def test_rejects(op, monkeypatch):
    s = scores(2, 600)
    for args, match in (
        ((s.cpu(),), "device"),
        ((s, torch.tensor([601, 0], device="cuda")), "length"),
        ((s.t(),), "column stride"),
    ):
        with pytest.raises(IndexerTopKError, match=match):
            op(*args)
    with pytest.raises(IndexerTopKError, match="Graph"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            op(s)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *_: SimpleNamespace(gcnArchName="gfx950")
    )
    with pytest.raises(IndexerTopKError, match="gfx950"):
        TritonIndexerTopKBackend()


def test_fresh_processes(tmp_path):
    outputs = []
    for run in range(2):
        output = tmp_path / f"run{run}.json"
        env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / f"triton{run}"))
        result = subprocess.run(
            [sys.executable, "-m", "tests.models.deepseek_v4.topk_process_probe", str(output)],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(output.read_text()))
    assert outputs[0] == outputs[1]
