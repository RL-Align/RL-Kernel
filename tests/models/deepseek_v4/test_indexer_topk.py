# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.attention import AttentionContractError
from rl_engine.contracts.operators.indexer_topk import IndexerTopKContract
from rl_engine.models.deepseek_v4.indexer_topk import IndexerTopK
from tests.models.deepseek_v4.topk_helpers import ReferenceTopKBackend


def small():
    return IndexerTopK(IndexerTopKContract(topk=4), backend=ReferenceTopKBackend())


@pytest.mark.parametrize(
    "changes",
    [{"topk": 0}, {"topk": 513}, {"topk": 300}, {"graph_capture": True}, {"backend_id": "x"}],
)
def test_contract_rejects(changes):
    with pytest.raises(AttentionContractError):
        replace(IndexerTopKContract(), **changes)


def test_selection_rules():
    inf = float("-inf")
    scores = torch.tensor(
        [
            [1.0, 3.0, 3.0, 2.0, 3.0, 0.0],
            [0.0, -0.0, 0.0, -0.0, 1.0, 1.0],
            [inf, 2.0, inf, 1.0, inf, inf],
        ]
    )
    assert small()(scores).tolist() == [[4, 3, 2, 1], [5, 4, 1, 0], [3, 1, -1, -1]]


def test_rejects_invalid_inputs():
    op = small()
    for args, match in (
        ((torch.zeros(2, 4, dtype=torch.float64),), "FP32"),
        ((torch.tensor([[0.0, float("nan")]]),), "finite or -inf"),
        ((torch.tensor([[float("inf"), 0.0]]),), "finite or -inf"),
        ((torch.zeros(1, 4), torch.tensor([5])), "length"),
    ):
        with pytest.raises(AttentionContractError, match=match):
            op(*args)
    assert op(torch.tensor([[0.0, float("nan")]]), torch.tensor([1])).tolist() == [[0, -1, -1, -1]]
    with pytest.raises(AttentionContractError, match="fingerprint"):
        IndexerTopK(
            IndexerTopKContract(topk=4), backend=ReferenceTopKBackend(), expected_implementation="x"
        )
