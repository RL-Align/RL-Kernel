from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.attention import AttentionContractError
from rl_engine.contracts.operators.indexer_training import IndexerAuxContract
from rl_engine.models.deepseek_v4.indexer_training import IndexerAuxMetadata


def fixture(cp=2):
    c = IndexerAuxContract(8, 0.01, 64, 128, 512, selected_capacity=4, cp_size=cp)
    q = torch.tensor([[7, 11], [99, 3], [-1, -1]])
    k = torch.tensor([[7, 2], [7, 0], [99, 0], [7, 1]])
    ids = torch.tensor([[2, 0, 2, 1], [0, -1, -1, -1], [-1, -1, -1, -1]])
    owners = torch.tensor([1, 0, 1, 0]) if cp == 2 else torch.zeros(4, dtype=torch.int64)
    return c, q, k, ids, owners


@pytest.mark.parametrize(
    "changes",
    [
        {"global_query_count": 0},
        {"global_query_count": True},
        {"loss_coeff": -1},
        {"loss_coeff": float("nan")},
        {"num_heads": 0},
        {"num_heads": 129},
        {"teacher_heads": 129},
        {"teacher_dim": 256},
        {"head_dim": 64},
        {"ratio": 128},
        {"selected_capacity": 513},
        {"selected_capacity": 0},
        {"precision": "auto"},
        {"cp_size": 0},
        {"teacher_scale": 0},
        {"teacher_scale": float("inf")},
        {"graph_capture": True},
        {"quant_vjp": "implicit"},
        {"loss_profile": "other"},
    ],
)
def test_aux_contract_rejects(changes):
    with pytest.raises(AttentionContractError):
        replace(fixture()[0], **changes)


def test_canonical_slots_duplicates_and_owner_snapshots():
    c, q, k, ids, owners = fixture()
    meta = IndexerAuxMetadata.bind(c, q, k, ids, version="v1", key_owners=owners)
    assert meta.selected_group_ids.tolist() == [[0, 1, 2, 2], [0, -1, -1, -1], [-1] * 4]
    assert meta.selected_indices.tolist() == [[1, 3, 0, 0], [2, -1, -1, -1], [-1] * 4]
    assert meta.valid_rows.tolist() == [True, True, False]
    q.fill_(-1)
    k.fill_(999)
    ids.fill_(-1)
    owners.zero_()
    meta.validate(c, 3, 4, torch.device("cpu"), "v1")
    assert meta.key_owners.tolist() == [1, 0, 1, 0]
    assert meta.queries.keys.tolist()[0] == [7, 11]


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_q",
        "duplicate_k",
        "bad_padding",
        "future",
        "missing",
        "padding_selected",
        "bad_sentinel",
        "missing_owners",
        "owner_range",
        "count",
        "dtype",
    ],
)
def test_aux_admission_rejects(fault):
    c, q, k, ids, owners = fixture()
    if fault == "duplicate_q":
        q[1] = q[0]
    if fault == "duplicate_k":
        k[1] = k[0]
    if fault == "bad_padding":
        q[-1, 1] = 0
    if fault == "future":
        q[0, 1] = 4
    if fault == "missing":
        ids[0, 0] = 99
    if fault == "padding_selected":
        ids[-1, 0] = 0
    if fault == "bad_sentinel":
        ids[-1, 0] = -2
    if fault == "missing_owners":
        owners = None
    if fault == "owner_range":
        owners[0] = 2
    if fault == "count":
        c = replace(c, global_query_count=1)
    if fault == "dtype":
        ids = ids.float()
    with pytest.raises(AttentionContractError):
        IndexerAuxMetadata.bind(c, q, k, ids, version="v1", key_owners=owners)


@pytest.mark.parametrize(
    "field", ["key_ids", "key_owners", "selected_indices", "selected_group_ids", "valid_rows"]
)
def test_aux_snapshot_mutation(field):
    c, q, k, ids, owners = fixture()
    meta = IndexerAuxMetadata.bind(c, q, k, ids, version="v1", key_owners=owners)
    getattr(meta, field).zero_()
    with pytest.raises(AttentionContractError, match="mutated"):
        meta.validate(c, 3, 4, torch.device("cpu"), "v1")


def test_empty_and_single_owner_default():
    c = replace(fixture()[0], cp_size=1)
    meta = IndexerAuxMetadata.bind(
        c,
        torch.empty((0, 2), dtype=torch.int64),
        torch.empty((0, 2), dtype=torch.int64),
        torch.empty((0, 4), dtype=torch.int64),
        version="v1",
    )
    meta.validate(c, 0, 0, torch.device("cpu"), "v1")
    assert meta.selected_indices.shape == (0, 4) and meta.key_owners.shape == (0,)


def test_megatron_exports_same_training_boundary():
    from rl_engine.integrations.engines.train.megatron.indexer_training import (
        IndexerAuxiliary,
        IndexerAuxMetadata as EngineMetadata,
    )
    from rl_engine.models.deepseek_v4.indexer_training import IndexerAuxiliary as Shared

    assert IndexerAuxiliary is Shared and EngineMetadata is IndexerAuxMetadata
