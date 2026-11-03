# SPDX-License-Identifier: Apache-2.0
"""CPU-safe negative/adversarial tests for the raw quantized ICV contract."""

from dataclasses import replace

import pytest
import torch

from rl_engine.contracts.operators.indexer_icv import IndexerICVContract, IndexerICVError
from rl_engine.integrations.common.indexer_icv import bind_icv_queries, bind_global_icv_keys
from rl_engine.models.deepseek_v4.indexer_icv import IndexerICV, QuantizedICVRows
from tests.models.deepseek_v4.icv_helpers import ReferenceICVBackend, ids, payload

pytestmark = pytest.mark.unit


def setup():
    c = IndexerICVContract(num_heads=3, head_dim=33, query_tile_rows=3, key_tile_rows=5)
    return c, IndexerICV(c, backend=ReferenceICVBackend())


@pytest.mark.parametrize(
    "problem",
    [
        "q_rank",
        "q_heads",
        "packed_width",
        "data_dtype",
        "scale_rank",
        "scale_width",
        "scale_dtype",
        "noncontiguous_data",
        "noncontiguous_scales",
        "unknown_role",
        "empty_version",
        "nonstring_version",
    ],
)
def test_bad_quantized_abi_before_backend(problem):
    c, _ = setup()
    data, scales = payload(c, 4, q=True)
    role, version = "q", "v1"
    if problem == "q_rank":
        data = data.flatten(0, 1)
    elif problem == "q_heads":
        data, scales = data[:, :2].contiguous(), scales[:, :2].contiguous()
    elif problem == "packed_width":
        data = data[..., :-1].contiguous()
    elif problem == "data_dtype":
        data = data.float()
    elif problem == "scale_rank":
        scales = scales.flatten(0, 1)
    elif problem == "scale_width":
        scales = scales[..., :1].contiguous()
    elif problem == "scale_dtype":
        scales = scales.float()
    elif problem == "noncontiguous_data":
        data = data.transpose(0, 1).contiguous().transpose(0, 1)
    elif problem == "noncontiguous_scales":
        scales = scales.transpose(0, 1).contiguous().transpose(0, 1)
    elif problem == "unknown_role":
        role = "scores"
    elif problem == "empty_version":
        version = ""
    else:
        version = 7
    with pytest.raises(IndexerICVError):
        QuantizedICVRows.bind(data, scales, role=role, version=version, layout=c.layout, contract=c)


@pytest.mark.parametrize(
    "problem",
    [
        "partial_padding",
        "negative_request",
        "negative_position",
        "duplicate_query",
        "wrong_id_dtype",
        "wrong_id_shape",
        "undeclared_key",
        "nonzero_key_start",
        "wrong_count",
        "negative_count",
        "bool_count",
        "bad_owner_dtype",
        "negative_owner",
        "bool_cp_size",
    ],
)
def test_corrupt_global_metadata_is_rejected(problem):
    c, op = setup()
    qdata, qscale = payload(c, 3, q=True)
    kdata, kscale = payload(c, 5, seed=3)
    qid, kid = ids(7, 3), ids(7, 5)
    counts, owners, cp_size = {7: 5}, torch.zeros(5, dtype=torch.int64), 2
    if problem in (
        "partial_padding",
        "negative_request",
        "negative_position",
        "duplicate_query",
        "wrong_id_dtype",
        "wrong_id_shape",
    ):
        if problem == "partial_padding":
            qid[0] = torch.tensor([-1, 0])
        elif problem == "negative_request":
            qid[0, 0] = -2
        elif problem == "negative_position":
            qid[0, 1] = -1
        elif problem == "duplicate_query":
            qid[1] = qid[0]
        elif problem == "wrong_id_dtype":
            qid = qid.int()
        else:
            qid = qid.flatten()
        with pytest.raises(IndexerICVError):
            bind_icv_queries(op, qdata, qscale, qid, layout=c.layout, version="v1")
        return
    if problem == "undeclared_key":
        kid[0, 0] = 8
    elif problem == "nonzero_key_start":
        kid[:, 1] += 1
    elif problem == "wrong_count":
        counts = {7: 4}
    elif problem == "negative_count":
        counts = {7: -1}
    elif problem == "bool_count":
        counts = {7: True}
    elif problem == "bad_owner_dtype":
        owners = owners.float()
    elif problem == "negative_owner":
        owners[0] = -1
    else:
        cp_size = True
    with pytest.raises(IndexerICVError):
        bind_global_icv_keys(
            op,
            kdata,
            kscale,
            kid,
            layout=c.layout,
            version="v1",
            request_key_counts=counts,
            key_owners=owners,
            cp_size=cp_size,
        )


@pytest.mark.parametrize("scale", [0, 1, 253, 254])
def test_finite_scale_exceptions_require_every_nibble_in_group_to_be_zero(scale):
    c, _ = setup()
    data = torch.full((1, c.padded_dim // 2), 0x88, dtype=torch.uint8)
    scales = torch.full((1, c.padded_dim // 32), scale, dtype=torch.uint8)
    allowed = QuantizedICVRows.bind(
        data, scales, role="k", version="v1", layout=c.layout, contract=c
    )
    allowed.validate()
    data[0, 0] = 0x89  # One nonzero low nibble; high nibble still negative zero.
    with pytest.raises(IndexerICVError, match="exact-decode"):
        QuantizedICVRows.bind(data, scales, role="k", version="v1", layout=c.layout, contract=c)


def test_nan_scale_is_rejected_even_for_all_zero_payload():
    c, _ = setup()
    with pytest.raises(IndexerICVError):
        QuantizedICVRows.bind(
            torch.zeros((1, 32), dtype=torch.uint8),
            torch.full((1, 2), 255, dtype=torch.uint8),
            role="k",
            version="v1",
            layout=c.layout,
            contract=c,
        )


@pytest.mark.parametrize("encoding,nan_code", [("e4m3fnuz", 128), ("e4m3fn", 127), ("e4m3fn", 255)])
def test_fp8_nan_payload_bits_are_rejected(encoding, nan_code):
    c = IndexerICVContract(num_heads=1, head_dim=128, precision="fp8", fp8_encoding=encoding)
    data, scales = payload(c, 1)
    data.view(torch.uint8)[0, 0] = nan_code
    with pytest.raises(IndexerICVError, match="non-finite"):
        QuantizedICVRows.bind(data, scales, role="k", version="v1", layout=c.layout, contract=c)


@pytest.mark.parametrize(
    "version", ["model2-preproc1-kv1", "model1-preproc2-kv1", "model1-preproc1-kv2"]
)
def test_weight_scale_and_request_kv_epoch_barriers(version):
    c, op = setup()
    q = bind_icv_queries(op, *payload(c, 3, q=True), ids(7, 3), layout=c.layout, version=version)
    keys = bind_global_icv_keys(
        op,
        *payload(c, 5),
        ids(7, 5),
        layout=c.layout,
        version="model1-preproc1-kv1",
        request_key_counts={7: 5},
    )
    with pytest.raises(IndexerICVError, match="version mismatch"):
        op.materialize(q, keys)


def test_contract_and_prepared_representation_cannot_be_mixed():
    c, op = setup()
    q = bind_icv_queries(op, *payload(c, 3, q=True), ids(7, 3), layout=c.layout, version="v1")
    keys = bind_global_icv_keys(
        op, *payload(c, 5), ids(7, 5), layout=c.layout, version="v1", request_key_counts={7: 5}
    )
    with pytest.raises(IndexerICVError, match="contract"):
        IndexerICV(replace(c, key_tile_rows=7), backend=ReferenceICVBackend()).materialize(q, keys)
    # Same semantic contract but a different provider fingerprint cannot consume
    # a stale prepared workspace just because its shape happens to match.
    altered = ReferenceICVBackend()
    altered.provenance = {"different_library_build": True}
    other = IndexerICV(c, backend=altered)
    with pytest.raises(IndexerICVError, match="backend"):
        other.materialize(q, keys)
