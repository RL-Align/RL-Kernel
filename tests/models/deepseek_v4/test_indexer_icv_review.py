# SPDX-License-Identifier: Apache-2.0
"""Regressions for ownership, provider provenance and the distinct scale ABIs."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from rl_engine.contracts.operators.indexer_icv import (
    IndexerICVContract,
    IndexerICVError,
    source_sha256,
)
from rl_engine.integrations.common.indexer_icv import bind_global_icv_keys, bind_icv_queries
from rl_engine.models.deepseek_v4.indexer_icv import IndexerICV, QuantizedICVRows
from tests.models.deepseek_v4.icv_helpers import (
    ReferenceICVBackend,
    byte_equal,
    decode,
    ids,
    payload,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("alias_kind", ["data", "set_storage", "numpy"])
def test_snapshot_isolated_from_original_untracked_data_and_scale_aliases(alias_kind):
    c = IndexerICVContract(num_heads=2, head_dim=32)
    op = IndexerICV(c, backend=ReferenceICVBackend())
    qv, qs = payload(c, 3, q=True)
    kv, ks = payload(c, 5, seed=1)
    q = bind_icv_queries(op, qv, qs, ids(7, 3), layout=c.layout, version="v1", ownership="snapshot")
    k = bind_global_icv_keys(
        op,
        kv,
        ks,
        ids(7, 5),
        layout=c.layout,
        version="v1",
        request_key_counts={7: 5},
        ownership="snapshot",
    )
    before = op.materialize(q, k)[0].values.clone()
    originals = [qv, qs, kv, ks]
    versions = [t._version for t in originals]
    for tensor in originals:
        if alias_kind == "data":
            tensor.data.zero_()
        elif alias_kind == "set_storage":
            torch.empty(0, dtype=tensor.dtype).set_(
                tensor.untyped_storage(), tensor.storage_offset(), tensor.size(), tensor.stride()
            ).zero_()
        else:
            torch.from_numpy(tensor.numpy()).zero_()
    # These writes really bypassed the original tensors' version counters.
    assert [t._version for t in originals] == versions
    assert q.operand.ownership == k.operand.ownership == "snapshot"
    q.validate()
    k.validate()
    byte_equal(op.materialize(q, k)[0].values, before)


def test_borrowed_is_an_explicit_no_copy_lifetime_contract_not_immutable_storage():
    c = IndexerICVContract(num_heads=1, head_dim=32)
    values, scales = payload(c, 2)
    bound = QuantizedICVRows.bind(
        values, scales, role="k", version="v1", layout=c.layout, contract=c
    )
    assert bound.ownership == "borrowed"
    assert bound.data.data_ptr() == values.data_ptr()
    assert bound.scales.data_ptr() == scales.data_ptr()
    # Ordinary writes are detected. Untracked writes are forbidden by the
    # borrowing contract, not falsely advertised as detectable by _version.
    values.zero_()
    with pytest.raises(IndexerICVError, match="changed"):
        bound.validate()
    with pytest.raises(IndexerICVError, match="ownership"):
        QuantizedICVRows.bind(
            values,
            scales,
            role="k",
            version="v1",
            layout=c.layout,
            contract=c,
            ownership="immutable",
        )


@pytest.mark.parametrize("scale", [0, 1, 253, 254, 255])
def test_every_packed_byte_at_exception_scale(scale):
    c = IndexerICVContract(num_heads=1, head_dim=32)
    for byte in range(256):
        data = torch.full((1, 16), byte, dtype=torch.uint8)
        scales = torch.full((1, 1), scale, dtype=torch.uint8)
        args = dict(role="k", version="v1", layout=c.layout, contract=c)
        # Independent table definition: +0 or -0 in each nibble, no other codes.
        if byte in (0x00, 0x08, 0x80, 0x88) and scale != 255:
            QuantizedICVRows.bind(data, scales, **args).validate()
        else:
            with pytest.raises(IndexerICVError, match="exact-decode"):
                QuantizedICVRows.bind(data, scales, **args)


@pytest.mark.parametrize("encoding", ["e4m3fn", "e4m3fnuz"])
@pytest.mark.parametrize("exponent", [-60, -3, 3, 30])
def test_fp8_cpu_oracle_applies_distinct_q_head_and_k_row_multipliers_once(encoding, exponent):
    c = IndexerICVContract(num_heads=2, head_dim=32, precision="fp8", fp8_encoding=encoding)
    dtype = getattr(torch, "float8_" + encoding)
    qv = torch.ones((2, 2, 32)).to(dtype)
    kv = torch.tensor([1.0, -2.0, 4.0]).view(3, 1).expand(3, 32).to(dtype).contiguous()
    qs = torch.tensor([1, 0.5, 0.25, 1], dtype=torch.float32).reshape(2, 2, 1) * 2.0**exponent
    # Keep every exponent inside the contract even at the smallest boundary.
    if exponent == -60:
        qs.fill_(2.0**-60)
    ks = torch.tensor([[0.25], [2.0], [4.0]])
    op = IndexerICV(c, backend=ReferenceICVBackend())
    q = bind_icv_queries(op, qv, qs, ids(7, 2), layout=c.layout, version="v1")
    k = bind_global_icv_keys(
        op, kv, ks, ids(7, 3), layout=c.layout, version="v1", request_key_counts={7: 3}
    )
    expected = 32 * torch.tensor([1.0, -2.0, 4.0]).view(1, 1, 3) * (qs * ks.T)
    actual = op.materialize(q, k)[0].values
    byte_equal(actual, expected)
    # The independent float64 decoder must agree, not treat FP32 2**e as byte e.
    oracle = torch.einsum("mhd,nd->mhn", decode(qv, qs, c), decode(kv, ks, c)).float()
    byte_equal(actual, oracle)


@pytest.mark.parametrize(
    "bad", ["missing", "uint8", "shape", "zero", "negative", "nan", "inf", "small", "large", "127"]
)
def test_fp8_oracle_and_binding_reject_wrong_scale_abi(bad):
    c = IndexerICVContract(num_heads=1, head_dim=32, precision="fp8", fp8_encoding="e4m3fnuz")
    data, scales = payload(c, 2)
    if bad == "missing":
        scales = None
    elif bad == "uint8":
        scales = torch.full_like(scales, 127, dtype=torch.uint8)
    elif bad == "shape":
        scales = scales.expand(2, 2).contiguous()
    else:
        value = {
            "zero": 0,
            "negative": -1,
            "nan": float("nan"),
            "inf": float("inf"),
            "small": 2.0**-61,
            "large": 2.0**31,
            "127": 127,
        }[bad]
        scales.fill_(value)
    with pytest.raises(ValueError):
        decode(data, scales, c)
    with pytest.raises(IndexerICVError):
        QuantizedICVRows.bind(data, scales, role="k", version="v1", layout=c.layout, contract=c)


def test_source_fingerprint_readable_content_only(tmp_path):
    path = tmp_path / "provider.py"
    path.write_bytes(b"numerical implementation\n")
    assert source_sha256(path, label="provider") == hashlib.sha256(path.read_bytes()).hexdigest()
    for bad in (None, "", 12, tmp_path, tmp_path / "missing.py", "bad\0path"):
        with pytest.raises(IndexerICVError, match="source fingerprint unavailable for provider"):
            source_sha256(bad, label="provider")


@pytest.mark.parametrize("source", ["none", "missing_attribute", "missing_file"])
def test_backend_namespace_or_missing_source_fails_closed(monkeypatch, tmp_path, source):
    from rl_engine.backends.rocm.attention import indexer_icv as backend_module

    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName="gfx942")
    )
    attrs = dict(
        upcast_from_mxfp=object(),
        gemm_a16w16=object(),
        gemm_a8w8_per_token_scale=object(),
        normalize_e4m3fn_to_e4m3fnuz=object(),
        __version__="test",
    )
    if source != "missing_attribute":
        attrs["__file__"] = None if source == "none" else str(tmp_path / "missing.py")
    monkeypatch.setattr(
        backend_module.importlib, "import_module", lambda _: SimpleNamespace(**attrs)
    )
    with pytest.raises(IndexerICVError, match="source fingerprint unavailable for aiter"):
        backend_module.AiterIndexerICVBackend()


def test_model_missing_source_fails_closed(monkeypatch):
    from rl_engine.models.deepseek_v4 import indexer_icv as model_module

    monkeypatch.setattr(model_module, "__file__", None)
    with pytest.raises(IndexerICVError, match="source fingerprint unavailable"):
        IndexerICV(IndexerICVContract(), backend=ReferenceICVBackend())
