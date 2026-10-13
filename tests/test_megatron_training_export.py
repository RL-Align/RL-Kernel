# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 M10 export draft: round trip, integrity, signed zero / 1-ULP self-test."""

import dataclasses
import hashlib
import json

import pytest
import torch
from p7_doubles.fixtures import vime_request
from p7_doubles.harness import FIXTURES, make_context, run, strict_observer

from rl_engine.alignment.cross_config.adapters import megatron_training_gate as gate
from rl_engine.alignment.cross_config.adapters.megatron_training import T05ContractError
from rl_engine.alignment.cross_config.adapters.megatron_training_export import (
    BYTES_NAME,
    META_NAME,
    SELF_TEST_AUTHORITY,
    export_m10,
    load_m10,
    self_test_compare,
)
from rl_engine.alignment.cross_config.adapters.megatron_training_upstream import (
    UpstreamCorrelationKey,
    UpstreamRequirement,
    correlate_upstream,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def observer(monkeypatch):
    with strict_observer(monkeypatch) as instance:
        yield instance


def _evidence(observer, name="normal", cp_rank=0):
    fixture = FIXTURES[name]
    snapshot = run(observer, vime_request(fixture, cp_rank=cp_rank))
    return gate.validate_training_call(snapshot, make_context(fixture, cp_rank=cp_rank))


def _with_values(evidence, values):
    """Same evidence with replaced FP32 values (digest kept consistent)."""

    raw = values[evidence.active_token_mask].contiguous().view(torch.int32).numpy().tobytes()
    return dataclasses.replace(
        evidence,
        selected_logprob=values,
        active_raw_bytes=raw,
        active_bytes_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _expect(status, fn):
    with pytest.raises(T05ContractError) as info:
        fn()
    assert info.value.status == status, str(info.value)


def test_round_trip_is_bit_exact(observer, tmp_path):
    evidence = _evidence(observer, "packed_varlen_zigzag_cp2", cp_rank=1)
    loaded = load_m10(export_m10(evidence, tmp_path / "a"))
    assert loaded.active_raw_bytes == evidence.active_raw_bytes
    assert torch.equal(
        loaded.active_values.view(torch.int32),
        evidence.selected_logprob[evidence.active_token_mask].view(torch.int32),
    )
    assert loaded.row_keys == evidence.row_keys
    assert loaded.meta["byte_encoding"]["endianness"] == "little"
    assert loaded.meta["evidence_fingerprint"] == evidence.fingerprint


def test_padding_rows_are_null_and_never_active(observer, tmp_path):
    evidence = _evidence(observer)
    padded = dataclasses.replace(evidence, row_keys=evidence.row_keys[:-1] + (None,))
    mask = padded.active_token_mask.clone()
    mask[-1] = False
    padded = _with_values(
        dataclasses.replace(padded, active_token_mask=mask), evidence.selected_logprob
    )
    meta = json.loads((export_m10(padded, tmp_path / "p") / META_NAME).read_text())
    assert meta["row_keys"][-1] is None and meta["active_token_mask"][-1] is False


def test_two_exports_of_one_call_are_byte_identical(observer, tmp_path):
    evidence = _evidence(observer)
    a, b = export_m10(evidence, tmp_path / "a"), export_m10(evidence, tmp_path / "b")
    for name in (META_NAME, BYTES_NAME):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    report = self_test_compare(load_m10(a), load_m10(b))
    assert report.identical and report.authority == SELF_TEST_AUTHORITY


def test_self_test_sees_signed_zero_and_one_ulp(observer, tmp_path):
    evidence = _evidence(observer)
    row = int(evidence.active_token_mask.nonzero()[0])
    base = evidence.selected_logprob.clone()
    base[row] = 0.0
    neg = base.clone()
    neg[row] = -0.0
    ulp = base.clone()
    ulp[row] = torch.nextafter(base[row], torch.tensor(-1.0))
    plain = load_m10(export_m10(_with_values(evidence, base), tmp_path / "plain"))
    for name, values in (("neg", neg), ("ulp", ulp)):
        report = self_test_compare(
            plain, load_m10(export_m10(_with_values(evidence, values), tmp_path / name))
        )
        assert not report.identical
        assert report.first_difference["token"] == list(evidence.row_keys[row])


def test_export_refuses_to_overwrite(observer, tmp_path):
    evidence = _evidence(observer)
    export_m10(evidence, tmp_path / "a")
    _expect("CORRUPT_ARTIFACT", lambda: export_m10(evidence, tmp_path / "a"))


def test_corrupt_truncated_and_missing_files(observer, tmp_path):
    evidence = _evidence(observer)
    flipped = export_m10(evidence, tmp_path / "flip")
    data = bytearray((flipped / BYTES_NAME).read_bytes())
    data[0] ^= 1
    (flipped / BYTES_NAME).write_bytes(bytes(data))
    _expect("CORRUPT_ARTIFACT", lambda: load_m10(flipped))

    missing = export_m10(evidence, tmp_path / "missing")
    (missing / BYTES_NAME).unlink()
    _expect("INCOMPLETE_ARTIFACT", lambda: load_m10(missing))

    schema = export_m10(evidence, tmp_path / "schema")
    meta = json.loads((schema / META_NAME).read_text())
    meta["schema_version"] = "v0"
    (schema / META_NAME).write_text(json.dumps(meta))
    _expect("SCHEMA_MISMATCH", lambda: load_m10(schema))


def test_tampered_digest_in_evidence_is_rejected(observer, tmp_path):
    evidence = dataclasses.replace(_evidence(observer), active_bytes_sha256="0" * 64)
    _expect("CORRUPT_ARTIFACT", lambda: export_m10(evidence, tmp_path / "x"))


def test_self_test_requires_same_tokens(observer, tmp_path):
    a = load_m10(export_m10(_evidence(observer, "packed_varlen_zigzag_cp2", 0), tmp_path / "a"))
    b = load_m10(export_m10(_evidence(observer, "packed_varlen_zigzag_cp2", 1), tmp_path / "b"))
    _expect("IDENTITY_DRIFT", lambda: self_test_compare(a, b))


def test_upstream_evidence_is_embedded(observer, tmp_path):
    evidence = _evidence(observer)
    active = tuple(
        key
        for key, on in zip(evidence.row_keys, evidence.active_token_mask.tolist(), strict=True)
        if on
    )
    requirement = UpstreamRequirement("P1", "P1.mhc.v1", "M0.pre", "1", 0, "forward", "attention")
    payload = {
        "producer": "P1",
        "upstream_contract_id": "P1.mhc.v1",
        "producer_schema_version": "1",
        "boundary": "M0.pre",
        "run_id": "run-a",
        "case_id": "case-1",
        "absolute_layer": 0,
        "sublayer_site": "attention",
        "pass_name": "forward",
        "event_index": 0,
        "global_token_ids": [list(key) for key in active],
        "rank": "tp0",
        "requested_backend": "b",
        "actual_backend": "b",
        "kernel_build": "k",
        "comparison_mode": "byte_exact",
        "producer_verdict": "PASS",
        "fixture_checksum": "c",
    }
    upstream = correlate_upstream(
        UpstreamCorrelationKey("case-1", "run-a", "tp0", active), (requirement,), [payload]
    )
    meta = load_m10(export_m10(evidence, tmp_path / "u", upstream=upstream)).meta
    assert meta["upstream_fingerprint"] == upstream.fingerprint
    assert meta["upstream"]["forwarded"][0]["payload"]["producer_verdict"] == "PASS"
