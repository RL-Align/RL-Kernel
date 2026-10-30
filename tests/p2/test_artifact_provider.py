# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
import subprocess
import sys

import pytest

from rl_engine.p2.__main__ import build_payload
from rl_engine.p2.artifact import provenance, read_artifact, resume_identity, seal, validate_payload
from rl_engine.p2.contract import (
    ABI_VERSION,
    MODES,
    REFERENCE_PROFILE,
    SCHEMA_VERSION,
    ContractError,
    Status,
    canonical,
    digest,
    runtime_policy,
)
from rl_engine.p2.provider import (
    ProviderRegistry,
    RecordedProvider,
    compare_envelopes,
    validate_envelope,
)


@pytest.fixture(scope="module")
def payload():
    return build_payload(8)


@pytest.fixture
def envelope(payload):
    states = copy.deepcopy(payload["sequences"]["C4"]["snapshots"])
    return {
        "schema": SCHEMA_VERSION,
        "foundation_abi": ABI_VERSION,
        "identity": states[0]["identity"],
        "profile": REFERENCE_PROFILE,
        "case_id": "P2-F-STATE4-training.v1",
        "mode": "training",
        "fallback": False,
        "route": "recorded",
        "provenance": {"kind": "recorded", "artifact_checksum": digest(payload)},
        "states": states,
        "boundaries": {"M2.attention": {"data": "example"}},
        "runtime_policy": runtime_policy(),
    }


def test_seal_roundtrip_and_offline_reconstruction(tmp_path, payload):
    target = tmp_path / "artifact"
    result = seal(target, payload)
    loaded, report = read_artifact(target)
    assert loaded == payload
    assert result == report
    assert report["status"] == "PASS"
    assert report["live_ws1"] == report["live_ws2"] == "UNSUPPORTED_CAPABILITY"
    assert report["performance"] == "NOT_CERTIFIED"
    assert len(report["state_modes"]) == 12
    assert report["operator_recordings"] == 15
    with pytest.raises(ContractError, match="refusing to overwrite"):
        seal(target, payload)


def test_generated_fixture_checksums_reproducible(payload):
    other = build_payload(8)
    assert payload["recordings"] == other["recordings"]
    assert payload["sequences"] == other["sequences"]


@pytest.mark.parametrize(
    "kind,status",
    [
        ("truncated", Status.CORRUPT_ARTIFACT),
        ("bitflip", Status.CORRUPT_ARTIFACT),
        ("missing", Status.INCOMPLETE_ARTIFACT),
        ("extra", Status.INCOMPLETE_ARTIFACT),
        ("unsealed", Status.CORRUPT_ARTIFACT),
        ("completion", Status.INCOMPLETE_ARTIFACT),
        ("path_escape", Status.SCHEMA_MISMATCH),
        ("duplicate_json", Status.CORRUPT_ARTIFACT),
        ("malformed_json", Status.CORRUPT_ARTIFACT),
    ],
)
def test_corrupt_artifact(tmp_path, payload, kind, status):
    target = tmp_path / "artifact"
    seal(target, payload)
    file = target / "payload.json"
    if kind == "truncated":
        file.write_bytes(file.read_bytes()[:100])
    elif kind == "bitflip":
        raw = bytearray(file.read_bytes())
        raw[100] ^= 1
        file.write_bytes(raw)
    elif kind == "missing":
        file.unlink()
    elif kind == "extra":
        (target / "extra.json").write_text("{}")
    elif kind == "unsealed":
        (target / "seal.sha256").write_text("0" * 64)
    elif kind == "duplicate_json":
        (target / "manifest.json").write_text('{"schema":"x","schema":"y"}')
    elif kind == "malformed_json":
        (target / "manifest.json").write_text("{")
    else:
        m = json.loads((target / "manifest.json").read_text())
        if kind == "completion":
            m["completion"] = False
        else:
            m["payload_file"] = "../outside.json"
        (target / "manifest.json").write_bytes(canonical(m))
        (target / "seal.sha256").write_text(digest(m))
    with pytest.raises(ContractError) as error:
        read_artifact(target)
    assert error.value.status == status


@pytest.mark.parametrize(
    "kind,status",
    [
        ("missing_mode", Status.INCOMPLETE_ARTIFACT),
        ("mode_mismatch", Status.STATE_BYTES_MISMATCH),
        ("missing_operator", Status.INCOMPLETE_ARTIFACT),
        ("matrix", Status.INCOMPLETE_ARTIFACT),
        ("fixture", Status.IDENTITY_DRIFT),
        ("provenance", Status.MISSING_PROVENANCE),
        ("state_identity", Status.IDENTITY_DRIFT),
        ("tensor_bytes", Status.CORRUPT_ARTIFACT),
    ],
)
def test_payload_semantics_not_just_outer_checksum(payload, kind, status):
    p = copy.deepcopy(payload)
    if kind == "missing_mode":
        del p["sequences"]["C0"]["mode_state_hashes"]["graph_decode"]
    elif kind == "mode_mismatch":
        p["sequences"]["C4"]["mode_state_hashes"]["prefill"][0] = "wrong"
    elif kind == "missing_operator":
        p["recordings"].pop("indexer_scale")
    elif kind == "matrix":
        p["required_evidence"].pop()
    elif kind == "fixture":
        p["catalog"][0]["fixture_checksum"] = "wrong"
    elif kind == "provenance":
        p["provenance"]["backend"] = "pretend_cuda"
    elif kind == "state_identity":
        p["sequences"]["C4"]["snapshots"][2]["identity"]["main"] = "different"
    else:
        rec = p["recordings"]["indexer_scale"]
        rec["outputs"]["w"]["shape"] = [1]
        rec["checksum"] = digest({k: v for k, v in rec.items() if k != "checksum"})
    with pytest.raises(ContractError) as err:
        validate_payload(p)
    assert err.value.status == status


def test_schema_validation_after_outer_reseal(tmp_path, payload):
    target = tmp_path / "artifact"
    seal(target, payload)
    p = copy.deepcopy(payload)
    del p["sequences"]["C0"]["mode_state_hashes"]["graph_decode"]
    raw = canonical(p)
    (target / "payload.json").write_bytes(raw)
    m = json.loads((target / "manifest.json").read_text())
    m["sha256"], m["bytes"] = hashlib.sha256(raw).hexdigest(), len(raw)
    m["resume_identity"] = resume_identity(m["sha256"])
    (target / "manifest.json").write_bytes(canonical(m))
    (target / "seal.sha256").write_text(digest(m))
    with pytest.raises(ContractError, match=Status.INCOMPLETE_ARTIFACT.value):
        read_artifact(target)


def test_cli_verifies_and_returns_failure(tmp_path, payload):
    target = tmp_path / "artifact"
    seal(target, payload)
    command = [sys.executable, "-m", "rl_engine.p2", "verify", str(target)]
    good = subprocess.run(command, capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout)["scope"] == "synthetic_reference_only"
    (target / "payload.json").write_text("{}")
    bad = subprocess.run(command, capture_output=True, text=True)
    assert bad.returncode == 1
    assert json.loads(bad.stderr)["status"] == "CORRUPT_ARTIFACT"


def test_registry_does_not_fallback(envelope):
    recorded = RecordedProvider([envelope])
    registry = ProviderRegistry()
    registry.register("recorded", recorded)
    assert registry.run("recorded", envelope["case_id"], "training") == envelope
    with pytest.raises(ContractError, match=Status.UNSUPPORTED_CAPABILITY.value):
        registry.run("live", envelope["case_id"], "training")
    with pytest.raises(ContractError, match=Status.UNSUPPORTED_CAPABILITY.value):
        registry.run("recorded", "unknown_case", "training")
    with pytest.raises(ContractError, match=Status.SILENT_FALLBACK.value):
        ProviderRegistry().register("live", recorded)
    copy_of_recording = recorded.run(envelope["case_id"], "training")
    copy_of_recording["states"].clear()
    assert recorded.run(envelope["case_id"], "training") == envelope


@pytest.mark.parametrize("mode", MODES)
def test_state_first_four_separate_verdicts(envelope, mode):
    other = copy.deepcopy(envelope)
    other["mode"] = mode
    report = compare_envelopes(envelope, other)
    assert report["mode"] == mode and report["status"] == "PASS"
    other["states"][0]["recent"][0]["data"] = "AQ=="
    other["boundaries"] = {"deliberately_invalid_output": True}
    with pytest.raises(ContractError, match=Status.STATE_BYTES_MISMATCH.value):
        compare_envelopes(envelope, other)


def test_natural_route_and_output_attribution(envelope):
    with pytest.raises(ContractError, match=Status.NATURAL_ROUTE_MISMATCH.value):
        compare_envelopes(envelope, envelope, require_natural=True)
    other = copy.deepcopy(envelope)
    other["boundaries"] = {}
    with pytest.raises(ContractError, match=Status.BYTE_MISMATCH.value):
        compare_envelopes(envelope, other)
    other["fallback"] = True
    with pytest.raises(ContractError, match=Status.SILENT_FALLBACK.value):
        validate_envelope(other)


def test_live_replacement_same_abi_without_core_changes(envelope):
    class FakeLive:
        def describe(self):
            desc = RecordedProvider([envelope]).describe()
            desc["kind"] = "live"
            return desc

        def run(self, case_id, mode):
            result = copy.deepcopy(envelope)
            result.update(case_id=case_id, mode=mode, route="natural", provenance=provenance())
            return result

    registry = ProviderRegistry()
    registry.register("live", FakeLive())
    result = registry.run("live", envelope["case_id"], "training")
    assert compare_envelopes(envelope, result, require_natural=True)["status"] == "PASS"


def test_provider_consumes_verified_sealed_artifact(tmp_path, payload):
    path = tmp_path / "provider-source"
    seal(path, payload)
    provider = RecordedProvider.from_artifact(path)
    assert len(provider.describe()["capabilities"]) == 12
    for layer in ("c0", "c4", "c128"):
        case = f"P2-F-LAYER-{layer}.v1"
        train = provider.run(case, "training")
        for mode in MODES:
            assert compare_envelopes(train, provider.run(case, mode))["status"] == "PASS"


def test_resume_identity_changes_with_input_length(tmp_path, payload):
    seal(tmp_path / "eight", payload)
    seal(tmp_path / "nine", build_payload(9))
    a = json.loads((tmp_path / "eight" / "manifest.json").read_text())
    b = json.loads((tmp_path / "nine" / "manifest.json").read_text())
    assert a["resume_identity"] != b["resume_identity"]


@pytest.mark.parametrize(
    "fault,status",
    [
        ("valid", None),
        ("empty", Status.INCOMPLETE_ARTIFACT),
        ("reordered", Status.INVALID_GLOBAL_POSITION),
        ("missing", Status.INVALID_GLOBAL_POSITION),
        ("identity", Status.IDENTITY_DRIFT),
        ("corrupt", Status.STATE_BYTES_MISMATCH),
    ],
)
def test_artifact_and_provider_share_state_gate(payload, envelope, fault, status):
    candidate = copy.deepcopy(payload)
    sequence = candidate["sequences"]["C4"]
    states = sequence["snapshots"]
    if fault == "empty":
        states.clear()
    elif fault == "reordered":
        states[2], states[3] = states[3], states[2]
    elif fault == "missing":
        del states[3]
    elif fault == "identity":
        states[3]["identity"]["main"] = "different-main"
    elif fault == "corrupt":
        states[3]["state_hash"] = "corrupt"
    sequence["tokens"] = len(states)
    envelope["states"] = states
    before = canonical(states)
    for validator, value in ((validate_payload, candidate), (validate_envelope, envelope)):
        if status is None:
            validator(value)
        else:
            with pytest.raises(ContractError) as error:
                validator(value)
            assert error.value.status == status
    assert canonical(states) == before
