import gzip
import json
from pathlib import Path

import pytest

from rl_engine.p3.contract import manifest
from rl_engine.p3.fixtures import catalog, fixture_manifest
from rl_engine.p3.recordings import record_case
from rl_engine.p3.serialization import canonical, decode, encode, fingerprint

DATA = Path(__file__).with_name("data")


def test_frozen_machine_contract_and_catalog():
    assert json.loads((DATA / "contract.v23.json").read_text()) == encode(manifest())
    assert json.loads((DATA / "catalog.v2.json").read_text()) == encode(fixture_manifest())


@pytest.mark.parametrize("case", catalog(), ids=lambda c: c.case_id)
def test_frozen_golden(case):
    golden = decode(json.loads(gzip.decompress((DATA / "golden.v2.json.gz").read_bytes())))
    expected = next(r for r in golden["cases"] if r["case_id"] == case.case_id)
    actual = record_case(case)
    assert case.checksum() == expected["source_checksum"]
    assert actual["producer_verdict"] == expected["producer_verdict"]
    for operator, op in actual["operators"].items():
        assert fingerprint(op["payload"]) == expected["operators"][operator]["payload_checksum"]
        assert canonical(op["payload"]) == canonical(expected["operators"][operator]["payload"])
    assert canonical(actual["saved"]) == canonical(expected["saved"])
    assert expected["torch_paired_required"]
    if case.row_active.any():
        assert actual["torch_paired"]["checksum"] == fingerprint(actual["torch_paired"]["trace"])


def test_l3b_pending_never_labels_synthetic_as_miles():
    pending = json.loads((DATA / "l3b.pending.v1.json").read_text())
    assert pending["verdict"] == "MISSING_PROVENANCE"
    assert not pending["recorded_artifacts"]


def test_v22_original_numerical_and_saved_bytes_preserved():
    """v23 may add cases and metadata, never rewrite the old mathematical baseline."""
    legacy = decode(json.loads(gzip.decompress((DATA / "golden.v1.json.gz").read_bytes())))
    current = {c.case_id: c for c in catalog()}
    for expected in legacy["cases"]:
        case = current[expected["case_id"]]
        actual = record_case(case)
        assert case.checksum() == expected["source_checksum"]
        for name, op in expected["operators"].items():
            assert canonical(actual["operators"][name]["payload"]) == canonical(op["payload"])
        assert canonical(actual["saved"]) == canonical(expected["saved"])
