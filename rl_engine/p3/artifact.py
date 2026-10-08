"""Immutable checksum seals and input/source-bound CPU replay verification."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from . import bitmath
from .checker import check_actual_provenance, validate_events, validate_plan, validate_recording
from .contract import ARTIFACT_SCHEMA, P3Error, P3Verdict, manifest
from .fixtures import RouterCase, catalog, fixture_manifest
from .oracle import require
from .recordings import record_case
from .serialization import canonical, decode, encode, fingerprint


def kit_source_fingerprint():
    root = Path(__file__).parent
    paths = sorted(
        [
            *root.glob("*.py"),
            *root.joinpath("native").glob("*"),
        ]
    )
    return fingerprint(
        [(str(p.relative_to(root.parent)), p.read_bytes()) for p in paths if p.is_file()]
    )


def request_identity(backend="cpu"):
    return {
        "artifact_schema": ARTIFACT_SCHEMA,
        "contract": fingerprint(manifest()),
        "fixtures": fingerprint(fixture_manifest()),
        "bitmath_source": bitmath.source_fingerprint(),
        "kit_source": kit_source_fingerprint(),
        "backend": backend,
        "scope": "T01_START_KIT",
    }


def _write(path, value):
    with path.open("x") as out:
        json.dump(encode(value), out, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def create(path, *, backend="cpu", resume=False, hardware=None):
    path = Path(path)
    request = request_identity(backend)
    if resume:
        loaded = verify(path)
        require(
            loaded["request"] == request,
            P3Verdict.IDENTITY_DRIFT,
            "resume request/source/input drift",
        )
        return loaded
    require(not path.exists(), P3Verdict.CORRUPT_ARTIFACT, "new attempts require a fresh directory")
    path.mkdir(parents=True)
    cases = catalog()
    recordings = [record_case(c) for c in cases]
    validate_hardware(hardware or [], recordings, backend)
    payload = {
        "request": request,
        "contract": manifest(),
        "fixture_manifest": fixture_manifest(),
        "sources": [asdict(c) for c in cases],
        "recordings": recordings,
        "hardware": hardware or [],
        "evidence_matrix": evidence_matrix(backend),
    }
    _write(path / "artifact.json", payload)
    _write(
        path / "seal.json",
        {
            "schema": ARTIFACT_SCHEMA,
            "request_fingerprint": fingerprint(request),
            "files": {
                "artifact.json": hashlib.sha256((path / "artifact.json").read_bytes()).hexdigest()
            },
        },
    )
    fd = os.open(path, os.O_DIRECTORY | os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return verify(path)


def verify(path, *, replay=True):
    path = Path(path)
    try:
        require(
            path.is_dir() and (path / "seal.json").is_file(),
            P3Verdict.INCOMPLETE_ARTIFACT,
            "artifact not sealed",
        )
        require(
            {p.name for p in path.iterdir()} == {"artifact.json", "seal.json"}
            and not any(p.is_symlink() for p in path.iterdir()),
            P3Verdict.CORRUPT_ARTIFACT,
            "unexpected files or symlink",
        )
        seal = json.loads((path / "seal.json").read_text())
        require(seal["schema"] == ARTIFACT_SCHEMA, P3Verdict.SCHEMA_MISMATCH, "old seal")
        require(
            set(seal["files"]) == {"artifact.json"}, P3Verdict.CORRUPT_ARTIFACT, "seal file set"
        )
        raw = (path / "artifact.json").read_bytes()
        require(
            hashlib.sha256(raw).hexdigest() == seal["files"]["artifact.json"],
            P3Verdict.CORRUPT_ARTIFACT,
            "file checksum mismatch",
        )
        artifact = decode(json.loads(raw))
        require(
            artifact["request"]["artifact_schema"] == ARTIFACT_SCHEMA,
            P3Verdict.SCHEMA_MISMATCH,
            "artifact schema",
        )
        require(
            seal["request_fingerprint"] == fingerprint(artifact["request"]),
            P3Verdict.CORRUPT_ARTIFACT,
            "request seal mismatch",
        )
        require(
            artifact["request"] == request_identity(artifact["request"]["backend"]),
            P3Verdict.IDENTITY_DRIFT,
            "contract/fixture/source mismatch",
        )
        require(
            fingerprint(artifact["contract"]) == artifact["request"]["contract"]
            and fingerprint(artifact["fixture_manifest"]) == artifact["request"]["fixtures"],
            P3Verdict.IDENTITY_DRIFT,
            "embedded contract/fixture manifest mismatch",
        )
        require(
            len(artifact["sources"]) == len(artifact["recordings"]),
            P3Verdict.INCOMPLETE_ARTIFACT,
            "source/recording missing",
        )
        expected_sources = catalog()
        require(
            len(expected_sources) == len(artifact["sources"]),
            P3Verdict.INCOMPLETE_ARTIFACT,
            "missing source case",
        )
        for source, recording, expected in zip(
            artifact["sources"], artifact["recordings"], expected_sources, strict=False
        ):
            source["hidden_shape"] = tuple(source["hidden_shape"])
            case = RouterCase(**source)
            require(
                case.checksum() == expected.checksum() == recording["source_checksum"],
                P3Verdict.IDENTITY_DRIFT,
                "recording/source identity mismatch",
            )
            validate_recording(recording)
            if recording["producer_verdict"] == "PASS":
                validate_plan(recording["bundle"])
                validate_events(recording["events"])
                require(
                    recording["torch_paired"].get("checksum")
                    and recording["torch_paired"].get("status") == "RECORDED_DIAGNOSTIC",
                    P3Verdict.MISSING_PROVENANCE,
                    "Torch paired trace absent",
                )
                require(
                    recording["torch_paired"]["checksum"]
                    == fingerprint(recording["torch_paired"]["trace"]),
                    P3Verdict.CORRUPT_ARTIFACT,
                    "Torch diagnostic checksum mismatch",
                )
            if replay:
                rebuilt = record_case(case)
                require(
                    canonical({k: v for k, v in rebuilt.items() if k != "torch_paired"})
                    == canonical({k: v for k, v in recording.items() if k != "torch_paired"}),
                    P3Verdict.BYTE_MISMATCH,
                    "CPU replay changed recorded bytes",
                )
        validate_hardware(
            artifact["hardware"], artifact["recordings"], artifact["request"]["backend"]
        )
        require(
            artifact["evidence_matrix"] == evidence_matrix(artifact["request"]["backend"]),
            P3Verdict.CORRUPT_ARTIFACT,
            "unsupported or inconsistent certification claim",
        )
        return artifact
    except P3Error:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise P3Error(P3Verdict.CORRUPT_ARTIFACT, "malformed artifact") from exc


def validate_hardware(records, recordings, backend):
    if backend == "cpu":
        require(not records, P3Verdict.INVALID_PROFILE, "CPU attempt cannot assert CUDA evidence")
        return
    require(backend == "cuda", P3Verdict.UNSUPPORTED_CAPABILITY, "unknown attempt backend")
    import numpy as np

    from .stable_topk6 import cuda_source_fingerprint

    expected = {r["case_id"]: r for r in recordings if r["producer_verdict"] == "PASS"}
    keys = [(r["case_id"], r["provenance"]["block"]) for r in records]
    require(
        len(keys) == len(set(keys))
        and set(keys) == {(c, b) for c in expected for b in (1, 32, 64, 128)},
        P3Verdict.INCOMPLETE_ARTIFACT,
        "CUDA case/block coverage incomplete",
    )
    invocation_ids = []
    for record in records:
        p = record["provenance"]
        check_actual_provenance(p)
        require(
            record["verdict"] == "PASS"
            and p["actual_backend"] == "cuda"
            and p["source_fingerprint"] == cuda_source_fingerprint()
            and p["actual_binary_arch"] == 90
            and p["target_arch"] == "sm90"
            and "H100" in p["gpu"]
            and "fast-math=false" in p["build_flags"],
            P3Verdict.MISSING_PROVENANCE,
            "CUDA actual binary/device provenance missing",
        )
        case = expected[record["case_id"]]
        active = case["row_active"]
        ids = case["operators"]["stable_topk6_fwd"]["payload"]["ids"]
        require(
            record["ids"].shape == ids.shape
            and record["ids"].dtype == np.int32
            and np.array_equal(record["ids"][active], ids[active]),
            P3Verdict.TOPK_ORDER_MISMATCH,
            "archived CUDA ids differ from oracle",
        )
        invocation_ids.append((p["run_id"], p["engine_id"], p["rank"], p["invocation_id"]))
    require(
        all(0 < i[-1] < 2**64 for i in invocation_ids)
        and len(set(invocation_ids)) == len(invocation_ids),
        P3Verdict.CORRUPT_ARTIFACT,
        "CUDA invocation ids reused",
    )


def evidence_matrix(backend):
    """Only locally demonstrated gates can be asserted by this start-kit schema."""
    return [
        {"gate": "T01 CPU start kit", "verdict": "CASE_PASS"},
        {
            "gate": "CUDA stable_topk6",
            "verdict": "PASS" if backend == "cuda" else "UNSUPPORTED_CAPABILITY",
        },
        {"gate": "Miles anchor / recorded L3b", "verdict": "MISSING_PROVENANCE"},
        {"gate": "Foundation approved binding", "verdict": "NOT_CERTIFIED"},
        {"gate": "T02–T06 CUDA WS1", "verdict": "NOT_CERTIFIED"},
        {"gate": "WS2 multi-GPU", "verdict": "NOT_CERTIFIED"},
        {"gate": "P3 R0", "verdict": "NOT_CERTIFIED"},
        {"gate": "P4/P6/P7 live Integration", "verdict": "NOT_CERTIFIED"},
    ]
