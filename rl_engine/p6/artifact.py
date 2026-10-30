# SPDX-License-Identifier: Apache-2.0
"""Immutable P6 evidence publication, offline replay and completed-attempt reuse."""

import hashlib
import json
from pathlib import Path

from .contract import SCHEMA, check_compatibility, digest, exact_keys, manifest, require
from .recordings import boundary_recordings, catalog, verify_recordings, verify_source_cases

ARTIFACT_SCHEMA = "p6-sealed-artifact.v1"


def source_hash():
    root = Path(__file__).resolve().parent
    return digest(
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    )


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        # Raw bytes live in compact frozen fixtures; avoid thousands of noise lines.
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def evidence_matrix():
    return [
        {
            "operator": op["name"],
            "owner": op["owner"],
            "level": level,
            "status": "REFERENCE_BYTES_PASS" if level == "synthetic_reference" else "NOT_CERTIFIED",
        }
        for op in manifest()["operators"]
        for level in ("synthetic_reference", "live_ws1", "live_ws2", "integration")
    ]


def build_payload(records, provenance, checks, request):
    verify_source_cases(records)
    operators = boundary_recordings(records)
    verify_recordings(operators)
    from .negative import negative_fixtures

    return {
        "schema": ARTIFACT_SCHEMA,
        "contract": manifest(),
        "catalog": catalog(records),
        "cases": records,
        "operator_recordings": operators,
        "negative_results": negative_fixtures(),
        "required_evidence": evidence_matrix(),
        "source_sha256": source_hash(),
        "provenance": provenance,
        "checks": checks,
        "request": request,
        "resume_identity": digest(
            {"request": request, "source": source_hash(), "contract": manifest(), "cases": records}
        ),
        "production_certified": False,
        "foundation_compatibility": "UNVERIFIED",
        "performance": "NOT_CERTIFIED",
    }


def publish(directory, payload):
    directory = Path(directory)
    validate_payload(payload)
    require(not directory.exists(), "OUTPUT_EXISTS", "immutable attempt already exists")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "payload.json", payload)
    raw = (directory / "payload.json").read_bytes()
    seal_manifest = {
        "schema": ARTIFACT_SCHEMA,
        "profile_schema": SCHEMA,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_bytes": len(raw),
        "resume_identity": payload["resume_identity"],
        "complete": True,
    }
    write_json(directory / "manifest.json", seal_manifest)
    with (directory / "seal.sha256").open("x") as stream:
        stream.write(digest(seal_manifest) + "\n")
    return {
        "status": "REFERENCE_ARTIFACT_PUBLISHED",
        "resume_identity": payload["resume_identity"],
        "production_certified": False,
    }


def validate_payload(payload):
    require(payload["schema"] == ARTIFACT_SCHEMA, "SCHEMA_MISMATCH", "artifact schema")
    check_compatibility(payload["contract"])
    require(
        payload["production_certified"] is False
        and payload["foundation_compatibility"] == "UNVERIFIED"
        and payload["performance"] == "NOT_CERTIFIED",
        "SCHEMA_MISMATCH",
        "evidence scope",
    )
    verify_source_cases(payload["cases"])
    require(payload["catalog"] == catalog(payload["cases"]), "CORRUPT_ARTIFACT", "catalog binding")
    expected = boundary_recordings(payload["cases"])
    require(payload["operator_recordings"] == expected, "CORRUPT_ARTIFACT", "operator set/content")
    verify_recordings(payload["operator_recordings"])
    require(
        payload["required_evidence"] == evidence_matrix(), "CORRUPT_ARTIFACT", "evidence matrix"
    )
    from .negative import negative_fixtures

    require(
        payload["negative_results"] == negative_fixtures(), "CORRUPT_ARTIFACT", "negative evidence"
    )
    require(
        type(payload["source_sha256"]) is str and len(payload["source_sha256"]) == 64,
        "MISSING_PROVENANCE",
        "source hash",
    )
    provenance = payload["provenance"]
    require(
        type(provenance) is dict
        and provenance.get("provider") in ("stdlib-scalar-oracle", "pytorch-reference-only"),
        "MISSING_PROVENANCE",
        "provider",
    )
    require(
        type(payload["checks"]) is list and bool(payload["checks"]),
        "INCOMPLETE_ARTIFACT",
        "conformance checks",
    )
    expected_resume = digest(
        {
            "request": payload["request"],
            "source": payload["source_sha256"],
            "contract": payload["contract"],
            "cases": payload["cases"],
        }
    )
    require(payload["resume_identity"] == expected_resume, "CORRUPT_ARTIFACT", "resume identity")


def read_artifact(directory):
    directory = Path(directory)
    require(
        directory.is_dir()
        and all(
            (directory / name).is_file()
            for name in ("payload.json", "manifest.json", "seal.sha256")
        ),
        "INCOMPLETE_ARTIFACT",
        "missing artifact file",
    )
    seal_manifest = json.loads((directory / "manifest.json").read_text())
    exact_keys(
        seal_manifest,
        (
            "schema",
            "profile_schema",
            "payload_sha256",
            "payload_bytes",
            "resume_identity",
            "complete",
        ),
        "artifact manifest",
    )
    require(
        seal_manifest["schema"] == ARTIFACT_SCHEMA and seal_manifest["profile_schema"] == SCHEMA,
        "SCHEMA_MISMATCH",
        "seal schema",
    )
    require(seal_manifest["complete"] is True, "INCOMPLETE_ARTIFACT", "completion flag")
    require(
        (directory / "seal.sha256").read_text().strip() == digest(seal_manifest),
        "CORRUPT_ARTIFACT",
        "seal",
    )
    raw = (directory / "payload.json").read_bytes()
    require(
        len(raw) == seal_manifest["payload_bytes"]
        and hashlib.sha256(raw).hexdigest() == seal_manifest["payload_sha256"],
        "CORRUPT_ARTIFACT",
        "payload checksum",
    )
    payload = json.loads(raw)
    validate_payload(payload)
    require(
        seal_manifest["resume_identity"] == payload["resume_identity"],
        "CORRUPT_ARTIFACT",
        "seal resume identity",
    )
    return payload, seal_manifest


def verify(directory):
    payload, _ = read_artifact(directory)
    return {
        "status": "ARTIFACT_INTEGRITY_AND_CPU_REPLAY_PASS",
        "cases": len(payload["cases"]),
        "operator_recordings": len(payload["operator_recordings"]),
        "gpu_reexecuted": False,
        "production_certified": False,
    }


def resume(directory, records, request):
    payload, _ = read_artifact(directory)
    expected = digest(
        {"request": request, "source": source_hash(), "contract": manifest(), "cases": records}
    )
    require(payload["resume_identity"] == expected, "IDENTITY_DRIFT", "resume request/source/input")
    return {
        "status": "COMPLETE_ATTEMPT_REUSED",
        "gpu_reexecuted": False,
        "production_certified": False,
        "resume_identity": expected,
    }
