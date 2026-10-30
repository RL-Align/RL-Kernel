#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Collect candidates, seal locks, and validate identity/configuration provenance.

This CLI reads supplied evidence. ``enforce`` rejects until a real engine adapter
supplies resolved calls and completion evidence.
Numerical output comparison belongs to the existing operator/model harness.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from rl_engine.contracts.strict_provider import (  # noqa: E402
    ProviderLock,
    StrictProviderError,
)
from rl_engine.validation.common.strict_provider import (  # noqa: E402
    RESULT_SCHEMA,
    read_json,
    validate_run,
)


class _UsageError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _UsageError(message)


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _early_path(argv: list[str], option: str) -> Path | None:
    output: Path | None = None
    for index, value in enumerate(argv):
        if value.startswith(option + "="):
            output = Path(value.split("=", 1)[1])
        elif value == option and index + 1 < len(argv):
            output = Path(argv[index + 1])
    return output


def _failure(error: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "failed",
        "manifest_enforcement": "failed",
        "numerical_parity": "not_evaluated",
        "hardware_validation": "not_evaluated",
        "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    output = _early_path(argv, "--output")
    try:
        early_input = _early_path(argv, "--input")
        if (
            output is not None
            and early_input is not None
            and (output.resolve() == early_input.resolve())
        ):
            # Preserve the input; there is no distinct result path to invalidate.
            output = None
            raise _UsageError("--input and --output must be different files")
        # Invalidate a previous successful result before parsing/loading inputs.
        if output is not None:
            _write_atomic(
                output,
                {
                    "schema_version": RESULT_SCHEMA,
                    "status": "incomplete",
                    "manifest_enforcement": "not_evaluated",
                    "numerical_parity": "not_evaluated",
                    "hardware_validation": "not_evaluated",
                },
            )
        parser = _Parser(description=__doc__)
        parser.add_argument("operation", choices=("collect", "seal", "validate", "enforce"))
        parser.add_argument("--input", required=True, type=Path)
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--evidence-root", type=Path)
        args = parser.parse_args(argv)
        output = args.output
        if args.input.resolve() == output.resolve():
            raise _UsageError("--input and --output must be different files")
        if args.operation == "enforce":
            raise StrictProviderError(
                "execution_adapter_required",
                "Use a live observer/session runner; this CLI does not execute an engine",
            )
        supplied = read_json(args.input)
        if args.operation in {"collect", "seal"}:
            candidate = supplied.get("candidate", supplied)
            lock = ProviderLock.from_dict(candidate)
            if args.operation == "collect":
                result = {
                    "schema_version": RESULT_SCHEMA,
                    "status": "candidate",
                    "operation": "collect",
                    "collection_source": "supplied_structured_candidate",
                    "manifest_enforcement": "not_evaluated",
                    "numerical_parity": "not_evaluated",
                    "hardware_validation": "not_evaluated",
                    "candidate": lock.to_dict(),
                }
            else:
                # The sealed lock is directly consumable by ProviderLock.from_dict.
                result = lock.to_dict()
        else:
            if args.evidence_root is None:
                raise _UsageError("validate requires --evidence-root")
            allowed = {
                "lock",
                "expectation",
                "worker_summaries",
                "case_results",
            }
            if set(supplied) - allowed or not {
                "lock",
                "expectation",
                "worker_summaries",
                "case_results",
            } <= set(supplied):
                raise StrictProviderError("invalid_evidence", "Invalid validation bundle fields")
            result = validate_run(
                supplied["lock"],
                supplied["expectation"],
                supplied["worker_summaries"],
                args.evidence_root,
                case_results=supplied["case_results"],
            )
        _write_atomic(output, result)
        return 0
    except _UsageError as exc:
        error = {"code": "cli_usage_error", "message": str(exc)}
        code = 2
    except StrictProviderError as exc:
        error = exc.to_dict()
        code = 1
    except KeyboardInterrupt:
        error = {"code": "cancelled", "message": "Operation interrupted"}
        code = 1
    except Exception as exc:
        error = {"code": "runtime_error", "message": f"{type(exc).__name__}: {exc}"}
        code = 1
    if output is not None:
        try:
            _write_atomic(output, _failure(error))
        except OSError as exc:
            print(f"Cannot write failure result: {exc}", file=sys.stderr)
    print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
