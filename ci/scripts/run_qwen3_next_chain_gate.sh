#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Qwen3-Next chain-report adapter checks; with a report path, also validate that report.
#   bash ci/scripts/run_qwen3_next_chain_gate.sh [chain-report.json]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-python3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m pytest -q tests/models/qwen3_next/test_qwen3_next_chain_report.py
if [ "$#" -ge 1 ]; then
  "$PY" - "$1" <<'PY'
import dataclasses, json, sys

from rl_engine.validation.models.chain_gate import ChainGateReport
from rl_engine.validation.models.qwen3_next.qwen3_next_chain_report import COMPATIBLE_SCHEMA, SCHEMA_VERSION



def invalid(message):
    """Reject a malformed report with exit 2, apart from exit 1 for a failed gate."""
    print(f"invalid chain report: {message}", file=sys.stderr)
    sys.exit(2)


try:
    with open(sys.argv[1]) as handle:
        report = json.load(handle)
except (OSError, ValueError) as error:
    invalid(f"cannot read {sys.argv[1]!r} as JSON: {error}")
if not isinstance(report, dict):
    invalid(f"top level is {type(report).__name__}, expected an object")
# first_drift_family is added to the payload by build_report; it is not a dataclass field.
missing = ({f.name for f in dataclasses.fields(ChainGateReport)} | {"first_drift_family"}) - set(report)
if missing:
    invalid(f"lacks report fields: {sorted(missing)}")
if not isinstance(report["cells"], dict):
    invalid(f"cells is {type(report['cells']).__name__}, expected an object")
for key, expected in (("schema_version", SCHEMA_VERSION), ("compatible_schema", COMPATIBLE_SCHEMA)):
    if report.get(key) != expected:
        invalid(f"{key} is {report.get(key)!r}, expected {expected!r}")
print(f"passed={report['passed']} first_drift={report['first_drift']} "
      f"family={report['first_drift_family']} cells={len(report['cells'])}")
sys.exit(0 if report["passed"] else 1)
PY
fi
