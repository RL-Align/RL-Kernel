#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Qwen3-Next chain-report adapter checks; with a report path, also validate that report.
#   bash ci/scripts/run_qwen3_next_chain_gate.sh [chain-report.json]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m pytest -q tests/models/qwen3_next/test_qwen3_next_chain_report.py
if [ "$#" -ge 1 ]; then
  "$PY" - "$1" <<'PY'
import dataclasses, json, sys

from rl_engine.validation.models.chain_gate import ChainGateReport
from rl_engine.validation.models.qwen3_next.qwen3_next_chain_report import COMPATIBLE_SCHEMA, SCHEMA_VERSION

report = json.load(open(sys.argv[1]))
missing = {f.name for f in dataclasses.fields(ChainGateReport)} - set(report)
assert not missing, f"chain report lacks ChainGateReport fields: {sorted(missing)}"
assert report["schema_version"] == SCHEMA_VERSION, report["schema_version"]
assert report["compatible_schema"] == COMPATIBLE_SCHEMA, report["compatible_schema"]
print(f"passed={report['passed']} first_drift={report['first_drift']} "
      f"family={report['first_drift_family']} cells={len(report['cells'])}")
sys.exit(0 if report["passed"] else 1)
PY
fi
