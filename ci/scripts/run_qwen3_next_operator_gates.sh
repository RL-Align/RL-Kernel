#!/usr/bin/env bash
# Required operator acceptance and diagnostics use separate declared case sets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-python3}"
OUT="${QWEN3_NEXT_EVIDENCE_DIR:?Set QWEN3_NEXT_EVIDENCE_DIR to an artifact directory}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" tools/validation/models/qwen3_next_operator_gates.py --out "$OUT" \
  --suite "${QWEN3_NEXT_OPERATOR_SUITE:-all}"
