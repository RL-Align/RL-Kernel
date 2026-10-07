#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# WS1 single-op gtest + C8 four-judgment GPU gate.
# Assumes an editable install and a CUDA device. Fails closed on red cells
# and on silent fallback (C3/C4 already reject those).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
# Keep the artifact outside the repo so `_git_identity()` is not dirtied by
# the file we are in the process of writing.
OUT="${WS1_C8_JSON:-${TMPDIR:-/tmp}/ws1-c8-ci.json}"
export RL_KERNEL_REQUIRE_EXT="${RL_KERNEL_REQUIRE_EXT:-1}"

echo "[ws1-gtest] interpreter=$PY out=$OUT"

"$PY" -m pytest -q \
  tests/test_ws1_gtest_gpu.py \
  tests/test_qwen3_next_norm.py \
  tests/test_triton_batch_invariant_attention.py \
  tests/test_four_judgment_matrix.py \
  tests/test_ws1_candidate_evidence.py \
  tests/test_op_checks.py \
  tests/test_elementwise_inventory.py

echo "[ws1-gtest] C3/C4 CUDA + Triton smoke (silu)"
"$PY" scripts/check_forward_invariance.py \
  --op silu --candidate cuda --backend-profile cuda_bf16
"$PY" scripts/check_gradient_invariance.py \
  --op silu --candidate cuda --backend-profile cuda_bf16
"$PY" scripts/check_forward_invariance.py \
  --op silu --candidate triton --backend-profile triton_cuda_bf16
"$PY" scripts/check_gradient_invariance.py \
  --op silu --candidate triton --backend-profile triton_cuda_bf16

HOPPER=0
if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0]==9 else 1)"; then
  HOPPER=1
fi

echo "[ws1-gtest] C8 --execute hopper=$HOPPER"
if [ "$HOPPER" = 1 ]; then
  "$PY" scripts/sweep_ws1_four_judgments.py --execute --json > "$OUT"
else
  "$PY" scripts/sweep_ws1_four_judgments.py --execute --json --allow-pending-hopper > "$OUT"
fi

"$PY" - "$OUT" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
counts = payload.get("counts") or {}
red = int(counts.get("red", 0))
print(f"[ws1-gtest] C8 counts={counts} source={payload.get('git')}")
if red:
    raise SystemExit(f"C8 has {red} red cells")
cells = payload.get("cells") or []
if not cells:
    raise SystemExit("C8 artifact contains no cells")
if int(counts.get("green", 0)) == 0:
    raise SystemExit("C8 artifact has no green cells")
required = [c for c in cells if c.get("op_name") != "pack" and c.get("status") == "green"]
if not required:
    raise SystemExit("C8 artifact has no green required cells")
for cell in required:
    if not cell.get("judgment", "").endswith("invariance"):
        continue
    if not cell.get("actual_backend_id") or not cell.get("actual_kernel_config_id"):
        raise SystemExit(
            f"invariance cell missing provenance: {cell.get('profile')} {cell.get('op_name')}"
        )
print("[ws1-gtest] C8 gate passed")
PY
