#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# WS1 C3-C11 gate for the Ascend BF16 profile (#266, ascend_bf16).
#
# Runs on an Ascend host (Atlas A2 / 910B) with CANN and torch_npu. It is the
# NPU twin of ci/run_ws1_gtest.sh + ci/run_ws1_chain_gate.sh: same contract,
# same harnesses, same fail-closed rules. Nothing here may fall back to CPU or
# to another vendor's kernels - a required profile that cannot run is red.
#
# Required:
#   WS1_WEIGHTS_PATH (or QWEN3_8B)  pinned Qwen3-8B Dense safetensors snapshot
# Optional:
#   PY                              interpreter (default python3)
#   WS1_SKIP_BUILD=1                reuse an already-built rl_engine._C_npu

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
export RL_KERNEL_REQUIRE_EXT="${RL_KERNEL_REQUIRE_EXT:-1}"
WEIGHTS_PATH="${WS1_WEIGHTS_PATH:-${QWEN3_8B:-}}"

echo "[ws1-ascend] interpreter=$PY"

if [ "${WS1_SKIP_BUILD:-0}" != "1" ]; then
  echo "[ws1-ascend] building the Ascend C extension"
  KERNEL_ALIGN_FORCE_ASCEND=1 "$PY" -m pip install -e . --no-build-isolation --no-deps
fi

# Fail before any gate if the NPU or the compiled kernels are missing, so a
# later red cell is never confused with an environment problem.
"$PY" - <<'PY'
import sys

from rl_engine.kernels.gtest.accelerator import describe, npu_available, resolve_device

if not npu_available():
    sys.exit("[ws1-ascend] FATAL: torch_npu reports no available NPU")
info = describe(resolve_device(None, profile="ascend_bf16"))
print(f"[ws1-ascend] device={info.device} name={info.name} soc={info.arch_key}")

from rl_engine import _C_npu  # noqa: E402

required = (
    "rmsnorm_ascend",
    "rope_apply_ascend",
    "deterministic_attention_ascend",
    "embedding_ascend",
    "lm_head_ascend",
    "fused_logp_ascend",
    "batch_invariant_logp_ascend",
    "swiglu_forward",
    "silu_forward",
    "det_gemm_ascend_fwd",
    "det_gemm_rowwise_ascend_fwd_fp32",
)
missing = [name for name in required if not hasattr(_C_npu, name)]
if missing:
    sys.exit(f"[ws1-ascend] FATAL: _C_npu is missing {missing}; rebuild the extension")
print(f"[ws1-ascend] all {len(required)} required Ascend entry points are linked")
PY

echo "[ws1-ascend] CPU-side contract, workload and wiring tests"
"$PY" -m pytest -q \
  tests/test_tolerance_contract.py \
  tests/test_ws1_workload.py \
  tests/test_four_judgment_matrix.py \
  tests/test_elementwise_inventory.py \
  tests/test_ws1_ascend_closeout.py

echo "[ws1-ascend] Ascend operator tests"
"$PY" -m pytest -q \
  tests/test_det_gemm_ascend.py \
  tests/test_silu_ascend.py

echo "[ws1-ascend] C2 runtime candidate evidence"
"$PY" scripts/ws1_candidate_evidence.py \
  --profile ascend_bf16 --all --check-grad --emit-json /tmp/ws1-c2-ascend.json
"$PY" - /tmp/ws1-c2-ascend.json <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if not payload.get("passed"):
    failed = [c["case_id"] for c in payload["cases"] if c["runtime_status"] != "passed"]
    raise SystemExit(f"C2 Ascend runtime evidence failed: {failed}")
print(f"[ws1-ascend] C2 evidence passed for {len(payload['cases'])} pinned cases")
PY

echo "[ws1-ascend] C3/C4 smoke (silu)"
"$PY" scripts/check_forward_invariance.py \
  --op silu --candidate ascend --backend-profile ascend_bf16
"$PY" scripts/check_gradient_invariance.py \
  --op silu --candidate ascend --backend-profile ascend_bf16

echo "[ws1-ascend] C6 direct decode-prefill"
"$PY" scripts/check_decode_prefill.py --backend-profile ascend_bf16
echo "[ws1-ascend] C7 stateful KV + generate-rescore"
"$PY" scripts/check_stateful_kv.py --backend-profile ascend_bf16

C8_OUT="${WS1_C8_JSON:-${TMPDIR:-/tmp}/ws1-c8-ascend.json}"
export WS1_C8_EVIDENCE_PATH="$C8_OUT"
echo "[ws1-ascend] C8 four-judgment sweep -> $C8_OUT"
"$PY" scripts/sweep_ws1_four_judgments.py \
  --execute --profile ascend_bf16 --json > "$C8_OUT"
"$PY" - "$C8_OUT" <<'PY'
import json
import subprocess
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
git_meta = payload.get("git") or {}
expected = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if git_meta.get("commit") != expected or git_meta.get("dirty"):
    raise SystemExit(f"C8 is not from the clean current commit: {git_meta}")
counts = payload.get("counts") or {}
if int(counts.get("red", 0)):
    raise SystemExit(f"C8 contains red rows: {counts}")
if int(counts.get("green", 0)) == 0:
    raise SystemExit("C8 artifact has no green cells")
for cell in payload.get("cells") or []:
    if cell.get("op_name") == "pack" or cell.get("status") != "green":
        continue
    if not cell.get("judgment", "").endswith("invariance"):
        continue
    if not cell.get("actual_backend_id") or not cell.get("actual_kernel_config_id"):
        raise SystemExit(
            f"invariance cell missing provenance: {cell.get('profile')} {cell.get('op_name')}"
        )
print(f"[ws1-ascend] C8 passed counts={counts}")
PY

if [ -z "$WEIGHTS_PATH" ]; then
  echo "[ws1-ascend] FATAL: set WS1_WEIGHTS_PATH or QWEN3_8B for the C10/C11 full-model gate"
  exit 2
fi

echo "[ws1-ascend] C10/C11 full Qwen3-8B Dense model gate"
WS1_PROFILES="ascend_bf16" WS1_C8_JSON="$C8_OUT" bash ci/run_ws1_chain_gate.sh

echo "[ws1-ascend] ascend_bf16 passed every required WS1 gate"
