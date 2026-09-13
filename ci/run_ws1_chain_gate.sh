#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# WS1 C10/C11 full Qwen3-8B Dense model-level gate.
# Profiles come from WS1_PROFILES (default: the two CUDA-host profiles). One host
# has either a GPU or an NPU, so each vendor's job runs its own profiles here:
#   CUDA host    : WS1_PROFILES="cuda_bf16 triton_cuda_bf16"   (H20 / H100)
#   Ascend host  : WS1_PROFILES="ascend_bf16"                  (Atlas A2 / 910B)
# C11 closes only when every required profile has passed on its own hardware.
# Fails closed on skip, xfail, synthetic weights, or silent fallback.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
export RL_KERNEL_REQUIRE_EXT="${RL_KERNEL_REQUIRE_EXT:-1}"
WEIGHTS_PATH="${WS1_WEIGHTS_PATH:-${QWEN3_8B:-}}"
WS1_PROFILES="${WS1_PROFILES:-cuda_bf16 triton_cuda_bf16}"

if [ -z "$WEIGHTS_PATH" ]; then
  echo "[ws1-chain] FATAL: set WS1_WEIGHTS_PATH or QWEN3_8B to the pinned Qwen3-8B snapshot"
  exit 2
fi

echo "[ws1-chain] interpreter=$PY weights=$WEIGHTS_PATH profiles=$WS1_PROFILES"

"$PY" -m pytest -q \
  tests/test_kv_consistency.py \
  tests/test_ws1_qwen3_dense.py \
  tests/test_ws1_chain_integration.py

C8_OUT="${WS1_C8_JSON:-${TMPDIR:-/tmp}/ws1-c8-ci.json}"
export WS1_C8_EVIDENCE_PATH="$C8_OUT"
echo "[ws1-chain] C8 runtime evidence $C8_OUT"
C8_PROFILE_ARGS=()
for PROFILE in $WS1_PROFILES; do
  C8_PROFILE_ARGS+=(--profile "$PROFILE")
done
"$PY" scripts/sweep_ws1_four_judgments.py --execute "${C8_PROFILE_ARGS[@]}" --json > "$C8_OUT"
"$PY" - "$C8_OUT" <<'PY'
import json
import subprocess
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
git_meta = payload.get("git") or {}
expected = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if git_meta.get("commit") != expected or git_meta.get("dirty"):
    raise SystemExit(f"C8 is not from clean current commit: {git_meta}")
if int((payload.get("counts") or {}).get("red", 0)):
    raise SystemExit(f"C8 contains red rows: {payload.get('counts')}")
print(f"[ws1-chain] C8 passed source={git_meta}")
PY

for PROFILE in $WS1_PROFILES; do
  OUT="/tmp/ws1-c10-${PROFILE}.json"
  echo "[ws1-chain] C10/C11 $PROFILE"
  "$PY" scripts/ws1_chain_gate.py \
    --backend-profile "$PROFILE" \
    --model qwen3-8b-dense \
    --dtype bfloat16 \
    --seed 0 \
    --weights required \
    --weights-path "$WEIGHTS_PATH" \
    --json > "$OUT"
  "$PY" - "$OUT" "$PROFILE" <<'PY'
import json
import sys

path, profile = sys.argv[1], sys.argv[2]
payload = json.load(open(path, encoding="utf-8"))
manifest = json.load(
    open("rl_engine/testing/ws1_manifest.json", encoding="utf-8")
)
expected_weight_hash = manifest["model_identity"]["weight_snapshot"]["content_hash"]
if payload.get("schema_version") != "ws1-c10-c11-v5":
    raise SystemExit(f"{profile} artifact has an unsupported schema")
if payload.get("backend_profile") != profile:
    raise SystemExit(f"artifact profile mismatch for {profile}")
if payload.get("weight_hash") != expected_weight_hash:
    raise SystemExit(f"{profile} did not verify the pinned weight snapshot")
if payload.get("workload_seed") != manifest["seed"]:
    raise SystemExit(f"{profile} workload seed does not match the manifest")
if not payload.get("passed"):
    raise SystemExit(f"{profile} C10 gate failed first_drift={payload.get('first_drift')}")
if payload.get("weight_source", "").startswith("synthetic"):
    raise SystemExit(f"{profile} used synthetic weights; C11 forbids that")
if payload.get("git_sha") in {None, "", "unknown"}:
    raise SystemExit(f"{profile} has no commit SHA")
if payload.get("git_dirty"):
    raise SystemExit(f"{profile} was produced from a dirty worktree")
if not payload.get("backward_executed"):
    raise SystemExit(f"{profile} did not execute backward")
if not payload.get("train_infer_executed"):
    raise SystemExit(f"{profile} did not execute train/infer parity")
if payload.get("gradient_scope") != "all_required_trainable_parameters":
    raise SystemExit(f"{profile} has an unknown gradient scope")
if payload.get("all_parameter_gradients") is not True:
    raise SystemExit(f"{profile} must compare every required trainable parameter")
required_names = set(payload.get("required_grad_names") or [])
for name in (
    "embed_tokens.weight",
    "lm_head.weight",
    "norm.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "layers.35.mlp.down_proj.weight",
):
    if name not in required_names:
        raise SystemExit(f"{profile} missing required gradient {name}")
if len(required_names) != 3 + 36 * 11:
    raise SystemExit(f"{profile} required gradient count {len(required_names)} != 399")
if not payload.get("accuracy_executed"):
    raise SystemExit(f"{profile} did not execute FP32 forward_accuracy")
if not payload.get("gradient_accuracy_executed"):
    raise SystemExit(f"{profile} did not execute FP32 gradient_accuracy")
if payload.get("train_infer_bn") is None or not payload["train_infer_bn"].get("passed"):
    raise SystemExit(f"{profile} full-model BN decode/prefill parity failed")
decode_cases = {item.get("case_id") for item in payload.get("decode_prefill") or []}
for case_id in (
    "decode-b1-short",
    "decode-b1-long",
    "decode-bn-varlen",
    "decode-bn-padded-right",
    "decode-bn-padded-left",
    "decode-b1-primary-s3",
):
    if case_id not in decode_cases:
        raise SystemExit(f"{profile} missing full-model decode case {case_id}")
if not all(item.get("passed") for item in payload.get("decode_prefill") or []):
    raise SystemExit(f"{profile} full-model decode/prefill sweep failed")
if not payload.get("accuracy_aggregates") or not all(
    item.get("passed") for item in payload["accuracy_aggregates"]
):
    raise SystemExit(f"{profile} FP32 three-aggregate accuracy failed")
if not payload.get("accuracy") or not all(item.get("passed") for item in payload["accuracy"]):
    raise SystemExit(f"{profile} FP32 forward_accuracy failed")
if not payload.get("gradient_accuracy") or not all(
    item.get("passed") for item in payload["gradient_accuracy"]
):
    raise SystemExit(f"{profile} FP32 gradient_accuracy failed")
if not payload.get("gpu_name"):
    raise SystemExit(f"{profile} missing gpu_name")
if not payload.get("representative_case_ids"):
    raise SystemExit(f"{profile} missing representative_case_ids")
if not payload.get("c8_evidence_path"):
    raise SystemExit(f"{profile} missing c8_evidence_path")
c8 = json.load(open(payload["c8_evidence_path"], encoding="utf-8"))
c8_git = c8.get("git") or {}
if c8_git.get("commit") != payload.get("git_sha") or c8_git.get("dirty"):
    raise SystemExit(
        f"{profile} C8 evidence is not bound to the same clean commit: "
        f"c8={c8_git.get('commit')} c10={payload.get('git_sha')} "
        f"dirty={c8_git.get('dirty')}"
    )
if not any(
    tuple(item.get("config_pair", ())) == ("BN/packed", "fp32_reference")
    for item in payload.get("accuracy") or []
):
    raise SystemExit(f"{profile} missing packed FP32 forward accuracy")
if not any(
    tuple(item.get("config_pair", ())) == ("BN/packed", "fp32_reference")
    for item in payload.get("gradient_accuracy") or []
):
    raise SystemExit(f"{profile} missing packed FP32 gradient accuracy")
if not payload.get("workflow_url"):
    raise SystemExit(f"{profile} missing workflow_url")
bwd = payload.get("backward_runtime_observations") or {}
for kind in ("lm_head", "rms_norm", "det_gemm", "embedding"):
    event = bwd.get(kind) or {}
    if int(event.get("execution_count") or 0) <= 0:
        raise SystemExit(f"{profile} missing runtime backward record for {kind}")
    if not event.get("kernel_id"):
        raise SystemExit(f"{profile} backward {kind} missing kernel_id")
    family = {
        "cuda_bf16": "cuda",
        "triton_cuda_bf16": "triton",
        "ascend_bf16": "ascend",
    }[profile]
    if not event.get("kernel_ids"):
        raise SystemExit(f"{profile} backward {kind} missing kernel_ids")
    if not event.get("implementation_ids"):
        raise SystemExit(f"{profile} backward {kind} missing implementation_ids")
    if event.get("family") != family:
        raise SystemExit(
            f"{profile} backward {kind} family {event.get('family')!r} != {family!r}"
        )
if payload.get("first_drift") is not None:
    raise SystemExit(f"{profile} passed with a non-null first_drift")
observations = payload.get("runtime_backend_observations", {})
required_nodes = {
    "embedding", "rms_norm", "det_gemm", "qk_norm", "rope",
    "attention", "swiglu", "lm_head", "logprob",
}
if set(observations) != required_nodes:
    raise SystemExit(
        f"{profile} runtime observations mismatch: {sorted(observations)}"
    )
for node, observation in observations.items():
    if observation.get("execution_count", 0) <= 0:
        raise SystemExit(f"{profile} node {node} was not executed")
    if observation.get("expected_kernel_id") != observation.get("observed_kernel_id"):
        raise SystemExit(f"{profile} node {node} used an unexpected candidate")
    if observation.get("fallback_observed"):
        raise SystemExit(f"{profile} node {node} reported fallback")
print(f"[ws1-chain] {profile} passed first_drift={payload.get('first_drift')}")
PY
done

echo "[ws1-chain] profiles passed: $WS1_PROFILES"
