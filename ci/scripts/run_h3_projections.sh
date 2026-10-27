#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
mode="${1:-smoke}"
case "$mode" in smoke|full|replay|tasks|tasks-replay) ;; *) echo 'Expected smoke, full, replay, tasks or tasks-replay' >&2; exit 2 ;; esac
: "${H3_EVIDENCE_DIR:?Set H3_EVIDENCE_DIR to the local evidence directory}"
mkdir -p "$H3_EVIDENCE_DIR"
export TRITON_CACHE_DIR="$H3_EVIDENCE_DIR/cache-$mode"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.version.hip is not None:
    raise SystemExit('An NVIDIA SM90 GPU is required')
if torch.cuda.get_device_capability() != (9, 0):
    raise SystemExit('This profile requires SM90')
PY
if [[ "$mode" == smoke ]]; then
    RL_KERNEL_REQUIRE_FP32_GPU=1 python -m pytest -q tests/ops/gemm/test_det_gemm_fp32.py
    python -m pytest -q tests/ops/gemm/test_h3_projections.py \
        tests/validation/operators/test_h3_projection_*.py
fi
args=(--output "$H3_EVIDENCE_DIR/$mode.json")
if [[ "$mode" == smoke ]]; then args+=(--rows 1,33,129 --families random --repeats 3); fi
if [[ "$mode" == replay ]]; then args+=(--rows 1,33,129,8192,32768 --families random); fi
if [[ "$mode" == tasks* ]]; then args+=(--rows 32768,131072 --task-layouts --families random); fi
for kind in h3_ffn_gate_up_gemm h3_attention_o_gemm; do
    fixture="$H3_EVIDENCE_DIR/$kind.pt"
    if [[ ! -f "$fixture" ]]; then
        python tools/weights/prepare_h3_projection_fixture.py --operator "$kind" --output "$fixture"
    fi
    args+=(--fixture "$fixture")
done
python tools/validation/operators/validate_h3_projections.py "${args[@]}"
if [[ "$mode" == replay ]]; then
    python tools/validation/operators/compare_h3_projection_replays.py \
        "$H3_EVIDENCE_DIR/full.json" "$H3_EVIDENCE_DIR/replay.json" \
        --output "$H3_EVIDENCE_DIR/replay-comparison.json"
fi

if [[ "$mode" == tasks-replay ]]; then
    python tools/validation/operators/compare_h3_projection_replays.py \
        "$H3_EVIDENCE_DIR/tasks.json" "$H3_EVIDENCE_DIR/tasks-replay.json" \
        --output "$H3_EVIDENCE_DIR/tasks-replay-comparison.json"
fi
