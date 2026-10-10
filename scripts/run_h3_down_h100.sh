#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mode="${1:-smoke}"
case "$mode" in smoke|full) ;; *) echo 'Usage: bash scripts/run_h3_down_h100.sh [smoke|full]' >&2; exit 2 ;; esac
result_dir="${H3_RESULT_DIR:-$repo_dir/artifacts/h3-down/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$result_dir"
export KERNEL_ALIGN_USE_FAST_MATH=0
export KERNEL_ALIGN_DET_GEMM_SM90=1
export KERNEL_ALIGN_FORCE_SM90=0
export RL_KERNEL_REQUIRE_EXT=1
export RL_KERNEL_DET_GEMM_BACKEND=sm90
export MAX_JOBS="${MAX_JOBS:-8}"
export PYTHONUNBUFFERED=1
# SSH non-login shells in Runpod's development images may omit the toolkit PATH.
if ! command -v nvcc >/dev/null 2>&1 && [[ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export PATH="$CUDA_HOME/bin:$PATH"
fi
# BuildExtension's explicit 90a flags cover the TMA path; do not add plain compute_90 PTX.
unset TORCH_CUDA_ARCH_LIST

python - <<'PY'
from pathlib import Path
import os
import sysconfig
import torch
if torch.version.hip is not None or not torch.cuda.is_available():
    raise SystemExit('An NVIDIA H100 GPU and a CUDA PyTorch build are required')
if torch.cuda.get_device_capability() != (9, 0):
    raise SystemExit('This profile requires SM90')
if torch.version.cuda != '12.4':
    raise SystemExit('This initial runbook pins a CUDA 12.4 PyTorch build')
print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())
if os.environ.get('H3_BUILD_NATIVE', '0') == '1' and not (
    Path(sysconfig.get_path('include')) / 'Python.h'
).is_file():
    raise SystemExit('Python development headers are required; install the matching pythonX.Y-dev')
PY
nvcc --version | tee "$result_dir/nvcc.txt"
nvcc_version="$(< "$result_dir/nvcc.txt")"
if [[ "$nvcc_version" != *"release 12.4"* ]]; then
    echo 'This initial runbook requires nvcc 12.4' >&2
    exit 2
fi
nvidia-smi -q > "$result_dir/nvidia-smi.txt"
python -m pip freeze > "$result_dir/pip-freeze.txt"
git status --short > "$result_dir/git-status.txt"
git diff HEAD > "$result_dir/source.patch"
python -m pytest -q tests/test_h3_ffn_down.py -k 'not on_gpu' | tee "$result_dir/cpu-tests.txt"
if [[ "${H3_BUILD_NATIVE:-0}" == 1 ]]; then
    python -m pip install --no-deps --no-build-isolation -v -e . 2>&1 | tee "$result_dir/build.log"
fi
set +e
python -m pytest -q tests/test_h3_ffn_down.py | tee "$result_dir/gpu-tests.txt"
gpu_test_status=${PIPESTATUS[0]}
set -e

args=(--output "$result_dir/report.json")
if [[ -f "$result_dir/build.log" ]]; then args+=(--build-log "$result_dir/build.log"); fi
if [[ -n "${H3_BACKEND:-}" ]]; then args+=(--backend "$H3_BACKEND"); fi
if [[ "$mode" == smoke ]]; then
    args+=(--rows 1,33,129 --families random --repeats 3)
fi
if [[ -n "${H3_FIXTURE:-}" ]]; then args+=(--fixture "$H3_FIXTURE"); fi
if [[ "${H3_COMPARE_VLLM:-0}" == 1 ]]; then args+=(--compare-vllm); fi
set +e
python scripts/validate_h3_ffn_down.py "${args[@]}" 2>&1 | tee "$result_dir/validation.log"
validation_status=${PIPESTATUS[0]}
set -e
python scripts/plot_h3_down_report.py "$result_dir/report.json" --output-dir "$result_dir"
echo "Evidence directory: $result_dir"
if [[ "$gpu_test_status" -ne 0 ]]; then exit "$gpu_test_status"; fi
exit "$validation_status"
