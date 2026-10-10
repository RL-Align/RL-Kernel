#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${TIMESTEP_RUN_ROOT:?Set TIMESTEP_RUN_ROOT to an isolated output directory for this task}"
: "${CUDA_HOME:=/usr/local/cuda}"
export CUDA_HOME PYTHONDONTWRITEBYTECODE=1
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$TIMESTEP_RUN_ROOT"
run_dir=$(mktemp -d "$TIMESTEP_RUN_ROOT/validation.XXXXXX")
export TORCH_EXTENSIONS_DIR="$run_dir/torch-cache"
export TRITON_CACHE_DIR="$run_dir/triton-cache"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
export TIMESTEP_TARGET="${TIMESTEP_TARGET:-a100}"
python - <<'PYENV' > "$run_dir/environment.txt"
import os
import torch
name = torch.cuda.get_device_name()
capability = torch.cuda.get_device_capability()
target = os.environ["TIMESTEP_TARGET"]
print("torch:", torch.__version__, "GPU:", name, "capability:", capability)
print("target:", target, "arch list:", os.environ.get("TORCH_CUDA_ARCH_LIST"))
print("JIT only:", os.environ.get("TIMESTEP_CUDA_JIT_ONLY", "0"))
if target == "a100":
    assert "A100" in name, f"Expected A100, got {name}"
elif target == "sm90":
    assert capability == (9, 0), f"Expected sm90, got {capability}"
else:
    raise ValueError(f"Unsupported TIMESTEP_TARGET: {target}")
PYENV
nvidia-smi >> "$run_dir/environment.txt"
nvcc --version >> "$run_dir/environment.txt"
python -m pip freeze > "$run_dir/packages.txt"
python -m pytest tests/test_timestep_official.py tests/test_tolerance_contract.py tests/test_operator_inputs.py -q > "$run_dir/pytest.log" 2>&1
for backend in cuda triton; do
    for dtype in fp32 bf16; do
        python scripts/check_operator.py --op timestep_embed_mlp --candidate "$backend" \
            --device cuda --dtype "$dtype" --batch 3 --seq 1 --seed 386 --check-grad --json \
            > "$run_dir/gtest-$backend-$dtype.json" 2> "$run_dir/gtest-$backend-$dtype.stderr"
    done
done
python scripts/validate_timestep_official.py --trace --benchmark --output "$run_dir/matrix.json" \
    > "$run_dir/matrix.log" 2>&1
printf 'Evidence: %s\n' "$run_dir"
