#!/usr/bin/env bash
# Site options (partition, account, job name) are given at submission; see README.md.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=24:00:00
set -euo pipefail
source_dir=$1
python_bin=$2
failed_result=$3
result_root=$4
cd "$source_dir"
[[ -z $(git status --porcelain) ]]
result_dir="$result_root/job-$SLURM_JOB_ID-attempt-${SLURM_RESTART_COUNT:-0}"
mkdir "$result_dir"
export PYTHONPATH="$source_dir" RL_KERNEL_H3_WEIGHTS="$failed_result/preflight/weights"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 USE_TF=0 USE_FLAX=0
scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm.txt"
nvidia-smi -q > "$result_dir/device.txt"
git rev-parse HEAD > "$result_dir/commit.txt"
"$python_bin" -m pip freeze > "$result_dir/dependencies.txt"
# Preserve the exact native binary from the failed attempt; record and recheck
# its identity instead of changing both the kernel and diagnostic at once.
"$python_bin" - "$failed_result" "$result_dir" <<'PY'
import json
import shutil
import sys
from pathlib import Path
from tools.validation.models.h3_full_model import digest
failed, output = map(Path, sys.argv[1:])
identity = json.loads((failed / "preflight/identity.json").read_text())
source = Path(identity["extension"])
assert digest(source) == identity["extension_sha256"]
shutil.copy2(source, Path("rl_engine") / source.name)
(output / "native-origin.json").write_text(json.dumps(identity, indent=2) + "\n")
(output / "retry-of.txt").write_text(str(failed) + "\n")
PY
"$python_bin" -m tools.validation.models.h3_block_diagnostic --failed-job "$failed_result" --causal-sweep \
  --output "$result_dir/diagnostic" > "$result_dir/diagnostic.log" 2>&1
