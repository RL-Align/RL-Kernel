#!/usr/bin/env bash
# Site options (partition, account, job name) are given at submission; see README.md.
# Full-resolution candidate generation, one seed per array task (--array=0-2). Preemption
# requeues the task; the generation continues from its saved steps (--resume).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --requeue
set -euo pipefail
source_dir=$1
python_bin=$2
checkpoint=$3
gate_job=$4        # job id of the candidate.sbatch run whose gates passed for this commit
result_root=$5
reference_image=$6
provider_source=$7
extra_dependencies=$8
cd "$source_dir"
[[ -z $(git status --porcelain) ]]
commit=$(git rev-parse HEAD)
# Only a passing numerical comparison of the same model code permits full generation: the gates
# passed at a commit whose model paths (h3_generation_resume.MODEL_PATHS) are byte-identical to
# this checkout's. The same commit qualifies trivially.
gate_dir=
for attempt in "$result_root"/job-"$gate_job"-attempt-*; do
  if [[ -f $attempt/commit.txt ]] && "$python_bin" - "$attempt/commit.txt" "$attempt/full-model-acceptance.json" \
    2> /dev/null <<'PY'
import json
import sys

from rl_engine.validation.models.h3_generation_resume import model_trees

gate_commit = open(sys.argv[1]).read().strip()
assert json.load(open(sys.argv[2]))["status"] == "passed"
assert model_trees(gate_commit) == model_trees("HEAD")
PY
  then
    gate_dir=$attempt
  fi
done
[[ -n $gate_dir ]] || { echo "no passing numerical gate for these model paths in job $gate_job" >&2; exit 1; }
seeds=($("$python_bin" -c 'import json; print(*json.load(open("examples/models/minimax_h3/generation-contract.json"))["seeds"])'))
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
export PYTHONPATH="$source_dir:$provider_source/src:$extra_dependencies"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false KERNEL_ALIGN_USE_FAST_MATH=0
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
out="$result_root/generation-${commit:0:8}"
mkdir -p "$out"
"$python_bin" -m tools.validation.models.h3_full_generation --checkpoint "$checkpoint" \
  --reference-image "$reference_image" --candidate --seeds "$seed" --resume --output "$out" \
  >> "$out/seed-$seed.log" 2>&1
