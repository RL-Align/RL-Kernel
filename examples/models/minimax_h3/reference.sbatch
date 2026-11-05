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
checkpoint=$3
result_root=$4
provider_source=$5
extra_dependencies=$6
cd "$source_dir"
[[ -z $(git status --porcelain) ]]
result_dir="$result_root/job-$SLURM_JOB_ID-attempt-${SLURM_RESTART_COUNT:-0}"
mkdir "$result_dir"
export PYTHONPATH="$source_dir:$provider_source/src:$extra_dependencies"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm.txt"
nvidia-smi -q > "$result_dir/device.txt"
git rev-parse HEAD > "$result_dir/commit.txt"
"$python_bin" -m pip freeze > "$result_dir/dependencies.txt"
"$python_bin" -m pip list --path "$extra_dependencies" --format=json > "$result_dir/extra-dependencies.json"
for mode in fp32-reference bf16-provider; do
  "$python_bin" -m tools.validation.models.h3_full_model \
    --checkpoint "$checkpoint" --manifest rl_engine/validation/models/h3_full_manifest.json \
    --mode "$mode" --output "$result_dir/$mode" > "$result_dir/$mode.log" 2>&1
done
"$python_bin" -m tools.validation.models.h3_full_compare \
  --reference "$result_dir/fp32-reference" --other "$result_dir/bf16-provider" \
  --output "$result_dir/full-model-thresholds.json" > "$result_dir/calibration.log" 2>&1
# Candidate jobs are submitted only after these immutable reference artifacts
# are reviewed and the numerical thresholds have been committed.
