#!/usr/bin/env bash
#SBATCH --job-name=qwen3-ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=03:00:00
set -euo pipefail
source_dir=$1
python_bin=$2
checkpoint=$3
result_root=$4
cd "$source_dir"
[[ -z $(git status --porcelain) ]]
run_id=${SLURM_JOB_ID:+job-$SLURM_JOB_ID-attempt-${SLURM_RESTART_COUNT:-0}}
result_dir="$result_root/${run_id:-run-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir "$result_dir"
export PYTHONPATH="$source_dir"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ -n ${SLURM_JOB_ID:-} ]]; then scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm.txt"; fi
nvidia-smi -q > "$result_dir/device.txt"
git rev-parse HEAD > "$result_dir/commit.txt"
"$python_bin" -m pip freeze > "$result_dir/dependencies.txt"
"$python_bin" tools/validation/models/qwen3_ppo/calibrate_qwen3.py --checkpoint "$checkpoint" \
  --manifest tools/validation/models/qwen3_ppo/qwen3-8b-manifest.json --output "$result_dir/reference" \
  > "$result_dir/reference.log" 2>&1
