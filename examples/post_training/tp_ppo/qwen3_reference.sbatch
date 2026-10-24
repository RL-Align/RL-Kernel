#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --partition=overflow
#SBATCH --account=wrd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
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
result_dir="$result_root/job-$SLURM_JOB_ID-attempt-${SLURM_RESTART_COUNT:-0}"
mkdir "$result_dir"
export PYTHONPATH="$source_dir"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm.txt"
nvidia-smi -q > "$result_dir/device.txt"
git rev-parse HEAD > "$result_dir/commit.txt"
"$python_bin" -m pip freeze > "$result_dir/dependencies.txt"
"$python_bin" examples/post_training/tp_ppo/calibrate_qwen3.py --checkpoint "$checkpoint" \
  --manifest examples/post_training/tp_ppo/qwen3-8b-manifest.json --output "$result_dir/reference" \
  > "$result_dir/reference.log" 2>&1
