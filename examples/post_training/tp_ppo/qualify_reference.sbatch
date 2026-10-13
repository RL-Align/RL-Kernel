#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --partition=overflow
#SBATCH --account=wrd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=24:00:00
set -euo pipefail
source_dir=$1
python_bin=$2
vllm_python=$3
checkpoint=$4
calibration=$5
result_root=$6
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
"$vllm_python" -m pip freeze > "$result_dir/vllm-dependencies.txt"
"$python_bin" -m examples.post_training.tp_ppo.capture_qwen3 --checkpoint "$checkpoint" \
  --vllm-python "$vllm_python" --calibration "$calibration" --output "$result_dir/rollout" \
  > "$result_dir/capture.log" 2>&1
"$python_bin" -m examples.post_training.tp_ppo.hf_ppo_reference --checkpoint "$checkpoint" \
  --rollout "$result_dir/rollout" --score-only --output "$result_dir/targets" > "$result_dir/targets.log" 2>&1
for role in actor critic; do
  for precision in fp32 bf16; do
    for repeat in 0 1; do
      run="$result_dir/$role-$precision-$repeat"
      "$python_bin" -m examples.post_training.tp_ppo.hf_ppo_reference --checkpoint "$checkpoint" \
        --rollout "$result_dir/rollout" --targets "$result_dir/targets" --role "$role" \
        --precision "$precision" --output "$run" > "$run.log" 2>&1
    done
    "$python_bin" -m examples.post_training.tp_ppo.compare_fixed --reference "$result_dir/$role-$precision-0" \
      --other "$result_dir/$role-$precision-1" --output "$result_dir/$role-$precision-repeat-bounds.json"
  done
  "$python_bin" -m examples.post_training.tp_ppo.compare_fixed --reference "$result_dir/$role-fp32-0" \
    --other "$result_dir/$role-bf16-0" --output "$result_dir/$role-cross-precision-bounds.json"
done
# Bounds and evidence manifests must be committed before any formal candidate.

"$python_bin" -m examples.post_training.tp_ppo.freeze_fixed_reference --reference-root "$result_dir" --output "$result_dir/acceptance.json"
