#!/usr/bin/env bash
#SBATCH --job-name=qwen3-ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=24:00:00
set -euo pipefail
source_dir=$1
python_bin=$2
vllm_python=$3
framework_overlay=$4
checkpoint=$5
reference_root=$6
acceptance_manifest=$7
result_root=$8
cd "$source_dir"
source tools/validation/models/qwen3_ppo/fixed_matrix.sh
[[ -z $(git status --porcelain) ]]
run_id=${SLURM_JOB_ID:+job-$SLURM_JOB_ID-attempt-${SLURM_RESTART_COUNT:-0}}
result_dir="$result_root/${run_id:-run-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir "$result_dir"
export PYTHONPATH="$source_dir:$framework_overlay"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ -n ${SLURM_JOB_ID:-} ]]; then scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm.txt"; fi
nvidia-smi -q > "$result_dir/device.txt"
git rev-parse HEAD > "$result_dir/commit.txt"
"$python_bin" -m pip freeze > "$result_dir/dependencies.txt"
"$python_bin" -m pip list --path "$framework_overlay" --format=json > "$result_dir/framework-dependencies.json"
"$python_bin" -c 'from pathlib import Path; import sys; from tools.validation.models.qwen3_ppo.fixed_contract import verify_reference; verify_reference(Path(sys.argv[1]),Path(sys.argv[2]))' "$reference_root" "$acceptance_manifest"
for engine in megatron deepspeed; do
  for precision in fp32 bf16; do
    for role in actor critic; do
      sizes=$(fixed_tp_sizes "$engine" "$precision" "$role")
      for tp in $sizes; do
        run="$result_dir/$engine-$precision-$role-tp$tp"
        "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$tp" \
          -m tests.distributed.tp.check_qwen3_fixed --framework "$engine" --precision "$precision" \
          --role "$role" --checkpoint "$checkpoint" --rollout "$reference_root/rollout" \
          --targets "$reference_root/targets" --acceptance-manifest "$acceptance_manifest" \
          --output "$run" > "$run.log" 2>&1
        bounds="$reference_root/$role-fp32-repeat-bounds.json"
        if [[ "$precision" == bf16 ]]; then bounds="$reference_root/$role-cross-precision-bounds.json"; fi
        "$python_bin" -m tools.validation.models.qwen3_ppo.compare_fixed --reference "$reference_root/$role-fp32-0" \
          --other "$run" --thresholds "$bounds" --acceptance-manifest "$acceptance_manifest" \
          --reference-root "$reference_root" --checkpoint "$checkpoint" --output "$run/comparison.json"
      done
      if [[ "$sizes" != "2 4" ]]; then continue; fi
      "$python_bin" -m tools.validation.models.qwen3_ppo.compare_fixed --tp-comparison \
        --reference "$result_dir/$engine-$precision-$role-tp2" --other "$result_dir/$engine-$precision-$role-tp4" \
        --thresholds "$reference_root/$role-$precision-repeat-bounds.json" --acceptance-manifest "$acceptance_manifest" \
        --reference-root "$reference_root" --checkpoint "$checkpoint" \
        --output "$result_dir/$engine-$precision-$role-tp-comparison.json"
    done
  done
  # The pretrained two-round vLLM topologies (actor, critic and vLLM on dedicated GPUs) are not
  # run here: at TP2 each trainer needs about 80 GB, so they need more than four 80-94 GB GPUs.
  "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m tests.distributed.tp.check_tp_ppo_rollout --framework "$engine" --actor-tp 2 --critic-tp 1 \
    --vllm-python "$vllm_python" --gpu-layout dedicated --output "$result_dir/$engine-tiny" > "$result_dir/$engine-tiny.log" 2>&1
done
