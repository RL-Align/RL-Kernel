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
reference_root=$4
result_root=$5
reference_image=$6
provider_source=$7
extra_dependencies=$8
cd "$source_dir"
[[ -z $(git status --porcelain) ]]
result_dir="$result_root/job-$SLURM_JOB_ID-attempt-${SLURM_RESTART_COUNT:-0}"
mkdir "$result_dir"
export PYTHONPATH="$source_dir:$provider_source/src:$extra_dependencies"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false KERNEL_ALIGN_USE_FAST_MATH=0
export CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_KERNEL_CACHE_PATH="$result_dir/torch-cache" TRITON_CACHE_DIR="$result_dir/triton-cache"
mkdir -p "$TORCH_KERNEL_CACHE_PATH" "$TRITON_CACHE_DIR"
scontrol show job "$SLURM_JOB_ID" > "$result_dir/slurm.txt"
nvidia-smi -q > "$result_dir/device.txt"
git rev-parse HEAD > "$result_dir/commit.txt"
"$python_bin" -m pip freeze > "$result_dir/dependencies.txt"
"$python_bin" -m pip list --path "$extra_dependencies" --format=json > "$result_dir/extra-dependencies.json"
"$python_bin" setup.py build_ext --inplace > "$result_dir/build.log" 2>&1
"$python_bin" -m tools.validation.models.h3_candidate_prepare --checkpoint "$checkpoint" \
  --reference-root "$reference_root" --output "$result_dir/preflight" > "$result_dir/preflight.log" 2>&1
export RL_KERNEL_H3_WEIGHTS="$result_dir/preflight/weights"
"$python_bin" -m pytest -q tests/models/minimax_h3/test_h3_one_block.py \
  --junitxml="$result_dir/block0.xml" > "$result_dir/block0.log" 2>&1
"$python_bin" -c 'import sys,xml.etree.ElementTree as E; r=E.parse(sys.argv[1]); assert all(int(s.get("skipped",0)) == 0 for s in r.iter("testsuite")), "block0 skipped"' "$result_dir/block0.xml"
"$python_bin" -m tools.validation.models.h3_full_model --checkpoint "$checkpoint" \
  --manifest rl_engine/validation/models/h3_full_manifest.json --mode candidate \
  --thresholds examples/models/minimax_h3/full-model-thresholds.json \
  --output "$result_dir/candidate" > "$result_dir/candidate.log" 2>&1
"$python_bin" -m tools.validation.models.h3_full_compare --reference "$reference_root/fp32-reference" \
  --other "$result_dir/candidate" --thresholds examples/models/minimax_h3/full-model-thresholds.json \
  --reference-root "$reference_root" --reference-manifest examples/models/minimax_h3/reference-manifest.json \
  --output "$result_dir/full-model-comparison.json" > "$result_dir/comparison.log" 2>&1
"$python_bin" -m tools.validation.models.h3_full_generation --checkpoint "$checkpoint" \
  --reference-image "$reference_image" --candidate --preflight --output "$result_dir/generation-preflight" \
  > "$result_dir/generation-preflight.log" 2>&1
"$python_bin" -m tools.validation.models.h3_full_generation --checkpoint "$checkpoint" \
  --reference-image "$reference_image" --candidate --output "$result_dir/generation-candidate" \
  > "$result_dir/generation-candidate.log" 2>&1
"$python_bin" -m tools.validation.models.h3_generation_compare --reference "$reference_root/generation-reference" \
  --candidate "$result_dir/generation-candidate" --contract examples/models/minimax_h3/generation-contract.json \
  --reference-root "$reference_root" --reference-manifest examples/models/minimax_h3/reference-manifest.json \
  --output "$result_dir/generation-comparison.json" > "$result_dir/generation-comparison.log" 2>&1
