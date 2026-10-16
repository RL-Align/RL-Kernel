#!/usr/bin/env bash
# Site options (partition, account, job name) are given at submission; see README.md.
# Quality comparison of the three candidate seeds with the provider reference (CPU only).
# Submit after the generation array finished successfully (--dependency=afterok:<array id>).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
set -euo pipefail
source_dir=$1
python_bin=$2
reference_root=$3
result_root=$4
cd "$source_dir"
[[ -z $(git status --porcelain) ]]
commit=$(git rev-parse HEAD)
export PYTHONPATH="$source_dir"
"$python_bin" -m tools.validation.models.h3_generation_compare \
  --reference "$reference_root/generation-reference" \
  --candidate "$result_root/generation-${commit:0:8}" \
  --contract examples/models/minimax_h3/generation-contract.json \
  --reference-root "$reference_root" \
  --reference-manifest examples/models/minimax_h3/reference-manifest.json \
  --output "$result_root/generation-comparison-${commit:0:8}.json"
