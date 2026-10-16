#!/usr/bin/env bash
# Site options (partition, account, job name) are given at submission; see README.md.
# Resume check on the real pipeline at preflight size (128x128, 6 sampling steps): one run
# without interruption, one preempted twice (after 2 and after 4 transformer calls) and
# resumed. Every step file, the initial inputs and the decoded media must be byte-equal.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=04:00:00
set -euo pipefail
source_dir=$1
python_bin=$2
checkpoint=$3
reference_image=$4
provider_source=$5
extra_dependencies=$6
result_root=$7
cd "$source_dir"
[[ -z $(git status --porcelain) ]]
export PYTHONPATH="$source_dir:$provider_source/src:$extra_dependencies"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false KERNEL_ALIGN_USE_FAST_MATH=0
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
out="$result_root/resume-check-$(git rev-parse --short=8 HEAD)"
mkdir "$out"
gen() {
  "$python_bin" -m tools.validation.models.h3_full_generation --checkpoint "$checkpoint" \
    --reference-image "$reference_image" --candidate --preflight --preflight-steps 6 "$@"
}
gen --output "$out/whole" > "$out/whole.log" 2>&1
gen --output "$out/cut" --stop-after-step 2 > "$out/cut-1.log" 2>&1 || true
gen --output "$out/cut" --resume --stop-after-step 4 > "$out/cut-2.log" 2>&1 || true
gen --output "$out/cut" --resume > "$out/cut-3.log" 2>&1
"$python_bin" - "$out" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
whole, cut = root / "whole" / "42", root / "cut" / "42"
names = sorted(
    p.name
    for p in whole.iterdir()
    if p.name.startswith("step-") and p.suffix == ".pt"
    or p.name in ("initial.pt", "video.npy", "audio.npy")
)
differing = [n for n in names if (whole / n).read_bytes() != (cut / n).read_bytes()]
resume = json.loads((cut / "report.json").read_text())["resume"]
result = {
    "compared_files": len(names),
    "steps": json.loads((whole / "report.json").read_text())["steps"],
    "differing": differing,
    "resume": resume,
    "failure_markers": sorted(p.name for p in (root / "cut").glob("failure*.json")),
}
(root / "resume-check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
assert not differing, differing
assert resume["replayed_steps"] > 0 and resume["recomputed_and_verified_step"] is not None
PY
