#!/usr/bin/env bash
# Site options (partition, account, job name) are given at submission; see README.md.
# Resume check on the real pipeline at preflight size (128x128, 6 sampling steps, 5 transformer
# calls): one run without interruption, and two preempted runs that are resumed -- "cut"
# after 2, 4 and 5 calls (the last one after the final step, before decoding, so the final
# attempt has nothing left to compute) and "cutverify" after 2 calls with --verify-last-step.
# Every step file, the initial inputs and the decoded media must be byte-equal to "whole".
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
gen --output "$out/cut" --resume --stop-after-step 5 > "$out/cut-3.log" 2>&1 || true
gen --output "$out/cut" --resume > "$out/cut-4.log" 2>&1
gen --output "$out/cutverify" --stop-after-step 2 > "$out/cutverify-1.log" 2>&1 || true
gen --output "$out/cutverify" --resume --verify-last-step > "$out/cutverify-2.log" 2>&1
"$python_bin" - "$out" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
whole = root / "whole" / "42"
whole_report = json.loads((whole / "report.json").read_text())
names = sorted(
    p.name
    for p in whole.iterdir()
    if p.name.startswith("step-") and p.suffix == ".pt"
    or p.name in ("initial.pt", "video.npy", "audio.npy")
)
result = {"compared_files": len(names), "steps": whole_report["steps"], "runs": {}}
for name in ("cut", "cutverify"):
    directory = root / name / "42"
    report = json.loads((directory / "report.json").read_text())
    result["runs"][name] = {
        "differing": [n for n in names if (whole / n).read_bytes() != (directory / n).read_bytes()],
        "resume": {k: v for k, v in report["resume"].items() if k != "attempts"},
        "attempts": len(report["resume"]["attempts"]),
        "trace_calls_match": report["trace"]["calls"].keys() == whole_report["trace"]["calls"].keys()
        and report["trace"]["candidate_calls"].keys()
        == whole_report["trace"]["candidate_calls"].keys(),
        "failure_markers": sorted(p.name for p in (root / name).glob("failure*.json")),
    }
(root / "resume-check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
cut, verify = result["runs"]["cut"], result["runs"]["cutverify"]
assert not cut["differing"] and not verify["differing"], result
assert cut["resume"]["real_forward_calls"] == 0, "the final cut attempt should replay everything"
assert cut["resume"]["recomputed_and_verified_step"] is None
assert verify["resume"]["recomputed_and_verified_step"] is not None
assert cut["trace_calls_match"] and verify["trace_calls_match"]
PY
