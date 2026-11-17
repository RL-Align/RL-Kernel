#!/usr/bin/env bash
# Re-run the two groups sequentially using the immutable measured image.
set -euo pipefail
new=${1:?usage: reproduce.sh NEW_ABSOLUTE_SCRATCH_OUTPUT}
[[ "$new" == /scratch/rl-kernel/vensen/* && ! -e "$new" ]] || {
  echo 'Choose a new output directory under /scratch/rl-kernel/vensen/' >&2
  exit 2
}
report=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
base=$(cd "$report/../.." && pwd)
quota -s
common=(--output-root "$new" --image-manifest "$report/image.json"
  --checkpoint /scratch/rl-kernel/vensen/dsv4-4l/models/DeepSeek-V4-Flash-4l
  --cache /scratch/rl-kernel/vensen/dsv4-4l/runs/20261116T024532Z/cache
  --fixed-cases "$report/predeclared-cases.json")
python3 "$base/run_comparison.py" --group P/P "${common[@]}"
python3 "$base/run_comparison.py" --group R/R "${common[@]}"
python3 "$base/aggregate_comparison.py" \
  --native-job "$new/artifacts/job-native-PP.json" \
  --rlk-job "$new/artifacts/job-rlk-RR-rocm_triton.json" --output-dir "$new/report"
plot_deps=/scratch/rl-kernel/vensen/dsv4-4l/runs/20261117T105017Z-native-vs-rlk/cache/plot-packages
PYTHONPATH="$plot_deps" MPLCONFIGDIR="$new/cache/mpl" \
  python3 "$base/plot_comparison.py" "$new/report/summary.json" --output-dir "$new/report"
python3 "$base/write_comparison_report.py" "$new/report/summary.json"
