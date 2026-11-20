#!/usr/bin/env bash
# Measure trainer peak GPU memory on the frozen rollout before any candidate run.
#
# Runs tests.distributed.tp.check_qwen3_fixed --probe for every engine, precision and role at
# each TP size given that the row runs at (fixed_matrix.sh). With no sizes given, each row is
# probed once, at its smallest TP size. A probe writes no evidence tensors and needs no
# acceptance manifest, so it cannot set or widen a bound. A failed probe (for example an
# out-of-memory error) does not stop the loop; summary.tsv lists each run's status and per-rank
# peak.
#
# Usage: probe_trainer_memory.sh <source> <python> <framework_overlay> <checkpoint> \
#   <reference_root> <result_root> [tp ...]
set -euo pipefail
source_dir=$1
python_bin=$2
framework_overlay=$3
checkpoint=$4
reference_root=$5
result_root=$6
shift 6
sizes=("$@")
cd "$source_dir"
source tools/validation/models/qwen3_ppo/fixed_matrix.sh
[[ -z $(git status --porcelain) ]]
result_dir="$result_root/probe-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$result_dir"
export PYTHONPATH="$source_dir:$framework_overlay"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv > "$result_dir/device.txt"
git rev-parse HEAD > "$result_dir/commit.txt"
for engine in megatron deepspeed; do
  for precision in fp32 bf16; do
    for role in actor critic; do
      read -r -a row_sizes <<< "$(fixed_tp_sizes "$engine" "$precision" "$role")"
      if [[ ${#sizes[@]} -eq 0 ]]; then row_sizes=("${row_sizes[0]}"); fi
      for tp in "${row_sizes[@]}"; do
        if [[ ${#sizes[@]} -gt 0 && " ${sizes[*]} " != *" $tp "* ]]; then continue; fi
        run="$result_dir/$engine-$precision-$role-tp$tp"
        rc=0
        "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$tp" \
          -m tests.distributed.tp.check_qwen3_fixed --probe --framework "$engine" \
          --precision "$precision" --role "$role" --checkpoint "$checkpoint" \
          --rollout "$reference_root/rollout" --targets "$reference_root/targets" \
          --output "$run" > "$run.log" 2>&1 || rc=$?
        echo "$run rc=$rc" >> "$result_dir/steps.txt"
      done
    done
  done
done
"$python_bin" - "$result_dir" <<'EOF'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
total = None
lines = ["run\tstatus\tpeak_allocated_gib_per_rank\tseconds\terror"]
for run in sorted(p for p in root.iterdir() if p.is_dir()):
    ranks = sorted(run.glob("rank*.json"))
    rows = [json.loads(r.read_text()) for r in ranks]
    if not rows:
        lines.append(f"{run.name}\tno report\t\t\t")
        continue
    peaks = ",".join(f'{r.get("peak_allocated_bytes", 0) / 2**30:.1f}' for r in rows)
    first = rows[0]
    error = (first.get("error") or "").replace("\t", " ").replace("\n", " ")[:300]
    lines.append(f'{run.name}\t{first["status"]}\t{peaks}\t{first.get("seconds", 0):.0f}\t{error}')
(root / "summary.tsv").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
