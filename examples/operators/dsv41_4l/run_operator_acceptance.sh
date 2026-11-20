#!/usr/bin/env bash
# Isolated components only. This does not construct or shard a four-layer model.
set -euo pipefail
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HSA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=4,5,6,7
if [[ $# != 5 ]]; then
  echo "usage: $0 IMAGE PHYSICAL_GPU_CSV OPERATOR_WEIGHTS NEW_OUTPUT CACHE" >&2
  exit 2
fi
image=$1
gpu_csv=$2
weights=$(realpath "$3")
output=$(realpath -m "$4")
cache=$(realpath -m "$5")
repo=$(git rev-parse --show-toplevel)
python3 - "$gpu_csv" <<'PYGPU'
import sys
from examples.operators.dsv41_4l.gpu_visibility import physical_ids
physical_ids(sys.argv[1])
PYGPU
export ROCR_VISIBLE_DEVICES="$gpu_csv"
source_sha=$(git -C "$repo" rev-parse HEAD)
if [[ -n $(git -C "$repo" status --porcelain --untracked-files=normal -- rl_engine tests examples/operators/dsv41_4l) ]]; then
  echo "commit the tested source before launching acceptance" >&2
  exit 2
fi
IFS=, read -ra task_gpus <<< "$gpu_csv"
python3 - "$gpu_csv" <<'PY'
import sys
ids=sys.argv[1].split(',')
if not ids or any(not i.isdigit() for i in ids) or len(ids) != len(set(ids)):
    raise SystemExit('expected unique nonnegative physical GPU IDs')
PY
for gpu in "${task_gpus[@]}"; do
  python3 "$repo/examples/operators/dsv4_4l/guard_gpu.py" "$gpu"
done
mkdir "$output"
mkdir -p "$cache/tmp"
docker image inspect "$image" > "$output/image-inspect.json"
args=(python -m examples.operators.dsv41_4l.operator_acceptance)
if [[ ${#task_gpus[@]} -gt 1 ]]; then
  args=(python -m torch.distributed.run --nnodes=1 --master-addr=127.0.0.1 --master-port=29501 --nproc-per-node="${#task_gpus[@]}" -m examples.operators.dsv41_4l.operator_acceptance)
fi
docker run --rm --pull=never --network=none \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --shm-size=8g --ipc=private \
  -e ROCR_VISIBLE_DEVICES="$gpu_csv" -e OMP_NUM_THREADS=4 \
  -e PYTHONPATH=/work -e TMPDIR=/cache/tmp \
  -v "$repo:/work:ro" -v "$weights:/weights:ro" \
  -v "$output:/outputs:rw" -v "$cache:/cache:rw" -w /work \
  "$image" env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL -u HSA_VISIBLE_DEVICES "${args[@]}" --weights /weights --output /outputs --rlk-commit "$source_sha" \
  > "$output/container.log" 2>&1
python3 - "$output" "${#task_gpus[@]}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);world=int(sys.argv[2])
for rank in range(world):
    record=json.loads((root/f'operator-rank-{rank}.json').read_text())
    if record['status'] != 'PASS' or record['rank'] != rank or record['world_size'] != world:
        raise SystemExit(f'FAIL rank {rank}')
print(f'PASS {world} ranks: isolated operators and component RCCL only; model NOT RUN')
PY
