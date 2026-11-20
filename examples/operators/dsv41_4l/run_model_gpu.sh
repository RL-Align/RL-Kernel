#!/usr/bin/env bash
# Real text-model execution. Refuse occupied physical GPUs and preserve each run.
set -euo pipefail
# Clear inherited second-stage HIP/CUDA remapping before any Python child.
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HSA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=4,5,6,7
if [[ $# != 8 ]]; then
  echo "usage: $0 IMAGE PHYSICAL_GPU_CSV CHECKPOINT NEW_OUTPUT TP PP DP BASELINE_OR_DASH" >&2
  exit 2
fi
image=$1
gpu_csv=$2
checkpoint=$(realpath "$3")
output=$(realpath -m "$4")
tp=$5
pp=$6
dp=$7
baseline=$8
repo=$(git rev-parse --show-toplevel)
source_sha=$(git rev-parse HEAD)
if [[ $(git branch --show-current) != dsv4.1-rocm ]]; then
  echo "model acceptance must run on dsv4.1-rocm" >&2
  exit 2
fi
if [[ -n $(git status --porcelain --untracked-files=normal -- rl_engine tests examples/operators/dsv41_4l) ]]; then
  echo "commit the tested source before launching acceptance" >&2
  exit 2
fi
python3 - "$gpu_csv" "$tp" "$pp" "$dp" <<'PY'
import sys
from examples.operators.dsv41_4l.gpu_visibility import physical_ids
physical_ids(sys.argv[1])
ids=sys.argv[1].split(',')
tp,pp,dp=map(int,sys.argv[2:])
if (not ids or any(not i.isdigit() for i in ids) or len(set(ids))!=len(ids)
    or tp not in (1,2,4) or pp not in (1,2) or dp not in (1,2)
    or tp*pp*dp!=len(ids) or (pp==1 and dp!=1)):
    raise SystemExit('unsupported or inconsistent physical GPU/topology request')
PY
export ROCR_VISIBLE_DEVICES="$gpu_csv"
mkdir "$output"
python3 -m examples.operators.dsv41_4l.gpu_visibility host \
  --physical "$gpu_csv" --output "$output/gpu-expected.json" > "$output/host-mapping.log"
python3 - "$output" "$source_sha" "$image" "$gpu_csv" "$tp" "$pp" "$dp" <<'PY'
import json,pathlib,sys
out,sha,image,gpus,tp,pp,dp=sys.argv[1:]
pathlib.Path(out,'provenance.json').write_text(json.dumps({
 'source_commit':sha,'image':image,'physical_gpus':gpus,
 'tp':int(tp),'pp':int(pp),'dp':int(dp),
 'physical_gpu_ids':[int(i) for i in gpus.split(',')],
 'logical_gpu_ids':list(range(len(gpus.split(',')))),
 'rocr_visible_devices':gpus,
 'forbidden_physical_gpus':[0,1,2,3],
 'eight_gpu_tests':'CANCELED by current user instruction',
 'scope':'real original-layer text model; frozen quantized weights/embedding/head',
 'parity':'shared-kernel parity; TP shards attention, FFN replicated'
},indent=2)+'\n')
PY
IFS=, read -ra task_gpus <<< "$gpu_csv"
for gpu in "${task_gpus[@]}"; do
  if ! python3 "$repo/examples/operators/dsv4_4l/guard_gpu.py" "$gpu" >> "$output/gpu-preflight.log" 2>&1; then
    python3 - "$output" "$gpu" <<'PY'
import json,pathlib,sys
pathlib.Path(sys.argv[1],'summary.json').write_text(json.dumps({
 'status':'BLOCKED','phase':'GPU preflight','gpu':int(sys.argv[2]),
 'reason':'requested physical GPU occupied; see gpu-preflight.log',
 'model_execution':'NOT RUN'},indent=2)+'\n')
PY
    exit 3
  fi
done
mkdir -p "$output/cache/tmp"
if ! docker run --rm --pull=never --network=none --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -e ROCR_VISIBLE_DEVICES="$gpu_csv" -e PYTHONPATH=/work \
  -v "$repo:/work:ro" -v "$output:/outputs:rw" -w /work "$image" \
  env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL -u HSA_VISIBLE_DEVICES \
  python -m examples.operators.dsv41_4l.gpu_visibility verify \
  --expected /outputs/gpu-expected.json --output /outputs/gpu-mapping.json \
  > "$output/container-mapping.log" 2>&1; then
  echo '{"status":"BLOCKED","phase":"container GPU mapping","model_execution":"NOT RUN"}' > "$output/summary.json"
  exit 3
fi
cp "$checkpoint/manifest.json" "$output/checkpoint-manifest.json"
docker image inspect --format '{{.Id}}' "$image" > "$output/image-id.txt"
world=${#task_gpus[@]}
launch=(python)
if [[ $world -gt 1 ]]; then
  launch+=(-m torch.distributed.run --nnodes=1 --master-addr=127.0.0.1 --master-port=29501 --nproc-per-node="$world")
fi
mounts=()
if [[ $pp -eq 1 ]]; then
  launch+=(-m examples.operators.dsv41_4l.run_model --checkpoint /weights --output /outputs --tp "$tp" --tokens "${DSV41_MODEL_TOKENS:-8}" --case "${DSV41_MODEL_CASE:-0}")
else
  baseline=$(realpath "$baseline")
  mounts=(-v "$baseline:/baseline:ro")
  launch+=(-m examples.operators.dsv41_4l.run_pipeline --checkpoint /weights --output /outputs --baseline /baseline --tp "$tp" --dp "$dp" --optimizer-steps "${DSV41_MODEL_OPTIMIZER_STEPS:-0}")
fi
if [[ -n ${DSV41_DP_SECOND_BASELINE:-} ]]; then
  second_baseline=$(realpath "$DSV41_DP_SECOND_BASELINE")
  mounts+=(-v "$second_baseline:/second-baseline:ro")
  launch+=(--second-baseline /second-baseline)
fi
docker run --rm --pull=never --network=none --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --shm-size=8g --ipc=private -e ROCR_VISIBLE_DEVICES="$gpu_csv" \
  -e OMP_NUM_THREADS=4 -e PYTHONPATH=/work -e TMPDIR=/outputs/cache/tmp \
  -v "$repo:/work:ro" -v "$checkpoint:/weights:ro" -v "$output:/outputs:rw" \
  -e DSV41_EXPECTED_GPU_MAPPING=/outputs/gpu-expected.json \
  "${mounts[@]}" -w /work "$image" \
  env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL -u HSA_VISIBLE_DEVICES \
  "${launch[@]}" > "$output/container.log" 2>&1
