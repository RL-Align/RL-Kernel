#!/usr/bin/env bash
# True PP2 actor + PP2 rollout; optional DP2 adds native DDP gradient reduction.
set -euo pipefail
if [[ $# != 6 ]]; then
  echo 'usage: IMAGE ACTOR_PHYSICAL_GPU_CSV ROLLOUT_PHYSICAL_GPU_CSV CHECKPOINT NEW_OUTPUT CACHE' >&2
  exit 2
fi
image=$1
actors=$2
rollouts=$3
model=$(realpath "$4")
output=$(realpath -m "$5")
cache=$(realpath -m "$6")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
backend=${DSV4_MHC_BACKEND:-rocm}
[[ "$backend" == rocm || "$backend" == rocm_triton ]] || exit 2
actor_count=$(python3 - "$actors" "$rollouts" <<'PY'
import sys
actors, rollouts = [value.split(',') for value in sys.argv[1:]]
if (len(actors) not in (2, 4) or len(rollouts) != 2
    or any(not value.isdecimal() for value in actors + rollouts)
    or any(value != str(int(value)) for value in actors + rollouts)
    or len(set(map(int, actors + rollouts))) != len(actors + rollouts)):
    raise SystemExit('require 2 or 4 actor cards plus 2 disjoint rollout cards')
print(len(actors))
PY
)
# Check every selected physical device before preparing data and again at launch.
IFS=, read -ra gpu_ids <<< "$actors,$rollouts"
for gpu in "${gpu_ids[@]}"; do
  python3 "$script_dir/guard_gpu.py" "$gpu"
done
mkdir "$output"
mkdir -p "$cache"
docker image inspect "$image" > "$output/image-inspect.json"
checkpoint_sha=$(sha256sum "$model/model.safetensors" | cut -d ' ' -f 1)
for gpu in "${gpu_ids[@]}"; do
  python3 "$script_dir/guard_gpu.py" "$gpu"
done
python3 - "$actors" "$rollouts" "$backend" > "$output/requested-topology.json" <<'PY'
import json, sys
actors, rollout, backend = sys.argv[1:]
print(json.dumps(dict(actor_physical_gpus=actors.split(','), rollout_physical_gpus=rollout.split(','),
    actor=dict(TP=1, PP=2, EP=1, CP=1, DP=len(actors.split(','))//2),
    rollout=dict(TP=1, PP=2, EP=1, CP=1, DP=1), backend=backend,
    original_layers_by_pp=[[0,1,2],[3]], eager=True), indent=2))
PY
docker run --pull=never --rm --init --name "rlk-dsv4-pipeline-$$" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --shm-size=16g --ipc=private \
  -e ROCR_VISIBLE_DEVICES="$rollouts" -e HIP_VISIBLE_DEVICES=0,1 \
  -e DSV4_ACTOR_PHYSICAL_GPUS="$actors" -e DSV4_ROLLOUT_PHYSICAL_GPUS="$rollouts" \
  -e DSV4_ACTOR_PP=2 -e DSV4_ACTOR_DP="$((actor_count / 2))" \
  -e VLLM_PP_LAYER_PARTITION=3,1 \
  -e OMP_NUM_THREADS=4 -e VLLM_ROCM_USE_AITER=1 -e VLLM_ENABLE_V1_MULTIPROCESSING=1 \
  -e DSV4_FIXED_SHARED_DOWN=1 -e DSV4_CHECKPOINT_SHA256="$checkpoint_sha" \
  -e DSV4_MHC_BACKEND="$backend" \
  -e HF_HOME=/cache/huggingface -e TRITON_CACHE_DIR=/cache/triton \
  -e XDG_CACHE_HOME=/cache/xdg -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e TMPDIR=/outputs/tmp \
  -v "$cache:/cache" -v "$output:/outputs" -v "$model:/models/4l:ro" \
  "$image" bash -c 'mkdir /outputs/tmp; cd /opt/dsv4/RL-Kernel; exec /opt/dsv4-venv/bin/python -m examples.operators.dsv4_4l.cycle' \
  > "$output/container.log" 2>&1
python3 "$script_dir/verify_pipeline.py" "$output" --dp-size "$((actor_count / 2))"
