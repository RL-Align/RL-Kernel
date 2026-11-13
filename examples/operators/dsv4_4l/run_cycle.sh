#!/usr/bin/env bash
# Run from a fresh image; only checkpoint, output and caches are mounted.
set -euo pipefail
if [[ $# != 5 ]]; then
  echo "usage: $0 IMAGE PHYSICAL_GPU CHECKPOINT NEW_OUTPUT_DIR CACHE_DIR" >&2
  exit 2
fi
image=$1
gpu=$2
model=$(realpath "$3")
output=$(realpath -m "$4")
cache=$(realpath -m "$5")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python3 "$script_dir/guard_gpu.py" "$gpu"
mkdir "$output"
mkdir -p "$cache"
docker image inspect "$image" > "$output/image-inspect.json"
checkpoint_sha=$(sha256sum "$model"/model.safetensors | cut -d ' ' -f 1)
docker run --pull=never --rm --init --name "rlk-dsv4-parity-$$" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --shm-size=8g --ipc=private \
  -e ROCR_VISIBLE_DEVICES="$gpu" -e HIP_VISIBLE_DEVICES=0 -e DSV4_PHYSICAL_GPU="$gpu" \
  -e OMP_NUM_THREADS=4 -e VLLM_ROCM_USE_AITER=1 -e VLLM_ENABLE_V1_MULTIPROCESSING=1 \
  -e DSV4_FIXED_SHARED_DOWN=1 -e DSV4_CHECKPOINT_SHA256="$checkpoint_sha" \
  -e TMPDIR=/outputs/tmp \
  -v "$cache:/cache" -v "$output:/outputs" -v "$model:/models/4l:ro" \
  "$image" bash -c 'mkdir /outputs/tmp; cd /opt/dsv4/RL-Kernel; exec /opt/dsv4-venv/bin/python -m examples.operators.dsv4_4l.cycle' \
  > "$output/container.log" 2>&1
