#!/usr/bin/env bash
set -euo pipefail
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HSA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=4,5,6,7
if [[ $# != 3 ]]; then
  echo "usage: $0 IMAGE IDLE_PHYSICAL_GPU_CSV NEW_ARTIFACT_ROOT" >&2
  exit 2
fi
image=$1
gpu_csv=$2
artifacts=$(realpath -m "$3")
repo=$(git rev-parse --show-toplevel)
python3 - "$gpu_csv" <<'PYGPU'
import sys
from examples.operators.dsv41_4l.gpu_visibility import physical_ids
physical_ids(sys.argv[1])
PYGPU
export ROCR_VISIBLE_DEVICES="$gpu_csv"
mkdir "$artifacts"
mkdir "$artifacts/tmp"
docker run --rm --pull=never -e ROCR_VISIBLE_DEVICES="$gpu_csv" -e PYTHONPATH=/work -e TMPDIR=/artifacts/tmp \
  -v "$repo:/work:ro" -v "$artifacts:/artifacts:rw" -w /work \
  "$image" env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL -u HSA_VISIBLE_DEVICES python -m examples.operators.dsv41_4l.fetch_operator_weights /artifacts/weights \
  > "$artifacts/checkpoint-fetch.log" 2>&1
docker run --rm --pull=never -e ROCR_VISIBLE_DEVICES="$gpu_csv" -e PYTHONPATH=/work -e TMPDIR=/artifacts/tmp \
  -v "$repo:/work:ro" -v "$artifacts:/artifacts:rw" -w /work \
  "$image" env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL -u HSA_VISIBLE_DEVICES python -m examples.operators.dsv41_4l.prepare_token_map /artifacts/tokenizer \
  > "$artifacts/token-map.log" 2>&1
"$repo/examples/operators/dsv41_4l/run_operator_acceptance.sh" \
  "$image" "$gpu_csv" "$artifacts/weights" "$artifacts/acceptance" "$artifacts/cache"
# Component profile uses the 35MiB operator slice. Full source-layer tests require
# the model checkpoint and are run by the separate model validation workflow.
first_gpu=${gpu_csv%%,*}
python3 "$repo/examples/operators/dsv4_4l/guard_gpu.py" "$first_gpu"
docker run --rm --pull=never --network=none --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -e ROCR_VISIBLE_DEVICES="$first_gpu" -e PYTHONPATH=/work -e RLK_REQUIRE_ROCM=1 \
  -e OMP_NUM_THREADS=4 -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  -e DSV41_OPERATOR_WEIGHTS=/artifacts/weights -e DSV41_TOKENIZER=/artifacts/tokenizer \
  -e TMPDIR=/artifacts/tmp -v "$repo:/work:ro" -v "$artifacts:/artifacts:rw" -w /work \
  "$image" env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL -u HSA_VISIBLE_DEVICES python -m pytest -q -p no:cacheprovider \
  tests/models/deepseek_v41/test_config.py \
  tests/models/deepseek_v41/test_checkpoint.py \
  tests/models/deepseek_v41/test_bitwise.py \
  tests/models/deepseek_v41/test_operators_rocm.py \
  tests/models/deepseek_v41/test_official_operators_rocm.py \
  tests/models/deepseek_v41/test_attention_transforms_rocm.py \
  tests/models/deepseek_v41/test_delayed_mhc_rocm.py \
  tests/models/deepseek_v41/test_gpu_visibility.py \
  > "$artifacts/pytest.log" 2>&1
cat "$artifacts/pytest.log"
