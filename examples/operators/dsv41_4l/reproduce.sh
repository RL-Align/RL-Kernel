#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "usage: $0 IMAGE IDLE_PHYSICAL_GPU_CSV NEW_ARTIFACT_ROOT" >&2
  exit 2
fi
image=$1
gpu_csv=$2
artifacts=$(realpath -m "$3")
repo=$(git rev-parse --show-toplevel)
mkdir "$artifacts"
mkdir "$artifacts/tmp"
docker run --rm --pull=never -e PYTHONPATH=/work -e TMPDIR=/artifacts/tmp \
  -v "$repo:/work:ro" -v "$artifacts:/artifacts:rw" -w /work \
  "$image" python -m examples.operators.dsv41_4l.fetch_operator_weights /artifacts/weights \
  > "$artifacts/checkpoint-fetch.log" 2>&1
docker run --rm --pull=never -e PYTHONPATH=/work -e TMPDIR=/artifacts/tmp \
  -v "$repo:/work:ro" -v "$artifacts:/artifacts:rw" -w /work \
  "$image" python -m examples.operators.dsv41_4l.prepare_token_map /artifacts/tokenizer \
  > "$artifacts/token-map.log" 2>&1
"$repo/examples/operators/dsv41_4l/run_operator_acceptance.sh" \
  "$image" "$gpu_csv" "$artifacts/weights" "$artifacts/acceptance" "$artifacts/cache"
first_gpu=${gpu_csv%%,*}
python3 "$repo/examples/operators/dsv4_4l/guard_gpu.py" "$first_gpu"
docker run --rm --pull=never --network=none --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -e ROCR_VISIBLE_DEVICES="$first_gpu" -e PYTHONPATH=/work -e RLK_REQUIRE_ROCM=1 \
  -e OMP_NUM_THREADS=4 -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  -e DSV41_OPERATOR_WEIGHTS=/artifacts/weights -e DSV41_TOKENIZER=/artifacts/tokenizer \
  -e TMPDIR=/artifacts/tmp -v "$repo:/work:ro" -v "$artifacts:/artifacts:rw" -w /work \
  "$image" python -m pytest -q -p no:cacheprovider tests/models/deepseek_v41 \
  > "$artifacts/pytest.log" 2>&1
cat "$artifacts/pytest.log"
