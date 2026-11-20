#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
GRPO_ROOT=/scratch/rl-kernel/vensen/dsv4-4l/runs/20261117T125400Z-verl-grpo
GRPO_IMAGE=rlk/dsv4-4l-rocm:grpo-stress
GRPO_IMAGE_ID=sha256:f3bf76c2d3863bf5b3f6a912d106a6eb2042ad9898c54f60b1e9015d6aeb225f
GRPO_OUTPUT=${1:?Provide a NEW output directory on the scratch data disk}
case "$GRPO_OUTPUT" in
  /scratch/rl-kernel/vensen/*) ;;
  *) echo 'Output must be under /scratch/rl-kernel/vensen/' >&2; exit 1 ;;
esac
test ! -e "$GRPO_OUTPUT"
test "$(docker image inspect -f '{{.Id}}' "$GRPO_IMAGE")" = "$GRPO_IMAGE_ID"
quota -s
df -h "$GRPO_ROOT"
# The launcher guards physical GPUs 4,5,6,7 and isolates KFD, render devices and Ray.
# Worker/optimizer state stays live for three native PPO updates in this one job.
python3 "$GRPO_ROOT/src/RL-Kernel/examples/operators/dsv4_4l/run_grpo.py" \
  "$GRPO_IMAGE" \
  /scratch/rl-kernel/vensen/dsv4-4l/models/DeepSeek-V4-Flash-4l \
  "$GRPO_ROOT/final-data" "$GRPO_OUTPUT" "$GRPO_ROOT/cache" \
  --audit-response-length 16 \
  --audit-prompts-file "$GRPO_ROOT/final-data/extended-audit-prompts.json" \
  --stress-final-repeats 1000
# This CPU-only verifier rechecks raw files and fails missing exits/matrix items.
docker run --rm --network=none --cpus=4 -e OMP_NUM_THREADS=4 \
  -v "$GRPO_OUTPUT:/outputs" -v "$GRPO_ROOT/final-data:/data:ro" \
  "$GRPO_IMAGE" /opt/dsv4-venv/bin/python \
  /opt/dsv4/RL-Kernel/examples/operators/dsv4_4l/verify_grpo.py \
  /outputs/policy-sync --inventory /data/checkpoint-inventory.json
