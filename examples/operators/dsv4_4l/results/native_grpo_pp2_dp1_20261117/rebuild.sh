#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
GRPO_ROOT=/scratch/rl-kernel/vensen/dsv4-4l/runs/20261117T125400Z-verl-grpo
GRPO_BUILD=${1:?Provide a NEW scratch build directory}
case "$GRPO_BUILD" in
  /scratch/rl-kernel/vensen/*) ;;
  *) echo 'Build must be under /scratch/rl-kernel/vensen/' >&2; exit 1 ;;
esac
test ! -e "$GRPO_BUILD"
quota -s
docker info --format '{{.DockerRootDir}}'
mkdir "$GRPO_BUILD"
git -C "$GRPO_ROOT/src/RL-Kernel" worktree add --detach "$GRPO_BUILD/rlk" 478caf1a787f6c9d9d3747a45ad79756b6c2c616
git -C "$GRPO_ROOT/src/verl" worktree add --detach "$GRPO_BUILD/verl" 8196af485ae50e09aa42c1fc6e399f266fb166d3
python3 "$GRPO_BUILD/rlk/examples/operators/dsv4_4l/build_grpo.py" \
  "$GRPO_BUILD/rlk" "$GRPO_BUILD/verl" "$GRPO_BUILD/context" \
  rlk/dsv4-4l-rocm:grpo-stress-rebuild
# build_grpo.py requires the exact recorded base ID and committed clean sources.
# Original measured git bundles also remain in $GRPO_ROOT/build/stress-code/.
