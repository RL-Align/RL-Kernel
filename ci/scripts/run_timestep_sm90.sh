#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compatibility validation using the original FP32 arithmetic on Hopper.
set -euo pipefail
export TIMESTEP_TARGET=sm90
export TORCH_CUDA_ARCH_LIST=9.0
export TIMESTEP_CUDA_JIT_ONLY=1
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_timestep_official.sh" "$@"
