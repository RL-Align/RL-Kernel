#!/usr/bin/env bash
set -euo pipefail

REAL_PYTHON="${RL_KERNEL_REAL_PYTHON:?RL_KERNEL_REAL_PYTHON must name the real Python executable}"
RL_KERNEL_ROOT="${RL_KERNEL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)}"
export RL_KERNEL_ROOT
export PYTHONPATH="${RL_KERNEL_ROOT}:${PYTHONPATH:-}"

if [[ "${1:-}" == "train.py" || "${1:-}" == */train.py ]]; then
  : "${RL_KERNEL_ATTENTION_CASE:?the ablation runner must select an Attention case}"
  : "${RL_KERNEL_FFN_CASE:?the ablation runner must freeze the FFN case}"
  : "${RL_KERNEL_LOGP_CASE:?the ablation runner must freeze the Logp case}"
  : "${RL_KERNEL_READBACK_DIR:?the ablation runner must provide a readback directory}"

  export RL_KERNEL_VLLM_INTEGRATION=1
  export RL_KERNEL_PLATFORM=rocm
  export RL_KERNEL_ROCM_STRICT_ATTENTION=1
  export RL_KERNEL_ROUTE_REPORT=1

  exec "${REAL_PYTHON}" "$@" \
    --seed "${TRAIN_SEED:-1234}" \
    --rollout-seed "${ROLLOUT_SEED:-42}" \
    --vllm-enable-deterministic-inference \
    --vllm-attention-backend rocm_aiter_fa \
    --vllm-disable-custom-all-reduce \
    --deterministic-mode \
    --accumulate-allreduce-grads-in-fp32
fi

exec "${REAL_PYTHON}" "$@"
