#!/usr/bin/env bash
set -euo pipefail

REAL_PYTHON="${RL_KERNEL_REAL_PYTHON:?RL_KERNEL_REAL_PYTHON must name the real Python executable}"
RL_KERNEL_ROOT="${RL_KERNEL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)}"
export RL_KERNEL_ROOT
export PYTHONPATH="${RL_KERNEL_ROOT}:${PYTHONPATH:-}"

if [[ "${1:-}" == "train.py" || "${1:-}" == */train.py ]]; then
  # Ray job submission does not always forward the shell exports used by the
  # outer launcher. A strict linear-logp job must still install the matching
  # vLLM hooks; otherwise it silently falls back to native attention/FFN and
  # loses the R/R performance path. Preserve explicit ablation selections.
  strict_linear_logp=0
  rollout_batch_size=""
  n_samples_per_prompt="1"
  explicit_vllm_execution_config=0
  previous_arg=""
  for current_arg in "$@"; do
    if [[ "${previous_arg}" == "--linear-logp-provider-mode" && "${current_arg}" == "strict" ]]; then
      strict_linear_logp=1
    elif [[ "${previous_arg}" == "--rollout-batch-size" ]]; then
      rollout_batch_size="${current_arg}"
    elif [[ "${previous_arg}" == "--n-samples-per-prompt" ]]; then
      n_samples_per_prompt="${current_arg}"
    fi

    case "${current_arg}" in
      --linear-logp-provider-mode=strict)
        strict_linear_logp=1
        ;;
      --rollout-batch-size=*)
        rollout_batch_size="${current_arg#*=}"
        ;;
      --n-samples-per-prompt=*)
        n_samples_per_prompt="${current_arg#*=}"
        ;;
      --vllm-enforce-eager|--vllm-optimization-level|--vllm-optimization-level=*|--vllm-compilation-config|--vllm-compilation-config=*)
        explicit_vllm_execution_config=1
        ;;
    esac
    previous_arg="${current_arg}"
  done

  required_cudagraph_args=()
  if [[ "${strict_linear_logp}" == "1" ]]; then
    export RL_KERNEL_VLLM_INTEGRATION="${RL_KERNEL_VLLM_INTEGRATION:-1}"
    export RL_KERNEL_CUDA_ONLY="${RL_KERNEL_CUDA_ONLY:-1}"
    export VIME_RL_KERNEL_STRICT="${VIME_RL_KERNEL_STRICT:-1}"
    export RL_KERNEL_ATTENTION_CASE="${RL_KERNEL_ATTENTION_CASE:-R/R}"
    export RL_KERNEL_FFN_CASE="${RL_KERNEL_FFN_CASE:-R/R}"
    export RL_KERNEL_LOGP_CASE="${RL_KERNEL_LOGP_CASE:-R/R}"
  fi

  # CUDA Graph is a frozen matrix setting, independent of the linear-logp
  # provider route. Capturing the complete decode graph removes per-layer
  # host-launch gaps and keeps P/P and R/R performance comparisons aligned.
  # Capture every exact batch size; explicit execution flags still win.
  if [[ "${explicit_vllm_execution_config}" == "0" ]]; then
    if [[ "${rollout_batch_size}" =~ ^[1-9][0-9]*$ && "${n_samples_per_prompt}" =~ ^[1-9][0-9]*$ ]]; then
      max_capture_size=$((rollout_batch_size * n_samples_per_prompt))
      if [[ -n "${RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE:-}" ]]; then
        max_capture_size="${RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE}"
      fi
      if ! [[ "${max_capture_size}" =~ ^[1-9][0-9]*$ ]]; then
        echo "RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE must be a positive integer" >&2
        exit 2
      fi

      capture_sizes="["
      for ((batch_size = 1; batch_size <= max_capture_size; batch_size++)); do
        if ((batch_size > 1)); then
          capture_sizes+=","
        fi
        capture_sizes+="${batch_size}"
      done
      capture_sizes+="]"
      compilation_config="{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":${capture_sizes},\"max_cudagraph_capture_size\":${max_capture_size}}"
      required_cudagraph_args=(
        --vllm-optimization-level 0
        --vllm-compilation-config "${compilation_config}"
      )
      echo "[RL-Kernel] required vLLM full-decode CUDA Graph capture sizes: ${capture_sizes}" >&2
    else
      echo "[RL-Kernel] required CUDA Graph configuration lacks rollout batch sizes" >&2
      exit 2
    fi
  fi
  exec "${REAL_PYTHON}" "$@" \
    --seed "${RL_KERNEL_SEED:-1234}" \
    --rollout-seed "${RL_KERNEL_ROLLOUT_SEED:-42}" \
    --vllm-enable-deterministic-inference \
    --vllm-attention-backend flash_attn \
    --vllm-disable-custom-all-reduce \
    --deterministic-mode \
    --accumulate-allreduce-grads-in-fp32 \
    "${required_cudagraph_args[@]}"
fi

exec "${REAL_PYTHON}" "$@"
