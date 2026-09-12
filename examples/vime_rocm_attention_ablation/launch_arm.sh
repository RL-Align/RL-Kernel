#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

# Launch one correctness-first Qwen3-8B/Vime ROCm Attention ablation arm.
# `run.py` supplies every RLK_ABLATION_* value and invokes this script once for
# each of P/P, P/R, R/P, and R/R.  This launcher never reuses another arm's
# output checkpoint and never stops a Ray cluster it did not start.

set -euo pipefail

: "${RLK_ABLATION_CASE_ID:?}"
: "${RLK_ABLATION_ARM_DIR:?}"
: "${RLK_ABLATION_VIME_ROOT:?}"
: "${RLK_ABLATION_RL_KERNEL_ROOT:?}"
: "${RLK_ABLATION_MEGATRON_ROOT:?}"
: "${RLK_ABLATION_MODEL_ROOT:?}"
: "${RLK_ABLATION_REFERENCE_CHECKPOINT:?}"
: "${RLK_ABLATION_PROMPT_DATA:?}"
: "${RLK_ABLATION_NUM_GPUS:?}"
: "${RLK_ABLATION_TP_SIZE:?}"
: "${RLK_ABLATION_CP_SIZE:?}"
: "${RLK_ABLATION_ROLLOUT_TP_SIZE:?}"
: "${RLK_ABLATION_COLOCATE:?}"
: "${RLK_ABLATION_OFFLOAD_TRAIN:?}"
: "${RLK_ABLATION_OFFLOAD_ROLLOUT:?}"
: "${RLK_ABLATION_ROUTER_POLICY:?}"
: "${RLK_ABLATION_NUM_ROLLOUT:?}"
: "${RLK_ABLATION_ROLLOUT_BATCH_SIZE:?}"
: "${RLK_ABLATION_SAMPLES_PER_PROMPT:?}"
: "${RLK_ABLATION_GLOBAL_BATCH_SIZE:?}"
: "${RLK_ABLATION_MAX_RESPONSE_LENGTH:?}"
: "${RLK_ABLATION_MAX_TOKENS_PER_GPU:?}"
: "${RLK_ABLATION_SEED:?}"
: "${RLK_ABLATION_ROLLOUT_SEED:?}"
: "${RLK_ABLATION_RAY_PORT:?}"
: "${RLK_ABLATION_RAY_DASHBOARD_PORT:?}"
: "${RLK_ABLATION_RAY_DASHBOARD_AGENT_PORT:?}"
: "${RL_KERNEL_READBACK_DIR:?}"
: "${RL_KERNEL_MISMATCH_SIDECAR_DIR:?}"
: "${RL_KERNEL_VLLM_REAL_VOCAB_SIZE:?}"
: "${RL_KERNEL_VLLM_PADDED_VOCAB_SIZE:?}"

case "${RLK_ABLATION_CASE_ID}" in
  P/P|P/R|R/P|R/R) ;;
  *)
    echo "RLK_ABLATION_CASE_ID must be P/P, P/R, R/P, or R/R" >&2
    exit 2
    ;;
esac

if [[ "${RL_KERNEL_ATTENTION_CASE:-}" != "${RLK_ABLATION_CASE_ID}" ]]; then
  echo "RL_KERNEL_ATTENTION_CASE disagrees with the arm ID" >&2
  exit 2
fi
for module_case in "${RL_KERNEL_FFN_CASE:-}" "${RL_KERNEL_LOGP_CASE:-}"; do
  case "${module_case}" in
    P/P|R/R) ;;
    *)
      echo "FFN and Logp cases must each be P/P or R/R" >&2
      exit 2
      ;;
  esac
done
if [[ "${RL_KERNEL_VLLM_INTEGRATION:-}" != "1" ]]; then
  echo "RL_KERNEL_VLLM_INTEGRATION=1 is required for rollout route readback" >&2
  exit 2
fi

TRAIN_GPUS=$((RLK_ABLATION_TP_SIZE * RLK_ABLATION_CP_SIZE))
if [[ "${RLK_ABLATION_COLOCATE}" == "1" ]]; then
  ROLLOUT_GPUS="${RLK_ABLATION_NUM_GPUS}"
  if (( TRAIN_GPUS != RLK_ABLATION_NUM_GPUS )); then
    echo "colocated training TP*CP must use all visible GPUs" >&2
    exit 2
  fi
else
  ROLLOUT_GPUS=$((RLK_ABLATION_NUM_GPUS - TRAIN_GPUS))
fi
if (( TRAIN_GPUS <= 0 || ROLLOUT_GPUS <= 0 )); then
  echo "the requested TP/CP topology does not leave a valid rollout allocation" >&2
  exit 2
fi
if (( ROLLOUT_GPUS % RLK_ABLATION_ROLLOUT_TP_SIZE != 0 )); then
  echo "rollout GPU count must be divisible by rollout TP size" >&2
  exit 2
fi
if [[ "${RLK_ABLATION_ROUTER_POLICY}" != "round_robin" ]]; then
  echo "the two-engine strict matrix requires round_robin routing" >&2
  exit 2
fi

COLOCATE_ARGS=()
if [[ "${RLK_ABLATION_COLOCATE}" == "1" ]]; then
  COLOCATE_ARGS+=(--colocate)
fi
if [[ "${RLK_ABLATION_OFFLOAD_TRAIN}" == "1" ]]; then
  COLOCATE_ARGS+=(--offload-train)
else
  COLOCATE_ARGS+=(--no-offload-train)
fi
if [[ "${RLK_ABLATION_OFFLOAD_ROLLOUT}" == "1" ]]; then
  COLOCATE_ARGS+=(--offload-rollout)
else
  COLOCATE_ARGS+=(--no-offload-rollout)
fi

for required in \
  "${RLK_ABLATION_VIME_ROOT}/train.py" \
  "${RLK_ABLATION_VIME_ROOT}/scripts/models/qwen3-8B.sh" \
  "${RLK_ABLATION_RL_KERNEL_ROOT}/rl_engine" \
  "${RLK_ABLATION_RL_KERNEL_ROOT}/examples/vime_rocm_attention_ablation/tis_metrics.py" \
  "${RLK_ABLATION_MEGATRON_ROOT}" \
  "${RLK_ABLATION_MODEL_ROOT}" \
  "${RLK_ABLATION_REFERENCE_CHECKPOINT}" \
  "${RLK_ABLATION_PROMPT_DATA}"
do
  if [[ ! -e "${required}" ]]; then
    echo "required ROCm matrix path does not exist: ${required}" >&2
    exit 3
  fi
done

unset CUBLASLT_WORKSPACE_SIZE CUBLAS_WORKSPACE_CONFIG NCCL_ALGO
# Vime's rollout client (httpx) and the router honor proxy variables. A host
# proxy inherited from the operator's shell turns every generation longer than
# the proxy's upstream timeout into a 502 retry loop, so never forward them.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY
unset RL_KERNEL_CUDA_ONLY RL_KERNEL_DET_GEMM_SM90_ONLY RL_KERNEL_PRECOMPILE_FA4
unset VLLM_BATCH_INVARIANT NVTE_FUSED_ATTN NVTE_FLASH_ATTN NVTE_UNFUSED_ATTN

export PYTHONUNBUFFERED=1
export PYTHONPATH="${RLK_ABLATION_RL_KERNEL_ROOT}/examples:${RLK_ABLATION_RL_KERNEL_ROOT}:${RLK_ABLATION_VIME_ROOT}:${RLK_ABLATION_MEGATRON_ROOT}:${PYTHONPATH:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?}"
export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"
# AITER's JIT resolves GPU_ARCHS=native to an empty offload list inside vLLM
# workers and then compiles CK for the compiler default (gfx906), which fails.
# Pin the build target to the same architecture PyTorch targets.
export GPU_ARCHS="${PYTORCH_ROCM_ARCH}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export VLLM_ROCM_USE_AITER=1
# The strict paged materializer consumes AITER's packed NHD cache and rejects
# the optional shuffled physical layout.
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0
export VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
export RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE="${RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE:-32}"
export MIOPEN_DEBUG_CONV_DIRECT="${MIOPEN_DEBUG_CONV_DIRECT:-0}"

mkdir -p \
  "${RLK_ABLATION_ARM_DIR}/dump" \
  "${RL_KERNEL_READBACK_DIR}" \
  "${RL_KERNEL_MISMATCH_SIDECAR_DIR}"

SAVE_ARGS=()
if [[ "${RLK_ABLATION_SAVE_CHECKPOINT:-0}" == "1" ]]; then
  mkdir -p "${RLK_ABLATION_ARM_DIR}/checkpoint"
  SAVE_ARGS=(
    --save "${RLK_ABLATION_ARM_DIR}/checkpoint"
    --save-interval 1
  )
fi

python3 - <<'PY'
import os
import torch

expected = int(os.environ["RLK_ABLATION_NUM_GPUS"])
if torch.version.hip is None:
    raise SystemExit("PyTorch is not a ROCm build")
if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
    raise SystemExit(
        f"expected {expected} visible ROCm devices, got {torch.cuda.device_count()}"
    )
print(f"ROCm gate: HIP={torch.version.hip}, devices={torch.cuda.device_count()}")
PY

cd "${RLK_ABLATION_VIME_ROOT}"
# shellcheck source=/dev/null
source "${RLK_ABLATION_VIME_ROOT}/scripts/models/qwen3-8B.sh"

RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

names = [
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "PYTORCH_ROCM_ARCH",
    "GPU_ARCHS",
    "PYTORCH_ALLOC_CONF",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
    "HSA_NO_SCRATCH_RECLAIM",
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT",
    "VLLM_ATTENTION_BACKEND",
    "RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE",
    "MIOPEN_DEBUG_CONV_DIRECT",
    "RL_KERNEL_ATTENTION_CASE",
    "RL_KERNEL_FFN_CASE",
    "RL_KERNEL_LOGP_CASE",
    "RL_KERNEL_VLLM_REAL_VOCAB_SIZE",
    "RL_KERNEL_VLLM_PADDED_VOCAB_SIZE",
    "RL_KERNEL_VLLM_INTEGRATION",
    "RL_KERNEL_READBACK_DIR",
    "RL_KERNEL_MISMATCH_SIDECAR_DIR",
]
env_vars = {name: os.environ[name] for name in names}
for name in (
    "RL_KERNEL_ROCM_FIXED_PAGED_TILE",
    "RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS",
    "RL_KERNEL_DET_GEMM_BACKEND",
    "RL_KERNEL_ROCM_ATTENTION_BACKEND",
):
    if name in os.environ:
        env_vars[name] = os.environ[name]
print(json.dumps({"env_vars": env_vars}))
PY
)"

if ray status >/dev/null 2>&1; then
  echo "an existing Ray cluster is active; refusing to stop or reuse it" >&2
  exit 4
fi

ray_started=0
cleanup_ray() {
  if [[ "${ray_started}" == "1" ]]; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup_ray EXIT

ray start --head \
  --node-ip-address=127.0.0.1 \
  --port="${RLK_ABLATION_RAY_PORT}" \
  --num-gpus="${RLK_ABLATION_NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=127.0.0.1 \
  --dashboard-port="${RLK_ABLATION_RAY_DASHBOARD_PORT}" \
  --dashboard-agent-listen-port="${RLK_ABLATION_RAY_DASHBOARD_AGENT_PORT}"
ray_started=1

ray_job_address="http://127.0.0.1:${RLK_ABLATION_RAY_DASHBOARD_PORT}"
ray_job_agent_ready=0
for _attempt in {1..60}; do
  if ray job list --address="${ray_job_address}" >/dev/null 2>&1; then
    ray_job_agent_ready=1
    break
  fi
  sleep 1
done
if [[ "${ray_job_agent_ready}" != "1" ]]; then
  echo "Ray job agent did not become ready within 60 seconds" >&2
  exit 5
fi
# The dashboard endpoint can answer before the local node's job agent has
# finished registering. Give that registration a bounded stabilization window.
sleep 10

# The RL-Kernel paged CK path and device-sequenced IPC collectives are captured
# by vLLM's HIP graph runtime after adapter-owned warmup.
LINEAR_LOGP_ARGS=()
if [[ "${RL_KERNEL_LOGP_CASE%%/*}" == "R" ]]; then
  LINEAR_LOGP_ARGS+=(
    --linear-logp-provider
    rl_engine.integrations.vime.linear_logp_provider.provider
    --linear-logp-provider-mode strict
  )
fi
ROLLOUT_LOGPROBS_ARGS=()
if [[ "${RLK_ABLATION_USE_ROLLOUT_LOGPROBS:-0}" == "1" ]]; then
  ROLLOUT_LOGPROBS_ARGS+=(--use-rollout-logprobs)
fi
REFERENCE_MODEL_ARGS=()
if [[ "${RLK_ABLATION_USE_KL_LOSS:-0}" == "1" ]]; then
  : "${RLK_ABLATION_KL_LOSS_COEF:?}"
  REFERENCE_MODEL_ARGS+=(
    --use-kl-loss
    --kl-loss-coef "${RLK_ABLATION_KL_LOSS_COEF}"
  )
fi
ray job submit \
  --address="${ray_job_address}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  --working-dir="${RLK_ABLATION_VIME_ROOT}" \
  -- python3 "${RLK_ABLATION_VIME_ROOT}/train.py" \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${TRAIN_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  "${COLOCATE_ARGS[@]}" \
  --seed "${RLK_ABLATION_SEED}" \
  --rollout-seed "${RLK_ABLATION_ROLLOUT_SEED}" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${RLK_ABLATION_MODEL_ROOT}" \
  --ref-load "${RLK_ABLATION_REFERENCE_CHECKPOINT}" \
  "${REFERENCE_MODEL_ARGS[@]}" \
  --load "${RLK_ABLATION_REFERENCE_CHECKPOINT}" \
  --start-rollout-id 0 \
  "${SAVE_ARGS[@]}" \
  --prompt-data "${RLK_ABLATION_PROMPT_DATA}" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type deepscaler \
  --num-rollout "${RLK_ABLATION_NUM_ROLLOUT}" \
  --rollout-batch-size "${RLK_ABLATION_ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${RLK_ABLATION_SAMPLES_PER_PROMPT}" \
  --rollout-max-response-len "${RLK_ABLATION_MAX_RESPONSE_LENGTH}" \
  --rollout-temperature 1.0 \
  --rollout-top-p 1.0 \
  --global-batch-size "${RLK_ABLATION_GLOBAL_BATCH_SIZE}" \
  --balance-data \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --advantage-estimator grpo \
  --entropy-coef 0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --tensor-model-parallel-size "${RLK_ABLATION_TP_SIZE}" \
  --context-parallel-size "${RLK_ABLATION_CP_SIZE}" \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${RLK_ABLATION_MAX_TOKENS_PER_GPU}" \
  --router-policy "${RLK_ABLATION_ROUTER_POLICY}" \
  --rollout-num-gpus-per-engine "${RLK_ABLATION_ROLLOUT_TP_SIZE}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.4}" \
  --vllm-max-cudagraph-capture-size \
  "${RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE}" \
  --vllm-attention-backend ROCM_AITER_FA \
  --vllm-disable-custom-all-reduce \
  --attention-dropout 0 \
  --hidden-dropout 0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --train-memory-margin-bytes 2147483648 \
  --no-gradient-accumulation-fusion \
  "${LINEAR_LOGP_ARGS[@]}" \
  "${ROLLOUT_LOGPROBS_ARGS[@]}" \
  --get-mismatch-metrics \
  --custom-tis-function-path \
  vime_rocm_attention_ablation.tis_metrics.metrics_only_tis \
  --save-debug-rollout-data \
  "${RLK_ABLATION_ARM_DIR}/dump/rollout_data/{rollout_id}.pt" \
  --custom-megatron-init-path \
  rl_engine.integrations.megatron_runtime.initialize_from_environment
