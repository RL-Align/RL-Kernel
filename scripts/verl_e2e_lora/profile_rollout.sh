#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Decode latency of the rollout's vLLM server alone, under nsys (design doc section 10.3).
# One 8-GPU node, one mode per call; the vLLM arguments are the ones verl passed in E3 / A3 /
# BL2 (README section 8):
#   bash profile_rollout.sh bi|nonbi|bl
#     bi     RL-Kernel plugin, batch-invariant (E3): DP 8 / EP 8, graphs, adapters reserved
#     nonbi  the plugin with batch invariance off (A3)
#     bl     vLLM 0.32 as verl #7483 runs it (BL2): TP 8 / EP 8, no LoRA adapter (verl syncs
#            it during training), so slightly optimistic for the baseline
# prof_client.py times C concurrent requests of exactly 256 tokens each (C = 1, 8, 64, 512),
# then profiles PROF_PHASES (default "1:16,64:16", concurrency:tokens). Env:
#   NSYS (nsys >= 2025 for CUDA 13; default `nsys`), GTRACE=node|graph (graph granularity,
#   default node: every kernel inside a replayed graph), TAG (output suffix),
#   PROF_ONLY=1 (skip the timed phases), PROF_DIR (default $OUT_DIR/prof), LAYERS (env.sh).
# Output: $PROF_DIR/<mode><TAG>.{nsys-rep,json,client.log,server.log}.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$HERE/env.sh"
if [[ -n ${CONTAINER:-} && -z ${SINGULARITY_CONTAINER:-} ]]; then  # as run.sh
  libs=${CUDA_COMPAT_DIR:+$CUDA_COMPAT_DIR:}/.singularity.d/libs
  unset VIRTUAL_ENV
  exec env -u CC -u CXX "SINGULARITYENV_LD_LIBRARY_PATH=$libs" \
    singularity exec --nv -B "$LORA_E2E_ROOT" -B "$RLK_ROOT" "$CONTAINER" bash "$0" "$@"
fi
MODE=${1:?usage: profile_rollout.sh bi|nonbi|bl}
OUT=$MODE${TAG:-}
P=${PROF_DIR:-$OUT_DIR/prof}
mkdir -p "$P"
NSYS=${NSYS:-nsys}
M=$MODEL_CUT  # LAYERS=43 for the full model (env.sh)
PORT=${PORT:-18123}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} VLLM_USE_V1=1
export NCCL_NVLS_ENABLE=0 TORCH_NCCL_AVOID_RECORD_STREAMS=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
COMMON=(serve "$M" --port "$PORT" --dtype bfloat16 --load_format auto --trust_remote_code
  --distributed_executor_backend mp --max_model_len 5120 --enable_chunked_prefill
  --enable_prefix_caching --enable_sleep_mode --seed 42
  --compilation_config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' --kv_cache_dtype fp8
  --linear_backend deep_gemm --enable_expert_parallel --profiler-config '{"profiler": "cuda"}')
case $MODE in
  bi|nonbi)
    if [[ -n ${SYSTEM_CUDA_HOME:-} ]]; then export CUDA_HOME=$SYSTEM_CUDA_HOME
    else export CUDA_HOME=$LORA_E2E_ROOT/envs/cuda130-home; fi
    export PATH=$CUDA_HOME/bin:$RLK_ENV/bin:$PATH
    export LD_LIBRARY_PATH=${CUDA_COMPAT_DIR:+$CUDA_COMPAT_DIR:}$RLK_ENV/lib:${LD_LIBRARY_PATH:-}
    export PYTHONPATH=$RLK_ROOT RL_KERNEL_REQUIRE_EXT=1 RL_KERNEL_DSV4_PERSIST_W8=0
    export RL_KERNEL_DSV4_ROLLOUT=1 RL_KERNEL_DSV4_LORA=16,32
    if [[ $MODE == bi ]]; then
      export VLLM_BATCH_INVARIANT=1 VLLM_DS4_DECODE_KERNEL=sparse RL_KERNEL_DSV4_NON_BI=0
    else
      export VLLM_BATCH_INVARIANT=0 VLLM_DS4_DECODE_KERNEL=paged RL_KERNEL_DSV4_NON_BI=1
    fi
    PY=$RLK_ENV/bin/python
    ARGS=("${COMMON[@]}" --max_num_seqs 64 --max_num_batched_tokens 4096
      --gpu_memory_utilization 0.4 --tensor_parallel_size 1 --data_parallel_size 8
      --worker_cls rl_engine.integrations.dsv4.vllm_worker.RlkWorker
      --worker_extension_cls rl_engine.integrations.dsv4.vllm_worker.RlkWorkerExtension
      --no-enable_flashinfer_autotune --moe_backend deep_gemm) ;;
  bl)
    export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export VIRTUAL_ENV=$BASELINE_VENV PATH=$BASELINE_VENV/bin:$PATH  # ninja, for JIT builds
    PY=$BASELINE_VENV/bin/python
    ARGS=("${COMMON[@]}" --max_num_seqs 256 --max_num_batched_tokens 5120
      --gpu_memory_utilization 0.5 --tensor_parallel_size 8 --decode_context_parallel_size 1
      --moe_backend auto --enable_return_routed_experts) ;;
  *) echo "mode bi|nonbi|bl"; exit 64 ;;
esac
echo "=== $OUT $(date)"
"$NSYS" profile -o "$P/$OUT" -f true -t cuda,nvtx,osrt --cuda-graph-trace="${GTRACE:-node}" \
  --capture-range=cudaProfilerApi --capture-range-end=stop --trace-fork-before-exec=true \
  "$PY" -m vllm.entrypoints.cli.main "${ARGS[@]}" > "$P/$OUT.server.log" 2>&1 &
srv=$!
for _ in $(seq 180); do  # up to 30 min to load and capture
  "$PY" -c "import urllib.request as u; u.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" \
    2>/dev/null && break
  kill -0 $srv 2>/dev/null || { echo "server died"; tail -30 "$P/$OUT.server.log"; exit 1; }
  sleep 10
done
echo "ready $(date)"
"$PY" "$HERE/prof_client.py" --port "$PORT" --model "$M" --out "$P/$OUT.json" --profile 2>&1 \
  | tee "$P/$OUT.client.log"
sleep 20
kill -INT $srv; wait $srv
echo "done $(date)"; ls -la "$P/$OUT".*
