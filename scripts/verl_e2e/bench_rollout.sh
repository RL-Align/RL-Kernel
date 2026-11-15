#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Rollout-only speed and batch-invariance of vLLM on 8 GPUs (DP8 + EP8), INSIDE the container.
#
#   bash /e2e/tools/bench_rollout.sh bi       # VLLM_BATCH_INVARIANT=1, the verl recipe's settings
#   bash /e2e/tools/bench_rollout.sh nonbi    # same weights (FP8 experts), BI off, default kernels
#   bash /e2e/tools/bench_rollout.sh native   # BI off, checkpoint's own MXFP4 experts (production)
#
# Env: MODEL (default: the full checkpoint), CONCURRENCY ("1 8 32 128 256"), IN_LEN / OUT_LEN
# (1024 / 1024), MAX_MODEL_LEN (8192), GPU_UTIL (0.85), PORT (8000).
# Each variant: start server -> probe_server_bi.py -> vllm bench serve sweep -> stop server.
# Results: /e2e/outputs/rollout_<variant>_<stamp>/
set -euo pipefail
VARIANT="${1:?usage: bench_rollout.sh bi|nonbi|native}"
source /e2e/tools/env.sh
MODEL="${MODEL:-${C_MODEL}}"
PORT="${PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-1 8 32 128 256}"
IN_LEN="${IN_LEN:-1024}" OUT_LEN="${OUT_LEN:-1024}"
OUT="${C_OUT}/rollout_${VARIANT}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "${OUT}"

# Same runtime environment the verl recipe gives vLLM (single node, DeepEP NVL8 overlay).
export PATH="/opt/ds4-venv/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib.real:/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/ds4/overlays/deepep-nvl8/site-packages:${PYTHONPATH:-}"
export NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_MAX_TEAMS=7 NVSHMEM_DISABLE_NCCL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=1024 DEEPEP_MAX_NVL_PEERS=8 PYTHONHASHSEED=42

FP8_EXPERTS='{"expert_dtype":"fp8","quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","scale_fmt":"ue8m0","weight_block_size":[128,128]}}'
ARGS=(--served-model-name dsv4 --trust-remote-code --port "${PORT}"
      --data-parallel-size 8 --enable-expert-parallel
      --kv-cache-dtype fp8 --max-model-len "${MAX_MODEL_LEN:-8192}"
      --gpu-memory-utilization "${GPU_UTIL:-0.85}" --disable-custom-all-reduce --seed 42)
case "${VARIANT}" in
  bi)
    export VLLM_BATCH_INVARIANT=1 VLLM_DS4_DECODE_KERNEL=sparse
    export VLLM_BATCH_INVARIANT_KERNEL_LIB=/opt/ds4/kernels/_vllm_batch_invariant_C.so
    ARGS+=(--hf-overrides "${FP8_EXPERTS}" --moe-backend deep_gemm --linear-backend deep_gemm
           --all2all-backend deepep_low_latency)
    ;;
  nonbi)
    export VLLM_BATCH_INVARIANT=0 VLLM_DS4_DECODE_KERNEL=paged
    unset VLLM_BATCH_INVARIANT_KERNEL_LIB
    ARGS+=(--hf-overrides "${FP8_EXPERTS}" --all2all-backend deepep_low_latency)
    ;;
  native)
    export VLLM_BATCH_INVARIANT=0 VLLM_DS4_DECODE_KERNEL=paged
    unset VLLM_BATCH_INVARIANT_KERNEL_LIB
    ;;
  *) echo "unknown variant ${VARIANT}"; exit 64 ;;
esac
printf '%q ' vllm serve "${MODEL}" "${ARGS[@]}" > "${OUT}/server_cmd.txt"; echo >> "${OUT}/server_cmd.txt"
env | grep -E '^(VLLM_|NVSHMEM|DEEPEP)' | sort > "${OUT}/server_env.txt"

vllm serve "${MODEL}" "${ARGS[@]}" > "${OUT}/server.log" 2>&1 &
SERVER=$!
trap 'kill ${SERVER} 2>/dev/null || true; wait ${SERVER} 2>/dev/null || true' EXIT
echo "waiting for the server (weights load + JIT warmup can take 20-40 min); log: ${OUT}/server.log"
for _ in $(seq 1 480); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null && break
  kill -0 ${SERVER} 2>/dev/null || { echo "server exited"; tail -50 "${OUT}/server.log"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "server not ready"; exit 1; }

echo "=== batch-invariance probe"
python3 /e2e/tools/probe_server_bi.py --port "${PORT}" --out "${OUT}/probe.json" | tail -1 || true

echo "=== throughput sweep (random ${IN_LEN} in / ${OUT_LEN} out)"
for c in ${CONCURRENCY}; do
  n=$(( c * 4 > 32 ? c * 4 : 32 ))
  vllm bench serve --backend vllm --model "${MODEL}" --served-model-name dsv4 --port "${PORT}" \
    --dataset-name random --random-input-len "${IN_LEN}" --random-output-len "${OUT_LEN}" \
    --num-prompts "${n}" --max-concurrency "${c}" --ignore-eos --seed 42 \
    --save-result --result-dir "${OUT}" --result-filename "bench_c${c}.json" \
    > "${OUT}/bench_c${c}.log" 2>&1 || echo "bench at concurrency ${c} failed (see log)"
  python3 - "${OUT}/bench_c${c}.json" "${c}" <<'EOF' || true
import json, sys
r = json.load(open(sys.argv[1]))
print(f"c={sys.argv[2]:>4}  out tok/s {r.get('output_throughput', 0):9.1f}  "
      f"TPOT p50 {r.get('median_tpot_ms', 0):7.2f} ms  TTFT p50 {r.get('median_ttft_ms', 0):8.1f} ms")
EOF
done | tee "${OUT}/summary.txt"
echo "results in ${OUT}"
