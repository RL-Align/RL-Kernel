#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start (or reuse) a long-lived container with the workspace mounted at /e2e.
# Later steps run with:  docker exec -it dsv4-e2e bash /e2e/tools/<script>.sh ...
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "container ${CONTAINER_NAME} already running"
else
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker run -d --name "${CONTAINER_NAME}" \
    --gpus all --ipc=host --network host --shm-size=64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --cap-add SYS_PTRACE --cap-add IPC_LOCK \
    -v "${E2E_ROOT}:${C_E2E}" \
    -e HF_TOKEN="${HF_TOKEN:-}" -e WANDB_MODE="${WANDB_MODE:-disabled}" \
    -w "${C_E2E}" \
    "${IMAGE}" sleep infinity
fi
docker exec "${CONTAINER_NAME}" bash -lc '
  set -e
  cat /opt/ds4/image-manifest.json; echo
  test -s /opt/ds4/kernels/_vllm_batch_invariant_C.so && echo "BI kernel lib ok"
  /opt/ds4-venv/bin/python -c "import torch, vllm, deep_gemm; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"gpus\", torch.cuda.device_count()); print(\"vllm\", vllm.__version__); print(\"deep_gemm\", deep_gemm.__file__)"
  nvidia-smi -L
'
