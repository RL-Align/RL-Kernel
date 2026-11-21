#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Host-side setup: preflight checks, image pull, pinned checkouts, model and data download,
# and the bitwise log-prob metric patch for verl. Idempotent; rerun after a failure.
#
#   bash scripts/verl_e2e/setup.sh            # everything
#   SKIP_MODEL=1 bash scripts/verl_e2e/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

log() { printf '\n=== %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

log "preflight"
command -v docker >/dev/null || die "docker not found (see README: enroot / apptainer alternatives)"
command -v git >/dev/null || die "git not found"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
ngpu=$(nvidia-smi -L | wc -l)
(( ngpu >= 8 )) || echo "warning: ${ngpu} GPUs visible; the 8-GPU tests need 8"
nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | grep -qx '9.0' ||
  die "this guide targets Hopper (SM90: H100 / H200); see README for GB200"
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
(( drv >= 580 )) || die "driver ${drv} < 580: the image uses CUDA 13 (needs R580+)"
nvidia-smi topo -m | head -12
free -g | awk 'NR<=2'
mkdir -p "${E2E_ROOT}" "${SRC_DIR}" "${DATA_DIR}" "${OUT_DIR}" "$(dirname "${MODEL_DIR}")"
df -h "${E2E_ROOT}" | tail -1
docker info --format '{{json .Runtimes}}' | grep -q nvidia ||
  echo "warning: docker has no 'nvidia' runtime listed; --gpus all needs the NVIDIA Container Toolkit"

log "image ${IMAGE}"
docker pull "${IMAGE}"

clone_pinned() {   # dir repo ref commit
  local dir=$1 repo=$2 ref=$3 commit=$4
  if [[ ! -d "${dir}/.git" ]]; then
    git clone --filter=blob:none --branch "${ref}" "${repo}" "${dir}"
  fi
  git -C "${dir}" fetch --quiet origin "${ref}" || true
  git -C "${dir}" checkout --quiet --detach "${commit}"
  [[ "$(git -C "${dir}" rev-parse HEAD)" == "${commit}" ]] || die "${dir} not at ${commit}"
  echo "${dir} @ ${commit}"
}
log "sources"
clone_pinned "${SRC_DIR}/verl" "${VERL_REPO}" "${VERL_REF}" "${VERL_COMMIT}"
clone_pinned "${SRC_DIR}/Megatron-LM" "${MEGATRON_REPO}" "${MEGATRON_REF}" "${MEGATRON_COMMIT}"

log "verl: bitwise log-prob metric"
python3 "${HERE}/patch_verl_bitwise_metric.py" "${SRC_DIR}/verl"

log "tools"
mkdir -p "${E2E_ROOT}/tools"
cp "${HERE}"/*.sh "${HERE}"/*.py "${HERE}/../verl_e2e_lora/custom_reward.py" "${E2E_ROOT}/tools/"

log "data"
[[ -s "${TRAIN_FILE}" ]] || curl -fL -o "${TRAIN_FILE}" "${TRAIN_URL}"
[[ -s "${VAL_FILE}" ]] || curl -fL -o "${VAL_FILE}" "${VAL_URL}"
ls -la "${DATA_DIR}"

if [[ "${SKIP_MODEL:-0}" != 1 ]]; then
  log "model ${MODEL_REPO} -> ${MODEL_DIR} (~160 GB)"
  if command -v hf >/dev/null; then
    hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
  elif command -v huggingface-cli >/dev/null; then
    huggingface-cli download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
  else
    # The image has huggingface_hub; run the download there.
    docker run --rm -v "${E2E_ROOT}:${C_E2E}" -e HF_TOKEN="${HF_TOKEN:-}" "${IMAGE}" \
      python3 -c "from huggingface_hub import snapshot_download as s; s('${MODEL_REPO}', local_dir='${C_MODEL}')"
  fi
  python3 - "${MODEL_DIR}" <<'EOF'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
missing = sorted({f for f in idx["weight_map"].values() if not os.path.exists(os.path.join(d, f))})
cfg = json.load(open(os.path.join(d, "config.json")))
print("layers", cfg["num_hidden_layers"], "experts", cfg["n_routed_experts"], "missing shards", missing)
sys.exit(1 if missing else 0)
EOF
fi
log "setup done; next: bash ${HERE}/launch_container.sh"
