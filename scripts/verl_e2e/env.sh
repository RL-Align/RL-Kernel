# SPDX-License-Identifier: Apache-2.0
# Pins and paths for the verl + vLLM batch-invariant DSv4 end-to-end test (Hopper, x86_64).
# Source this file on the host; every script in this directory reads it.
# Override any variable by exporting it before sourcing.

# --- Workspace on the host (needs ~600 GB free: model 160 GB, image 16 GB, outputs) ---
: "${E2E_ROOT:=$HOME/dsv4-e2e}"
export E2E_ROOT
export SRC_DIR="${E2E_ROOT}/src"          # verl and Megatron-LM checkouts
export MODEL_DIR="${E2E_ROOT}/models/DeepSeek-V4-Flash"
export DATA_DIR="${E2E_ROOT}/data"
export OUT_DIR="${E2E_ROOT}/outputs"

# --- Runtime image (vLLM BI fork, DeepGEMM, DeepEP, BI kernel library), from verl #7688 ---
export IMAGE_TAG="iseekyan/verl:ds4_vllm_align.preview-amd64"
export IMAGE_DIGEST="sha256:486b15962fd255e9a7fc5ea5162e0b05cb7f289eb080196e22faffb008d2c8a9"
export IMAGE="${IMAGE_TAG%%:*}@${IMAGE_DIGEST}"
export CONTAINER_NAME="${CONTAINER_NAME:-dsv4-e2e}"

# --- Source pins (mounted into the container at run time) ---
export VERL_REPO="https://github.com/ISEEKYAN/verl.git"
export VERL_REF="preview/deepseek-v4-megatron-true-on-policy"
export VERL_COMMIT="0c849d86175340b8d1141acda101d91c551493cb"       # verl PR #7688 head
export MEGATRON_REPO="https://github.com/ISEEKYAN/Megatron-LM.git"
export MEGATRON_REF="ds4_vllm_align_preview"
export MEGATRON_COMMIT="327a238239448de6771beea185dc6902a8de3eaa"

# --- Model and data ---
export MODEL_REPO="deepseek-ai/DeepSeek-V4-Flash"   # 43 layers, 256 experts, ~160 GB
export TRAIN_URL="https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet"
export VAL_URL="https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet"
export TRAIN_FILE="${DATA_DIR}/dapo-math-17k.parquet"
export VAL_FILE="${DATA_DIR}/aime-2024.parquet"

# --- Paths as the container sees them (bind mounts in launch_container.sh) ---
export C_E2E=/e2e
export C_VERL="${C_E2E}/src/verl"
export C_MEGATRON="${C_E2E}/src/Megatron-LM"
export C_MODEL="${C_E2E}/models/DeepSeek-V4-Flash"
export C_DATA="${C_E2E}/data"
export C_OUT="${C_E2E}/outputs"
export C_TOOLS="${C_E2E}/tools"
