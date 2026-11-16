# SPDX-License-Identifier: Apache-2.0
# Pins and paths for the 8-GPU DSv4 LoRA comparison (README.md). Source it; every script here
# reads it. Export a variable before sourcing to override it.

# --- Workspace (~400 GB free: checkpoint ~160 GB, two envs, source builds, outputs) ---
: "${LORA_E2E_ROOT:=$HOME/dsv4-lora-e2e}"
export LORA_E2E_ROOT
export SRC_DIR=$LORA_E2E_ROOT/src
export MODEL_DIR=$LORA_E2E_ROOT/models/DeepSeek-V4-Flash
export DATA_DIR=$LORA_E2E_ROOT/data
export OUT_DIR=$LORA_E2E_ROOT/outputs
export LAYERS=${LAYERS:-4}            # first N layers of the checkpoint (README section 0)
export MODEL_CUT=$LORA_E2E_ROOT/models/DeepSeek-V4-Flash-L$LAYERS

# --- Model and data ---
export MODEL_REPO=deepseek-ai/DeepSeek-V4-Flash
HF_DATA=https://huggingface.co/datasets/BytedTsinghua-SIA
export TRAIN_URL=$HF_DATA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet
export VAL_URL=$HF_DATA/AIME-2024/resolve/main/data/aime-2024.parquet

# --- verl: one commit for both runs, verl #7483's head (it contains the commit the rlkernel
#     engine was tested on, 4b39670, and adds the DSv4 data / tokenizer handling both use) ---
export VERL_REPO=https://github.com/verl-project/verl.git
export VERL_COMMIT=8c047f4d24f333c9106feefa527f711679878ddc       # PR #7483 head
export VERL_TESTED_COMMIT=4b396702074369ae977db99387bd8470f02f62b9  # rlkernel engine, 2 GPUs

# --- Baseline (verl #7483: Megatron actor + vLLM 0.32 LoRA, not batch-invariant) ---
# The launcher's header names these two; verl's uv.lock alone gives Bridge 0.6.2 / core 0.19.2.
export BRIDGE_REPO=https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
export BRIDGE_COMMIT=c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11
export MEGATRON_REPO=https://github.com/NVIDIA/Megatron-LM.git
export MEGATRON_COMMIT=fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1
export VERL_DIR=$SRC_DIR/verl                 # one clone: the baseline's uv venv lives in it,
export BASELINE_VENV=$VERL_DIR/.venv          # the RL-Kernel env installs it editable
export BASELINE_LAUNCHER=examples/tuning/lora/run_deepseek_v4_flash_megatron.sh

# --- RL-Kernel env (conda, Python 3.12, torch 2.13 + CUDA 13.0) ---
export RLK_ENV=${RLK_ENV:-$LORA_E2E_ROOT/envs/rlk}
export VLLM_FORK_REPO=https://github.com/ISEEKYAN/vllm.git
export VLLM_FORK_COMMIT=1a6542cdebbaabbf9972c358de8d6baa54e46b0c   # vLLM #54955 lineage
export DEEPGEMM_REPO=https://github.com/vllm-project/DeepGEMM.git
export DEEPGEMM_COMMIT=e21c821f
export FLASHMLA_REPO=https://github.com/vllm-project/FlashMLA.git
export FLASHMLA_COMMIT=a8f794d1
export TORCH_INDEX=https://download.pytorch.org/whl/cu130
export VERL_WHEELHOUSE=https://verl-project.github.io/verl-wheelhouse/cu130/torch2.13/simple
# RL-Kernel itself: this checkout (the directory above scripts/).
RLK_ROOT=${RLK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export RLK_ROOT
