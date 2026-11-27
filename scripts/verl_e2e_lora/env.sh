# SPDX-License-Identifier: Apache-2.0
# Pins and paths for the 8-GPU DSv4 LoRA comparison (README.md). Source it; every script here
# reads it. Export a variable before sourcing to override it.

# --- Workspace (~400 GB free: checkpoint ~160 GB, two envs, source builds, outputs) ---
: "${LORA_E2E_ROOT:=$HOME/dsv4-lora-e2e}"
export LORA_E2E_ROOT
export SRC_DIR=$LORA_E2E_ROOT/src
# MODEL_FAMILY=dsv41: DeepSeek-V4.1-Flash on vLLM main (env RLK41_ENV); LAYERS=40 is the
# release itself.
export MODEL_FAMILY=${MODEL_FAMILY:-dsv4}
export DATA_DIR=$LORA_E2E_ROOT/data
export OUT_DIR=${OUT_DIR:-$LORA_E2E_ROOT/outputs}
if [[ $MODEL_FAMILY == dsv41 ]]; then
  export MODEL_DIR=$LORA_E2E_ROOT/models/DeepSeek-V4.1-Flash
  export LAYERS=${LAYERS:-40}
  if [[ -n ${MODEL_CUT:-} ]]; then :  # set by the caller (e.g. the toy checkpoint)
  elif [[ $LAYERS == 40 ]]; then export MODEL_CUT=$MODEL_DIR
  else export MODEL_CUT=$MODEL_DIR-L$LAYERS; fi
else
  export MODEL_DIR=$LORA_E2E_ROOT/models/DeepSeek-V4-Flash
  export LAYERS=${LAYERS:-4}          # first N layers of the checkpoint (README section 0)
  export MODEL_CUT=$LORA_E2E_ROOT/models/DeepSeek-V4-Flash-L$LAYERS
fi

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
# The launcher's apply_dsa_kernel_fusion=True needs these two (verl's Dockerfile builds them;
# `uv sync` does not): FlashMLA's nv_dev branch (`indexer_topk`), and the indexer's Hadamard.
export BL_FLASHMLA_REPO=https://github.com/deepseek-ai/FlashMLA.git
export BL_FLASHMLA_COMMIT=b7643bd54521f563b839b98289b5cd048c062ba2   # nv_dev, 2026-11-20
export HADAMARD_REPO=https://github.com/Dao-AILab/fast-hadamard-transform.git
export HADAMARD_COMMIT=e7706faf8d1c3b9f241e36860640ad1dac644ede

# --- RL-Kernel env (conda, Python 3.12, torch 2.13 + CUDA 13.0) ---
export RLK_ENV=${RLK_ENV:-$LORA_E2E_ROOT/envs/rlk}
# DSv4.1: torch 2.14, vLLM main 3f86db3 (editable from $SRC_DIR/vllm-main), verl as above
export RLK41_ENV=${RLK41_ENV:-$LORA_E2E_ROOT/envs/rlk41}
if [[ $MODEL_FAMILY == dsv41 ]]; then export RLK_ENV=$RLK41_ENV; fi
export VLLM_FORK_REPO=https://github.com/ISEEKYAN/vllm.git
export VLLM_FORK_COMMIT=1a6542cdebbaabbf9972c358de8d6baa54e46b0c   # vLLM #54955 lineage
# vLLM's DS4 deterministic top-k, built by setup.sh (the BI rollout loads it)
export DS4_BI_TOPK_LIB=${DS4_BI_TOPK_LIB:-$LORA_E2E_ROOT/build/ds4_bi_topk/ds4_bi_topk.so}
export DEEPGEMM_REPO=https://github.com/vllm-project/DeepGEMM.git
export DEEPGEMM_COMMIT=e21c821f
export FLASHMLA_REPO=https://github.com/vllm-project/FlashMLA.git
export FLASHMLA_COMMIT=a8f794d1
export TORCH_INDEX=https://download.pytorch.org/whl/cu130
export VERL_WHEELHOUSE=https://verl-project.github.io/verl-wheelhouse/cu130/torch2.13/simple
# A driver older than R580 cannot run CUDA 13. On data-center GPUs, NVIDIA's forward-compat
# libcuda works instead: `conda create -p DIR -c conda-forge cuda-compat=13.0`, then
# CUDA_COMPAT_DIR=DIR/cuda-compat. Both envs and every run put it first on LD_LIBRARY_PATH.
export CUDA_COMPAT_DIR=${CUDA_COMPAT_DIR:-}
if [[ -n $CUDA_COMPAT_DIR ]]; then
  export LD_LIBRARY_PATH=$CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
# RL-Kernel itself: this checkout (the directory above scripts/).
RLK_ROOT=${RLK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export RLK_ROOT
