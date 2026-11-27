#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# The full-model runs (README section 8; design doc section 10), one 8-GPU node, in order:
#   bash experiments.sh STAGE [STAGE ...]        DRY_RUN=1 prints the run.sh calls only
#
# Every stage: the real checkpoint's 43 layers (LAYERS), actor EP 8 with activation recompute,
# rollout DP 8 / EP 8, persist_w8 off, the DAPO math reward, wandb. Stages:
#   E2   rlkernel, BI, rollout CUDA graphs, W6 workload (32 x 8, response <= 2048), 100 steps,
#        constant lr 1e-4, checkpoint every 25.
#   BL2  verl #7483 (Megatron actor + vLLM 0.32 LoRA rollout), W7 workload, 50 steps.
#   E3   rlkernel, BI, graphs, W7 workload, 100 steps (a 100-step cosine), checkpoint every 25.
#   A3   E3 with the rollout's batch invariance off (run.sh ablation), A3_STEPS (default 50;
#        30 in the recorded run, to end before the allocation).
#   D41  DSv4.1 (MODEL_FAMILY=dsv41, the release's 40 layers), rlkernel, BI, eager rollout (no
#        graphs yet), W41 workload, 100 steps (a cosine), checkpoint every 25, no validation.
# W7 workload: 32 prompts x 16, prompt <= 1024, response <= 4096, AIME-2024 greedy before
# training and every 10 steps, lr 1e-4 with 3 warmup steps and a cosine to 1e-5, and the
# baseline launcher's PPO knobs in all runs (weight decay 0, clip 0.2 / 0.28, clip_c 10,
# token-mean). BL2's vLLM gets moe_backend=auto: verl's default (deep_gemm) has only an
# SM100 MXFP4 MoE.
#
# Each stage logs to $OUT_DIR/<stage>.log and appends "<stage> exit <rc> <date>" to
# $OUT_DIR/experiments.log. Set what env.sh / run.sh read (LORA_E2E_ROOT, CONTAINER,
# CUDA_COMPAT_DIR, ...) before calling.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$HERE/env.sh"
mkdir -p "$OUT_DIR"
WB='trainer.logger=["console","file","wandb"]'
PROJECT=${PROJECT:-dsv4_rlkernel}

stage() { # $1 stage, $2 run.sh variant, $3.. Hydra overrides; env: STEPS, GRAPH, workload
  local s=$1 v=$2; shift 2
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    echo "$s: STEPS=$STEPS GRAPH=${GRAPH:-0} TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}" \
      "ROLLOUT_N=${ROLLOUT_N:-8} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}" \
      "run.sh $v $*"
    return
  fi
  ( export LAYERS=${LAYERS:-43} RECOMPUTE=1 N_GPUS=8 ACTOR_EP=8 ROLLOUT_DP=8 REWARD=math
    export RL_KERNEL_DSV4_PERSIST_W8=0 GPU_UTIL=${GPU_UTIL:-0.4} STEPS GRAPH=${GRAPH:-0}
    export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray-$USER}-$s
    unset VIRTUAL_ENV
    echo "=== $s $(date): run.sh $v STEPS=$STEPS GRAPH=$GRAPH $*"
    bash "$HERE/run.sh" "$v" "$@" ) > "$OUT_DIR/$s.log" 2>&1
  echo "$s exit $? $(date)" | tee -a "$OUT_DIR/experiments.log"
}

W7=(trainer.test_freq=10 trainer.val_before_train=True
  actor_rollout_ref.rollout.val_kwargs.do_sample=False actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.temperature=0 actor_rollout_ref.rollout.val_kwargs.top_p=1
  actor_rollout_ref.rollout.val_kwargs.top_k=-1
  actor_rollout_ref.actor.optim.lr=1e-4 actor_rollout_ref.actor.optim.lr_warmup_steps=3
  actor_rollout_ref.actor.optim.weight_decay=0 actor_rollout_ref.actor.optim.clip_grad=1.0
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.clip_ratio_c=10.0 actor_rollout_ref.actor.loss_agg_mode=token-mean
  trainer.project_name="$PROJECT")
# The cosine: verl's own scheduler for the rlkernel engine, Megatron's for the baseline.
COS_RLK=(actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
  actor_rollout_ref.actor.optim.min_lr_ratio=0.1)
COS_MEGATRON=(actor_rollout_ref.actor.optim.lr_decay_style=cosine
  actor_rollout_ref.actor.optim.min_lr=1e-5)
# W41: 32 prompts x 8, prompt <= 1024, response <= 2048 (the eager DSv4.1 rollout decodes at
# about 0.33 s per step; W7's 4096 x 16 would not finish 100 steps in a 2-day allocation), W7's
# PPO knobs and cosine, no validation.
W41=(trainer.test_freq=-1 trainer.val_before_train=False)
for o in "${W7[@]}"; do
  case $o in trainer.test_freq=*|trainer.val_before_train=*|trainer.project_name=*) ;;
    *) W41+=("$o") ;;
  esac
done
w41() { ( export STEPS=$1 TRAIN_BATCH_SIZE=32 ROLLOUT_N=8 MAX_RESPONSE_LENGTH=2048 \
  TOKENS_PER_GPU=8192 OVERLONG=512 MAX_NUM_SEQS=64 MODEL_FAMILY=dsv41 LAYERS=40 \
  GPU_UTIL=${GPU_UTIL:-0.55}; unset MODEL_CUT; shift; "$@" ); }
w7() { ( export STEPS=$1 TRAIN_BATCH_SIZE=32 ROLLOUT_N=16 MAX_RESPONSE_LENGTH=4096 \
  TOKENS_PER_GPU=8192 OVERLONG=1024 MAX_NUM_SEQS=64; shift; "$@" ); }

for s in "$@"; do
  case $s in
    E2) STEPS=100 GRAPH=1 stage E2 rlkernel "$WB" trainer.project_name="$PROJECT" \
          trainer.experiment_name=rlkernel_L43_ep8_graph_100steps \
          trainer.save_freq=25 trainer.default_local_dir="$OUT_DIR/E2_ckpt" ;;
    BL2) w7 50 stage BL2 baseline "$WB" "${W7[@]}" "${COS_MEGATRON[@]}" \
          +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=auto \
          trainer.experiment_name=baseline7483_L43_n16_r4096_cos_50steps ;;
    E3) GRAPH=1 w7 100 stage E3 rlkernel "$WB" "${W7[@]}" "${COS_RLK[@]}" \
          trainer.experiment_name=rlkernel_L43_graph_n16_r4096_cos_100steps \
          trainer.save_freq=25 trainer.default_local_dir="$OUT_DIR/E3_ckpt" ;;
    A3) GRAPH=1 w7 "${A3_STEPS:-50}" stage A3 ablation "$WB" "${W7[@]}" "${COS_RLK[@]}" \
          trainer.experiment_name="rlkernel_nonbi_L43_graph_n16_r4096_cos_${A3_STEPS:-50}steps" \
          trainer.save_freq=25 trainer.default_local_dir="$OUT_DIR/A3_ckpt" ;;
    D41) w41 "${D41_STEPS:-100}" stage D41 rlkernel "$WB" "${W41[@]}" "${COS_RLK[@]}" \
          trainer.project_name=dsv41_rlkernel \
          trainer.experiment_name=rlkernel_dsv41_L40_n8_r2048_cos_100steps \
          trainer.save_freq=25 trainer.default_local_dir="$OUT_DIR/D41_ckpt" \
          trainer.resume_mode=auto ;;
    *) echo "unknown stage $s (E2 BL2 E3 A3 D41)"; exit 64 ;;
  esac
done
