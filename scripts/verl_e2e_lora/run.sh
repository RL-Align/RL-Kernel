#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One run of the 8-GPU DSv4 LoRA comparison (README.md). Outputs go to
# $OUT_DIR/<variant>_<stamp>/: train.log, metrics.jsonl (verl's per-step metrics), run_info.txt.
#
#   bash run.sh gate         # 1-2 GPUs: RL-Kernel's bitwise checks on the real checkpoint
#   bash run.sh rlkernel     # RL-Kernel + verl (Hopper, BI, LoRA)
#   bash run.sh control      # rlkernel with the adapter sync off (negative control, 3 steps)
#   bash run.sh ablation     # rlkernel with the rollout's batch invariance off (BI=0)
#   bash run.sh baseline     # verl #7483 (Megatron + vLLM LoRA, not batch-invariant)
#
# The shared workload below can be changed by exporting the variable; keep it the same for
# the runs you compare. Extra arguments are Hydra overrides for that run.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
VARIANT=${1:?usage: run.sh gate|rlkernel|control|ablation|baseline [hydra overrides]}
shift

# --- shared workload ---
export STEPS=${STEPS:-20}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}      # prompts per step
export ROLLOUT_N=${ROLLOUT_N:-8}                     # samples per prompt
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export LORA_RANK=${LORA_RANK:-16} LORA_ALPHA=${LORA_ALPHA:-32}
TOKENS_PER_GPU=${TOKENS_PER_GPU:-4096}               # micro-batch token budget (dynamic bsz)
OVERLONG=${OVERLONG:-512}
# Both runs: DAPO-Math prompts through #7483's DSv4 tokenization, the synthetic dense reward
# (a few-layer model scores 0 on math; this one moves the adapters every step) under the
# DAPO reward manager with the same overlong penalty.
COMMON=(
  data.prompt_key=prompt
  data.trust_remote_code=True
  +data.apply_chat_template_kwargs.enable_thinking=False
  +data.continuous_token.enable=True
  +data.continuous_token.model_family=deepseekv4
  reward.custom_reward_function.path=$RLK_ROOT/scripts/verl_e2e/custom_reward.py
  reward.custom_reward_function.name=compute_score
)
DAPO_REWARD=(
  reward.reward_manager.name=dapo
  +reward.reward_kwargs.overlong_buffer_cfg.enable=True
  +reward.reward_kwargs.overlong_buffer_cfg.len=$OVERLONG
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
  +reward.reward_kwargs.overlong_buffer_cfg.log=True
  +reward.reward_kwargs.max_resp_len=$MAX_RESPONSE_LENGTH
)

use_rlk_env() {
  if [[ -n ${SYSTEM_CUDA_HOME:-} ]]; then export CUDA_HOME=$SYSTEM_CUDA_HOME
  else export CUDA_HOME=$LORA_E2E_ROOT/envs/cuda130-home; fi
  export PATH=$CUDA_HOME/bin:$RLK_ENV/bin:$PATH
  export LD_LIBRARY_PATH=$RLK_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export RL_KERNEL_REQUIRE_EXT=1 PYTHONUNBUFFERED=1
}

stamp=$(date +%Y%m%d-%H%M%S)
RUN=$OUT_DIR/${VARIANT}_$stamp
mkdir -p "$RUN"
export VERL_FILE_LOGGER_PATH=$RUN/metrics.jsonl WANDB_MODE=${WANDB_MODE:-disabled}
info() {
  {
    echo "variant=$VARIANT model=$MODEL_CUT layers=$LAYERS steps=$STEPS"
    echo "batch=$TRAIN_BATCH_SIZE x $ROLLOUT_N prompt<=$MAX_PROMPT_LENGTH" \
      "response<=$MAX_RESPONSE_LENGTH"
    echo "lora rank=$LORA_RANK alpha=$LORA_ALPHA tokens/gpu=$TOKENS_PER_GPU overrides: $*"
    echo "verl $(git -C "$VERL_DIR" rev-parse HEAD) rl-kernel $(git -C "$RLK_ROOT" rev-parse HEAD)"
    python3 -c 'import importlib.metadata as m
for p in ("torch","vllm","verl","deep_gemm","flash_mla","megatron-core","megatron-bridge"):
    try: print(p, m.version(p))
    except Exception: pass'
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  } > "$RUN/run_info.txt" 2>&1
}

rlkernel() { # extra env / overrides for the RL-Kernel run
  use_rlk_env
  info "$@"
  MODEL_PATH=$MODEL_CUT DATA=$DATA_DIR N_GPUS=8 STEPS=$STEPS \
  LORA_RANK=$LORA_RANK LORA_ALPHA=$LORA_ALPHA \
  MAX_TOKEN_LEN_PER_GPU=$TOKENS_PER_GPU MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096} \
  MAX_NUM_SEQS=${MAX_NUM_SEQS:-64} GPU_UTIL=${GPU_UTIL:-0.35} ACTOR_LR=${ACTOR_LR:-1e-4} \
  LOGGER='["console","file"]' EXPERIMENT_NAME=$VARIANT \
    bash "$RLK_ROOT/scripts/verl_e2e/run_dsv4_rlkernel_grpo.sh" \
      "${COMMON[@]}" "${DAPO_REWARD[@]}" "$@" 2>&1 | tee "$RUN/train.log"
}

main() {
case $VARIANT in
  gate)
    use_rlk_env
    info
    cd "$RLK_ROOT"
    export PYTHONPATH=$RLK_ROOT
    set -x
    CUDA_VISIBLE_DEVICES=0 python scripts/check_dsv4_vllm_gate.py --ckpt "$MODEL_CUT" \
      --dump "$RUN/dump" --out "$RUN/vllm_gate.json" 2>&1 | tee "$RUN/vllm_gate.log"
    CUDA_VISIBLE_DEVICES=0 python scripts/check_dsv4_vllm_gate.py --ckpt "$MODEL_CUT" \
      --lora 16,32,7 --dump "$RUN/dump" --out "$RUN/vllm_gate_lora.json" 2>&1 |
      tee "$RUN/vllm_gate_lora.log"
    # The ablation's rollout (report only: its bytes differ by design).
    CUDA_VISIBLE_DEVICES=0 python scripts/check_dsv4_vllm_gate.py --ckpt "$MODEL_CUT" \
      --non-bi --lora 16,32,7 --dump "$RUN/dump" --out "$RUN/vllm_ablation.json" 2>&1 |
      tee "$RUN/vllm_ablation.log"
    CUDA_VISIBLE_DEVICES=0 python scripts/check_dsv4_layerwise_gate.py --ckpt "$MODEL_CUT" \
      --out "$RUN/layerwise_gate.json" 2>&1 | tee "$RUN/layerwise_gate.log"
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 scripts/check_dsv4_verl_engine.py \
      --model "$MODEL_CUT" 2>&1 | tee "$RUN/verl_engine.log"
    rm -rf "$RUN/dump"
    ;;
  rlkernel)
    rlkernel "$@"
    ;;
  control)
    # The rollout receives the adapters but keeps its old ones: the mismatch count must turn
    # nonzero from step 2 (step 1 runs the zero-B adapters, equal to the base model).
    export RL_KERNEL_DSV4_SKIP_ADAPTER_SYNC=1 STEPS=${CONTROL_STEPS:-3}
    rlkernel "$@"
    ;;
  ablation)
    # The same run with the rollout plugin's non-invariant mode: vLLM's own attention paths
    # and paged FP8 decode, RL-Kernel's model otherwise. Mismatches are expected; the run
    # measures what batch invariance costs the rollout.
    export BI=0
    rlkernel "$@"
    ;;
  baseline)
    # verl #7483's own launcher, unmodified except the logger (wandb -> file), on one node:
    # actor EP 8 (no PP / CP), rollout TP 8 / EP 8, the same workload as the rlkernel run.
    # shellcheck disable=SC1091
    source "$BASELINE_VENV/bin/activate"
    info "$@"
    cp "$VERL_DIR/$BASELINE_LAUNCHER" "$RUN/launcher.sh"
    sed -i 's/trainer.logger=.\["console","wandb"\]./trainer.logger='"'"'["console","file"]'"'"'/' \
      "$RUN/launcher.sh"
    grep -q '"console","file"' "$RUN/launcher.sh" || { echo "logger edit failed"; exit 1; }
    cd "$VERL_DIR"
    VERL_USE_UV=0 MODEL_PATH=$MODEL_CUT NNODES=1 NGPUS_PER_NODE=8 \
    TRAIN_FILE=$DATA_DIR/train.parquet TEST_FILE=$DATA_DIR/test.parquet \
    TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE=$TRAIN_BATCH_SIZE ROLLOUT_N=$ROLLOUT_N \
    MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH \
    PPO_MAX_TOKEN_LEN_PER_GPU=$TOKENS_PER_GPU OVERLONG_BUFFER_LEN=$OVERLONG \
    LORA_RANK=$LORA_RANK LORA_ALPHA=$LORA_ALPHA \
    ACTOR_PP=1 ACTOR_CP=1 ACTOR_EP=8 ROLLOUT_TP=8 ROLLOUT_EP=8 \
    TOTAL_TRAINING_STEPS=$STEPS TEST_FREQ=-1 SAVE_FREQ=-1 \
    EXPERIMENT_NAME=$VARIANT CKPTS_DIR=$RUN/ckpt \
      bash "$RUN/launcher.sh" \
        reward.custom_reward_function.path="$RLK_ROOT/scripts/verl_e2e/custom_reward.py" \
        reward.custom_reward_function.name=compute_score "$@" 2>&1 | tee "$RUN/train.log"
    ;;
  *) echo "unknown variant $VARIANT"; return 64 ;;
esac
}

# Keep going after a failure: record the exit code and summarize what was logged.
set +e
main "$@"
rc=$?
set -e
echo "exit code $rc" | tee -a "$RUN/run_info.txt"
[[ -s $RUN/metrics.jsonl ]] && python3 "$HERE/compare.py" "$RUN" || true
exit "$rc"
