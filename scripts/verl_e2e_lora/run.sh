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
# A host with glibc < 2.38 (RHEL 8 / 9) cannot load the flash-attn / TransformerEngine wheels
# of either env: with CONTAINER set to an Ubuntu 24.04 CUDA 13 image, rerun inside it.
if [[ -n ${CONTAINER:-} && -z ${SINGULARITY_CONTAINER:-} ]]; then
  libs=${CUDA_COMPAT_DIR:+$CUDA_COMPAT_DIR:}/.singularity.d/libs  # before --nv's host libcuda
  # CC / CXX may name a host toolchain (an Lmod module) that is not in the image; Triton's JIT
  # would call it. Unset, the image's gcc is used.
  exec env -u CC -u CXX "SINGULARITYENV_LD_LIBRARY_PATH=$libs" \
    singularity exec --nv -B "$LORA_E2E_ROOT" -B "$RLK_ROOT" "$CONTAINER" bash "$0" "$@"
fi
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
# Both runs: DAPO-Math prompts through #7483's DSv4 tokenization under the DAPO reward manager
# with the same overlong penalty. REWARD picks the score:
# - synthetic (default): the fraction of digits in the response (custom_reward.py). A
#   few-layer model scores 0 on math; this one moves the adapters every step, which is what
#   the bitwise check needs.
# - math: verl's own scorer for the data's data_source (math_dapo: the "Answer:" line
#   against the ground truth), for the full model.
REWARD=${REWARD:-synthetic}
[[ $MODEL_FAMILY == dsv41 ]] && TOKEN_FAMILY=deepseekv41 || TOKEN_FAMILY=deepseekv4
COMMON=(
  data.prompt_key=prompt
  data.trust_remote_code=True
  +data.apply_chat_template_kwargs.enable_thinking=False
  +data.continuous_token.enable=True
  +data.continuous_token.model_family=$TOKEN_FAMILY
)
case $REWARD in
  synthetic) REWARD_FN=(
      reward.custom_reward_function.path=$HERE/custom_reward.py
      reward.custom_reward_function.name=compute_score) ;;
  math) REWARD_FN=(reward.custom_reward_function.path=null) ;;
  *) echo "REWARD must be synthetic or math"; exit 64 ;;
esac
COMMON+=("${REWARD_FN[@]}")
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
  elif [[ $MODEL_FAMILY == dsv41 ]]; then export CUDA_HOME=$LORA_E2E_ROOT/envs/cuda130-home41
  else export CUDA_HOME=$LORA_E2E_ROOT/envs/cuda130-home; fi
  export PATH=$CUDA_HOME/bin:$RLK_ENV/bin:$PATH
  # $RLK_ENV/lib holds the toolkit's libcuda stub; the compat libcuda must come before it.
  local lib=$RLK_ENV/lib
  if [[ -n $CUDA_COMPAT_DIR ]]; then lib=$CUDA_COMPAT_DIR:$lib; fi
  export LD_LIBRARY_PATH=$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
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
    echo "lora rank=$LORA_RANK alpha=$LORA_ALPHA tokens/gpu=$TOKENS_PER_GPU reward=$REWARD" \
      "overrides: $*"
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
  MODEL_PATH=$MODEL_CUT DATA=$DATA_DIR N_GPUS=${N_GPUS:-8} STEPS=$STEPS \
  LORA_RANK=$LORA_RANK LORA_ALPHA=$LORA_ALPHA \
  MAX_TOKEN_LEN_PER_GPU=$TOKENS_PER_GPU MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096} \
  MAX_NUM_SEQS=${MAX_NUM_SEQS:-64} GPU_UTIL=${GPU_UTIL:-0.35} ACTOR_LR=${ACTOR_LR:-1e-4} \
  LOGGER='["console","file"]' EXPERIMENT_NAME=$VARIANT \
    bash "$HERE/run_dsv4_rlkernel_grpo.sh" \
      "${COMMON[@]}" "${DAPO_REWARD[@]}" "$@" 2>&1 | tee "$RUN/train.log"
}

BASELINE_APPEND_DEFAULT="actor_rollout_ref.model.lora.lora_plus_ratio"
# Dropped:
# - rollout.use_dynamic_bsz: RolloutConfig refuses it (the launcher's next line sets
#   log_prob_use_dynamic_bsz).
# - lora.experts_shared_outer_loras: its adapters (fc1 A / fc2 B shared, the other side per
#   expert) are SGLang's format; verl does not turn on vLLM's enable_moe_shared_loras, and
#   vLLM 0.32 fails on the first sync. Without it Bridge shares one adapter by all experts
#   (share_expert_adapters, the default), as RL-Kernel's engine does.
BASELINE_DROP_DEFAULT="actor_rollout_ref.rollout.use_dynamic_bsz
  actor_rollout_ref.model.lora.experts_shared_outer_loras"

main() {
case $VARIANT in
  gate)
    # vLLM's share of the GPU in the vLLM gates: 0.4 of an H200 holds the 4-layer view; an
    # H100 80 GB needs about 0.8.
    GATE_GPU_UTIL=${GATE_GPU_UTIL:-0.4}
    vgate=(python scripts/check_dsv4_vllm_gate.py --ckpt "$MODEL_CUT" --gpu-util "$GATE_GPU_UTIL")
    use_rlk_env
    info
    cd "$RLK_ROOT"
    export PYTHONPATH=$RLK_ROOT
    set -x
    CUDA_VISIBLE_DEVICES=0 "${vgate[@]}" --dump "$RUN/dump" --out "$RUN/vllm_gate.json" \
      2>&1 | tee "$RUN/vllm_gate.log"
    CUDA_VISIBLE_DEVICES=0 "${vgate[@]}" --lora 16,32,7 --dump "$RUN/dump" \
      --out "$RUN/vllm_gate_lora.json" 2>&1 | tee "$RUN/vllm_gate_lora.log"
    # The ablation's rollout (report only: its bytes differ by design).
    CUDA_VISIBLE_DEVICES=0 "${vgate[@]}" --non-bi --lora 16,32,7 --dump "$RUN/dump" \
      --out "$RUN/vllm_ablation.json" 2>&1 | tee "$RUN/vllm_ablation.log"
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
    export RL_KERNEL_DSV4_SKIP_ADAPTER_SYNC=1 RL_KERNEL_DSV41_SKIP_ADAPTER_SYNC=1
    export STEPS=${CONTROL_STEPS:-3}
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
    # The launcher sets keys verl's config schema at this pin lacks; Hydra refuses them unless
    # appended ("+"). BASELINE_APPEND_KEYS lists them (found with `run.sh baseline --cfg job`).
    # Model.lora's go to Megatron-Bridge's create_peft, which drops fields it does not have.
    for key in ${BASELINE_APPEND_KEYS:-$BASELINE_APPEND_DEFAULT}; do
      sed -i "s/^\(\s*\)$key=/\1+$key=/" "$RUN/launcher.sh"
      grep -q "+$key=" "$RUN/launcher.sh" || { echo "$key edit failed"; exit 1; }
    done
    for key in ${BASELINE_DROP_KEYS:-$BASELINE_DROP_DEFAULT}; do
      grep -q "^\s*$key=" "$RUN/launcher.sh" || { echo "$key not in the launcher"; exit 1; }
      sed -i "/^\s*$key=/d" "$RUN/launcher.sh"
    done
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
        "${REWARD_FN[@]}" \
        ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS:-$(nproc)}" "$@" 2>&1 | tee "$RUN/train.log"
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
