#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run one verl + vLLM-BI DSv4 end-to-end test INSIDE the container (launch_container.sh).
# Wraps verl #7688's examples/grpo_trainer/run_deepseek_v4_true_on_policy_preview_megatron.sh.
#
#   bash /e2e/tools/run_e2e.sh T1      # upstream quick_alignment_test: 4 GPUs, 4 layers, 2 steps
#   bash /e2e/tools/run_e2e.sh T2      # 8 GPUs (EP8, rollout DP8/EP8), 4 layers, 5 steps
#   LAYERS=8 bash /e2e/tools/run_e2e.sh T3   # 8 GPUs, LAYERS layers, longer responses
#   bash /e2e/tools/run_e2e.sh T4      # 200 training steps, 8 layers, synthetic dense reward
#
# Any variable the upstream script reads (TRAIN_BATCH_SIZE, ROLLOUT_N, MAX_RESPONSE_LENGTH,
# TOTAL_TRAINING_STEPS, ROLLOUT_GPU_MEMORY_UTILIZATION, ...) can be exported to override.
# Extra arguments after the test name are passed through as Hydra overrides.
set -euo pipefail
TEST="${1:?usage: run_e2e.sh T1|T2|T3|T4 [hydra overrides...]}"
shift
source /e2e/tools/env.sh

export VERL_ROOT="${C_VERL}" MEGATRON_ROOT="${C_MEGATRON}"
export TRAIN_FILES="${C_DATA}/dapo-math-17k.parquet" VAL_FILES="${C_DATA}/aime-2024.parquet"
export WANDB_MODE="${WANDB_MODE:-disabled}"
UPSTREAM="${VERL_ROOT}/examples/grpo_trainer/run_deepseek_v4_true_on_policy_preview_megatron.sh"
[[ -f "${UPSTREAM}" ]] || { echo "missing ${UPSTREAM}; run setup.sh"; exit 1; }
grep -q "dsv4-e2e" "${VERL_ROOT}/verl/utils/debug/metrics.py" ||
  { echo "verl is missing the bitwise metric patch; run setup.sh"; exit 1; }

# A model directory with the first N layers: symlinks to the checkpoint plus an edited config
# (the same trick the upstream quick test uses for 4 layers).
make_layers_dir() {
  local n=$1 dst="${C_E2E}/models/DeepSeek-V4-Flash-L$1"
  mkdir -p "${dst}"
  for f in "${C_MODEL}"/*; do
    [[ "$(basename "$f")" == config.json ]] && continue
    ln -sfn "$f" "${dst}/$(basename "$f")"
  done
  python3 - "${C_MODEL}/config.json" "${dst}/config.json" "$n" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1])); c["num_hidden_layers"] = int(sys.argv[3])
json.dump(c, open(sys.argv[2], "w"), indent=2)
EOF
  echo "${dst}"
}

stamp=$(date +%Y%m%d-%H%M%S)
case "${TEST}" in
  T1)
    MODE=quick_alignment_test
    export MODEL_PATH="${C_MODEL}"     # upstream makes its own 4-layer view
    ;;
  T2|T3|T4)
    MODE=aligned
    case "${TEST}" in
      T2) LAYERS="${LAYERS:-4}" ;;
      *) LAYERS="${LAYERS:-8}" ;;
    esac
    MODEL_PATH="$(make_layers_dir "${LAYERS}")"
    export MODEL_PATH
    export NNODES=1 NGPUS_PER_NODE=8 ACTOR_PP=1 ACTOR_CP=1 ACTOR_EP=8
    export ROLLOUT_DP=8 ROLLOUT_EP=8 ROLLOUT_AGENT_WORKERS="${ROLLOUT_AGENT_WORKERS:-8}"
    if [[ "${TEST}" == T4 ]]; then
      # Long run: 200 optimizer updates. GRPO needs several samples per prompt; the
      # synthetic reward (custom_reward.py) gives nonzero advantages on a truncated model,
      # and a larger LR makes the BF16 / FP8 deployment weights change visibly every step.
      export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
      export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
      export ROLLOUT_N="${ROLLOUT_N:-8}"
      export ACTOR_LR="${ACTOR_LR:-1e-5}"
      export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
      export OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-1024}"
      set -- "reward.custom_reward_function.path=/e2e/tools/custom_reward.py" \
             "reward.custom_reward_function.name=compute_score" "$@"
    fi
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    # One optimizer update per step: the first actor forward must also equal the rollout
    # (actor/ppo_kl == 0 exactly), a third, grad-enabled copy of the log-probs.
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
    export ROLLOUT_N="${ROLLOUT_N:-4}"
    if [[ "${TEST}" == T3 ]]; then
      export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
    else
      export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
    fi
    export OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-512}"
    export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
    # dynamic batching needs room for one whole prompt + response
    export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$(( ${MAX_PROMPT_LENGTH:-2048} + MAX_RESPONSE_LENGTH ))}"
    export SAVE_FREQ=-1 TEST_FREQ=-1
    ;;
  *) echo "unknown test ${TEST}"; exit 64 ;;
esac

export RUN_NAME="${TEST}_${stamp}"
export OUTPUT_ROOT="${C_OUT}/${RUN_NAME}"
export LOG_FILE="${OUTPUT_ROOT}/train.log" JSONL_FILE="${OUTPUT_ROOT}/metrics.jsonl"
mkdir -p "${OUTPUT_ROOT}"
{
  echo "test=${TEST} mode=${MODE} model=${MODEL_PATH} layers=${LAYERS:-4}"
  env | grep -E '^(NNODES|NGPUS_PER_NODE|ACTOR_|ROLLOUT_|TRAIN_BATCH_SIZE|PPO_|MAX_RESPONSE|TOTAL_TRAINING)' | sort
  cat /opt/ds4/image-manifest.json
  git -C "${VERL_ROOT}" rev-parse HEAD
  git -C "${MEGATRON_ROOT}" rev-parse HEAD
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} > "${OUTPUT_ROOT}/run_info.txt" 2>&1

set +e
bash "${UPSTREAM}" --hardware h100 --mode "${MODE}" "$@"
rc=$?
set -e
echo "upstream exit code ${rc}" | tee -a "${OUTPUT_ROOT}/run_info.txt"
python3 /e2e/tools/summarize_metrics.py "${JSONL_FILE}" --out "${OUTPUT_ROOT}/summary" || true
exit "${rc}"
