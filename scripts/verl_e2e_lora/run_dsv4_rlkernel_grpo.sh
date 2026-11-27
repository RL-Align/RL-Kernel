#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DSv4 / DSv4.1 LoRA GRPO with verl on RL-Kernel: the rlkernel engine (actor) + the vLLM rollout
# plugin, both on RL-Kernel's batch-invariant kernels. verl and vLLM run unmodified; RL-Kernel
# plugs in through verl's model.external_lib / hydra search path and vLLM's
# worker_extension_cls.
#
#   MODEL_PATH=... DATA=... N_GPUS=2 \
#     bash scripts/verl_e2e_lora/run_dsv4_rlkernel_grpo.sh [overrides]
#
# The defaults are the 2-GPU toy run (design doc 10.1); every workload knob below can be
# exported to change it (scripts/verl_e2e_lora/run.sh sets them for the 8-GPU comparison).
# Extra arguments are Hydra overrides for keys this script does not set.
#
# The verdict is verl's training/rollout_logprobs_mismatch_count: response tokens whose
# rollout (vLLM) and recomputed (actor) FP32 log-probs differ. 0 at every step means bitwise.
# The synthetic reward (custom_reward.py) gives nonzero advantages on a model that cannot
# solve the task, so the adapters move every step and every step re-tests the weight sync.
#
# Parallelism: the actor is data parallel over N_GPUS with its routed experts split over
# groups of ACTOR_EP ranks (rlkernel.expert_parallel_size); the rollout runs N_GPUS / ROLLOUT_DP
# vLLM replicas, each data parallel attention over ROLLOUT_DP GPUs with its experts split over
# them (expert_parallel_size = ROLLOUT_DP, tensor parallel 1). Neither changes a byte.
#
# GRAPH=1 runs the rollout with vLLM's CUDA graphs (the fork's breakable graphs; decode replays
# them, attention runs eagerly inside). The adapters are reserved at load
# (RL_KERNEL_DSV4_LORA) and every sync copies into them, which the graphs read. Same bytes.
#
# BI=0 is the ablation: the rollout without batch invariance (vLLM's own attention paths,
# paged FP8 decode; rl_engine/integrations/dsv4/vllm_model.py). The mismatch count is then
# nonzero by design; the run measures the rollout's speed without the invariance.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RLK_ROOT=${RLK_ROOT:-$(cd "$HERE/../.." && pwd)}
MODEL_PATH=${MODEL_PATH:?checkpoint directory}
DATA=${DATA:?directory with train.parquet / test.parquet}
N_GPUS=${N_GPUS:-2}
ACTOR_EP=${ACTOR_EP:-1}
ROLLOUT_DP=${ROLLOUT_DP:-1}
# Thread budget. Every Ray process starts gRPC threads by the node's CPU count, and a cluster's
# per-user limit (ulimit -u; 2048 on some Slurm sites) counts every thread on the node. Past it,
# workers abort with "Resource temporarily unavailable" and the run hangs or fails at start.
# So: Ray's CPU count is the CPUs this process may use (its default counts the whole node and
# prestarts a worker per CPU), no prestart, no dashboard, fewer gRPC threads (GCS and raylet
# size theirs by the node's CPUs: about 220 threads for the GCS alone on a 224-CPU node), and
# AUX_WORKERS (not 8) reward workers, agent-loop workers and transfer-queue storage units.
# OMP_NUM_THREADS too: unset, every process that runs a torch CPU op starts about 2 x nproc
# threads (OpenMP and torch's pools; nproc is 96 in an 8-GPU allocation). The actor's CPU work
# is small (no offload), so 4 is enough. None of it changes a byte.
RAY_NUM_CPUS=${RAY_NUM_CPUS:-$(nproc)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
AUX_WORKERS=${AUX_WORKERS:-2}
for kv in prestart_worker_first_driver=0 worker_num_grpc_internal_threads=2 \
  num_server_call_thread=2 num_grpc_internal_threads=2 gcs_server_rpc_server_thread_num=2 \
  gcs_server_rpc_client_thread_num=2; do
  k=RAY_${kv%%=*}
  export "$k=${!k:-${kv#*=}}"
done
RECOMPUTE=${RECOMPUTE:-0}  # 1: the actor's per-layer activation recompute (same bytes)
[[ $RECOMPUTE == 1 ]] && ACT_RECOMPUTE=True || ACT_RECOMPUTE=False
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-$((2 * LORA_RANK))}
STEPS=${STEPS:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
# Dynamic batching needs room for one whole prompt + response per micro-batch.
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-$((MAX_MODEL_LEN > 4096 ? MAX_MODEL_LEN : 4096))}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
GPU_UTIL=${GPU_UTIL:-0.35}
ACTOR_LR=${ACTOR_LR:-1e-4}
LOGGER=${LOGGER:-'["console"]'}
BI=${BI:-1}
GRAPH=${GRAPH:-0}
[[ $GRAPH == 1 ]] && EAGER=False || EAGER=True
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_lora_bi$BI}
VLLM_KW=+actor_rollout_ref.rollout.engine_kwargs.vllm
# MODEL_FAMILY: dsv4 or dsv41, from the checkpoint's architecture by default.
if [[ -z ${MODEL_FAMILY:-} ]]; then
  MODEL_FAMILY=dsv4
  grep -q DeepseekV41ForCausalLM "$MODEL_PATH/config.json" && MODEL_FAMILY=dsv41
fi
export PYTHONPATH=$RLK_ROOT${PYTHONPATH:+:$PYTHONPATH}
ROLLOUT_EXTRA=()
if [[ $MODEL_FAMILY == dsv41 ]]; then
  # DSv4.1 on vLLM main (rl_engine/integrations/dsv41): eager, no prefix caching; vLLM keeps
  # scheduling and caches, RL-Kernel computes. No BI=0 ablation and no CUDA graphs yet.
  if [[ $BI != 1 || $GRAPH == 1 ]]; then
    echo "DSv4.1: only BI=1 GRAPH=0 so far" >&2
    exit 64
  fi
  RLK_WORKER=rl_engine.integrations.dsv41.vllm_worker
  export RL_KERNEL_DSV41_ROLLOUT=1
  ROLLOUT_EXTRA=(actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.actor.rlkernel.max_model_len=$MAX_MODEL_LEN)
else
  RLK_WORKER=rl_engine.integrations.dsv4.vllm_worker
  export RL_KERNEL_DSV4_ROLLOUT=1 RL_KERNEL_DSV4_LORA=$LORA_RANK,$LORA_ALPHA
  ROLLOUT_EXTRA=($VLLM_KW.kv_cache_dtype=fp8)
fi
if [[ $MODEL_FAMILY == dsv41 ]]; then
    :  # vLLM's batch-invariant mode is not needed: no vLLM kernel computes
elif [[ $BI == 1 ]]; then
    # vLLM's BI indexer selects with its DS4 deterministic top-k, a separate library
    # (csrc/ds4_bi_topk_extension.cu; scripts/verl_e2e_lora/setup.sh builds it). Without it the
    # rollout fails at the first sequence past ~2048 tokens (index_topk = 512 C4 entries).
    if [[ ! -f ${DS4_BI_TOPK_LIB:-} ]]; then
        echo "DS4_BI_TOPK_LIB must name vLLM's ds4_bi_topk library (BI=1)" >&2
        exit 1
    fi
    export DS4_BI_TOPK_LIB
    export VLLM_BATCH_INVARIANT=1 VLLM_DS4_DECODE_KERNEL=sparse RL_KERNEL_DSV4_NON_BI=0
else
    export VLLM_BATCH_INVARIANT=0 VLLM_DS4_DECODE_KERNEL=paged RL_KERNEL_DSV4_NON_BI=1
fi
export VLLM_USE_V1=1 NCCL_NVLS_ENABLE=0 TORCH_NCCL_AVOID_RECORD_STREAMS=1
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray-$USER}
# Slurm also sets ROCR_VISIBLE_DEVICES, which verl refuses next to CUDA_VISIBLE_DEVICES.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

python3 -m verl.trainer.main_ppo \
    "hydra.searchpath=[pkg://rl_engine.integrations.dsv4.verl_config]" \
    model_engine=rlkernel \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$DATA/train.parquet \
    data.val_files=$DATA/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    data.dataloader_num_workers=0 \
    reward.custom_reward_function.path=$HERE/custom_reward.py \
    reward.custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.external_lib=rl_engine.integrations.dsv4.verl_engine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.rlkernel.lora_rank=$LORA_RANK \
    actor_rollout_ref.actor.rlkernel.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.actor.rlkernel.expert_parallel_size=$ACTOR_EP \
    actor_rollout_ref.actor.rlkernel.activation_recompute=$ACT_RECOMPUTE \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=$ROLLOUT_DP \
    actor_rollout_ref.rollout.expert_parallel_size=$ROLLOUT_DP \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.enforce_eager=$EAGER \
    actor_rollout_ref.rollout.load_format=auto \
    "${ROLLOUT_EXTRA[@]}" \
    $VLLM_KW.worker_cls=$RLK_WORKER.RlkWorker \
    $VLLM_KW.worker_extension_cls=$RLK_WORKER.RlkWorkerExtension \
    $VLLM_KW.enable_flashinfer_autotune=False \
    trainer.logger="$LOGGER" \
    trainer.project_name=${MODEL_FAMILY}_rlkernel \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS \
    +ray_kwargs.ray_init.include_dashboard=False \
    actor_rollout_ref.rollout.agent.num_workers=$AUX_WORKERS \
    reward.num_workers=$AUX_WORKERS \
    transfer_queue.backend.SimpleStorage.num_data_storage_units=$AUX_WORKERS \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=$STEPS \
    trainer.total_epochs=1 \
    "$@"
