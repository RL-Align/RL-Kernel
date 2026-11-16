#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DSv4 LoRA GRPO with verl on RL-Kernel: the rlkernel engine (actor) + the vLLM rollout plugin,
# both on RL-Kernel's batch-invariant kernels. verl and vLLM run unmodified; RL-Kernel plugs in
# through verl's model.external_lib / hydra search path and vLLM's worker_extension_cls.
#
#   MODEL_PATH=... DATA=... N_GPUS=2 bash scripts/verl_e2e/run_dsv4_rlkernel_grpo.sh [overrides]
#
# The verdict is verl's training/rollout_logprobs_mismatch_count: response tokens whose
# rollout (vLLM) and recomputed (actor) FP32 log-probs differ. 0 at every step means bitwise.
# The synthetic reward (custom_reward.py) gives nonzero advantages on a model that cannot
# solve the task, so the adapters move every step and every step re-tests the weight sync.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RLK_ROOT=${RLK_ROOT:-$(cd "$HERE/../.." && pwd)}
MODEL_PATH=${MODEL_PATH:?checkpoint directory}
DATA=${DATA:?directory with train.parquet / test.parquet}
N_GPUS=${N_GPUS:-2}
LORA_RANK=${LORA_RANK:-16}
STEPS=${STEPS:-4}
VLLM_KW=+actor_rollout_ref.rollout.engine_kwargs.vllm
RLK_WORKER=rl_engine.integrations.dsv4.vllm_worker

export PYTHONPATH=$RLK_ROOT${PYTHONPATH:+:$PYTHONPATH}
export VLLM_BATCH_INVARIANT=1 RL_KERNEL_DSV4_ROLLOUT=1 VLLM_DS4_DECODE_KERNEL=sparse
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
    data.train_batch_size=8 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
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
    actor_rollout_ref.actor.rlkernel.lora_alpha=$((2 * LORA_RANK)) \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_model_len=512 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.load_format=auto \
    $VLLM_KW.kv_cache_dtype=fp8 \
    $VLLM_KW.worker_cls=$RLK_WORKER.RlkWorker \
    $VLLM_KW.worker_extension_cls=$RLK_WORKER.RlkWorkerExtension \
    $VLLM_KW.enable_flashinfer_autotune=False \
    trainer.logger='["console"]' \
    trainer.project_name=dsv4_rlkernel \
    trainer.experiment_name=grpo_lora_bi \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=$STEPS \
    trainer.total_epochs=1 \
    "$@"
