# Training TP sizes of each fixed-rollout candidate row. Sourced by qualify_candidate.sbatch and
# probe_trainer_memory.sh, so both run the same matrix.
#
# TP4 only: the BF16 actors and the DeepSpeed BF16 critic. The Megatron BF16 actor ran out of
# memory at TP2 on a 93 GiB H100 (92.9 GiB per rank on B200); the DeepSpeed BF16 actor holds
# the same state; the DeepSpeed BF16 critic peaked at 98.9 GiB per rank at TP2 on B200 (its
# flat BF16 copy and FP32 gradient buffer). Every other row runs at TP2 and TP4.
fixed_tp_sizes() {
  local engine=$1 precision=$2 role=$3
  if [[ "$precision-$role" == bf16-actor || "$engine-$precision" == deepspeed-bf16 ]]; then
    echo 4
  else
    echo 2 4
  fi
}
