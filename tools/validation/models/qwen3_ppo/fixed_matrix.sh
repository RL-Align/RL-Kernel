# Training TP sizes of each fixed-rollout candidate row. Sourced by qualify_candidate.sbatch and
# probe_trainer_memory.sh, so both run the same matrix.
#
# The BF16 actor rows are TP4 only: the Megatron BF16 actor ran out of memory at TP2 on a
# 93 GiB H100 (92.9 GiB per rank on B200), and the DeepSpeed BF16 actor holds the same model,
# gradients and optimizer state. Every other row runs at TP2 and TP4.
fixed_tp_sizes() {
  local engine=$1 precision=$2 role=$3
  if [[ "$precision-$role" == bf16-actor ]]; then
    echo 4
  else
    echo 2 4
  fi
}
