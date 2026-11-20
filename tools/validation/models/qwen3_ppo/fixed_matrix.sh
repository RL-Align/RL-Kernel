# Training TP sizes of each fixed-rollout candidate row. Sourced by qualify_candidate.sbatch and
# probe_trainer_memory.sh, so both run the same matrix.
#
# Megatron BF16 actor is TP4 only: its TP2 probe peaked at 92.9 GiB per rank on B200, above
# what an 80-94 GB GPU holds. Every other row runs at TP2 and TP4.
fixed_tp_sizes() {
  local engine=$1 precision=$2 role=$3
  if [[ "$engine-$precision-$role" == megatron-bf16-actor ]]; then
    echo 4
  else
    echo 2 4
  fi
}
