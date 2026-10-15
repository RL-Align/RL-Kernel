# Qwen3-Next distributed logprob parity (TP)

## Summary

RFC #428 row `distributed_logprob_parity` asks for a locked TP/EP reduction
order and an exactly equal selected-token logprob between VIME replay and vLLM
rollout. This page covers tensor parallelism (TP4, EP = 1, CP = 1) for
Qwen3-Next-80B-A3B-Instruct. Expert parallelism is a separate PR.

Every place where TP ranks combine values either moves bytes (an NCCL
all-gather, which is exact whatever NCCL algorithm runs) or sums them with
RL-Kernel's fixed-order collective, whose order does not depend on NCCL at all.
No new collective or kernel is added; the row documents and proves the order.

## Entry Point

```python
from rl_engine.models.qwen3_next.qwen3_next_tp import _ReduceFromTP, _CopyToTP
from rl_engine.models.qwen3_next.qwen3_next_logprob import qwen_logprob_contract
from rl_engine.models.qwen3_next.qwen3_next_forward import shared_logprob

contract = qwen_logprob_contract(local_logits, rank=rank, world_size=4)
selected, lse, entropy = shared_logprob(local_logits, targets, contract=contract, tp_group=tp_group)
```

vLLM rollout (`RLKernelQwen3NextForCausalLM.compute_selected_logprobs`) and VIME
replay (`calculate_log_probs_and_entropy`, installed by
`install_vime_logprob_provider`) both call `shared_logprob` on their own
37984-column LM-head shard.

## Contract

| Cross-rank point | Where | Operation | Order / dtype |
| --- | --- | --- | --- |
| Vocabulary-parallel embedding | `qwen3_next_embedding.shared_embedding` | each rank selects its rows, zeros elsewhere; sum | fixed-order collective, BF16; exactly one non-zero term per element |
| Embedding weight gradient | same, backward | stable sort by row, pairwise FP32 tree in token order | local, no collective |
| GDN `out_proj`, attention `o_proj`, MoE routed + shared output | `_ReduceFromTP` in `qwen3_next_tp_gdn`, `qwen3_next_tp_blocks`, vLLM adapters | row-parallel partial sums | fixed-order collective, BF16 |
| Input gradients of column-parallel projections | `_CopyToTP.backward` | sum of per-rank partial gradients | fixed-order collective |
| Selected-token logprob, LSE | `shared_logprob` (WS2 `CudaVocabParallelLogprobOp`) | per-rank FP32 max / sum-exp per tile, all-gather of tile statistics, merge in global tile order over a fixed `[tokens, 64]` grid | FP32; independent of the TP degree |
| Entropy | same | per-shard contributions, all-gather, sum in TP-rank order | FP32; depends on the TP degree (outside the selected-logprob contract) |
| Sampling logits | `compute_logits` (vLLM) | all-gather of the column shards | exact data movement |

RL-Kernel's fixed-order collective (`rl_engine.distributed.algorithms.collectives`,
backend `cuda_ipc_fixed_tree`) stages every rank's input in a symmetric CUDA IPC
buffer and runs one reduction kernel that reads all peers and applies the same
balanced tree on every rank; NCCL is not involved in the arithmetic, so
`NCCL_ALGO`, `NCCL_PROTO` and the channel count cannot change the result. The
logprob all-gathers move per-tile statistics whose merge order is fixed by the
tile index, so changing the TP degree does not change selected logprobs or LSE
either.

Fail-closed: `shared_logprob` rejects a vocabulary other than 151936, CP > 1, a
TP degree other than 1 or 4, non-finite logits and a missing contract; the
fixed-order collective checks on first use that every rank calls it with the
same shape and dtype, and supports only TP sizes 1, 2, 4 and 8. `tests/models/qwen3_next/test_qwen3_next_tp_reductions.py` fails
if a raw `torch.distributed` reduction appears in a Qwen3-Next model, engine or
VIME module (an integer MAX of per-rank flags is the one documented exception).

## Results

Evidence: [`reports/experiments/qwen3_next/qwen3-next-distributed-logprob-b200.md`](https://github.com/fusheng-ji/RL-Kernel/blob/feat/ws1-qwen3-next-distributed-logprob-tp/reports/experiments/qwen3_next/qwen3-next-distributed-logprob-b200.md).

- 13 NCCL settings (defaults; Ring and all-reduce Tree × LL / LL128 / Simple ×
  1 channel / default channels), real Qwen3-Next weights, 1024 tokens, 4 × B200:
  every point in the contract table has one digest per rank across all
  settings, and the TP4 selected logprob and LSE equal TP1. The negative
  control -- real attention `o_proj` partials summed with NCCL `all_reduce` --
  gives 5 different results over the same settings.
- The full-model chain (#543) shows the end-to-end consequence: selected
  logprobs of vLLM rollout and VIME recompute are bitwise equal for 16 cases at
  four weight versions.

## Tests

| File | Device | Covers |
| --- | --- | --- |
| `tests/models/qwen3_next/test_qwen3_next_tp_reductions.py` | CPU | no raw NCCL reduction in Qwen3-Next forward paths; the TP reduction uses the fixed-order collective |
| `tools/validation/models/qwen3_next/qwen3_next_logprob_parity.py` + `..._compare.py` | 4 × CUDA | digests at every cross-rank point under each NCCL setting; TP4 == TP1 |
| `tests/distributed/tp/test_vocab_parallel_logp.py` (existing) | CUDA | WS2 TP = n == TP = 1 on synthetic shapes |

## Known Limitations

- TP4 with EP = 1 and CP = 1 only; expert parallelism is the second PR of this row.
- Entropy merges per-shard contributions in rank order and therefore changes
  with the TP degree; rollout and replay run the same TP degree, so they still
  agree, but entropy is not TP-degree invariant.
