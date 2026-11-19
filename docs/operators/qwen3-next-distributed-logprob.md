# Qwen3-Next distributed logprob parity (TP and EP)

## Summary

RFC #428 row `distributed_logprob_parity` asks for a locked TP/EP reduction
order and an exactly equal selected-token logprob between VIME replay and vLLM
rollout. This page covers tensor parallelism (TP4, EP = 1, CP = 1) and expert
parallelism (TP4 x EP4, CP = 1) for Qwen3-Next-80B-A3B-Instruct.

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

## Expert parallelism (EP4)

```python
from rl_engine.models.qwen3_next.qwen3_next_ep import EP4MoE, shared_moe_ep

moe = EP4MoE(group=tp_group, device=device)  # rank r owns experts [128r, 128r + 128)
```

`--ep 4` on `tools/validation/models/qwen3_next/qwen3_next_run.py` runs both
engines with expert parallel size 4 over the TP4 group and expert tensor
parallel size 1 (vLLM `enable_expert_parallel` with linear placement; Megatron
`expert_model_parallel_size` 4, `expert_tensor_parallel_size` 1). Attention,
GDN, embedding, LM head and the shared expert stay TP4 as above.

| Cross-rank point | Where | Operation | Order / dtype |
| --- | --- | --- | --- |
| Router and top-10 routes | `shared_router`, `stable_top10_routes` | replicated on every rank (input is already replicated) | FP32; no collective |
| Dispatch | `qwen3_next_ep._slots_by_owner` | each rank selects the routes whose expert it owns | local selection; no token all-to-all |
| Routed expert rows | `qwen3_next_ep._exchange` | each rank's `[slots, 2048]` rows, padded to the largest slot count, all-gathered and placed by slot | fixed-order collective `all_gather`: byte copy through CUDA IPC, no arithmetic; a slot owned by no rank or by two fails closed |
| Route combine | `combine_routes` | the 10 route rows per token | FP32, route order, on every rank (as EP1) |
| Shared expert | `_ReduceFromTP` | row-parallel partial sums, then `routed + shared * gate` | fixed-order collective, BF16 (as TP4) |
| Routed-row input gradient | `_EPRoutedExperts.backward` | per-route gradients exchanged like the forward rows, summed per token | ascending expert order, as EP1 |
| Expert weight gradients | same | only the owning rank computes them | local; no cross-rank reduction |

The EP4 block equals all 512 experts evaluated at full width on one GPU (EP1)
bitwise, forward and backward. It differs from the TP4 (expert tensor parallel
4) block by up to 4.9e-4 on layer 0, because each expert's down projection is
one K = 512 GEMM instead of four K = 128 partials and a fixed-order sum. EP4 is
therefore its own certified profile (`qwen3-next-moe-ep4-v1`); parity is EP4
rollout == EP4 replay. Fail-closed: both engines accept only EP 1 (expert
tensor parallel 4) or EP 4 (expert tensor parallel 1) with TP4, PP1, CP1; vLLM
must use linear expert placement, and every rank checks that its expert map is
`[128r, 128r + 128)`.

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
- EP4: [`reports/experiments/qwen3_next/qwen3-next-distributed-logprob-ep4-b200.md`](https://github.com/fusheng-ji/RL-Kernel/blob/feat/ws1-qwen3-next-distributed-logprob-ep/reports/experiments/qwen3_next/qwen3-next-distributed-logprob-ep4-b200.md).
  The EP4 MoE gate passes on real layer-0 weights (each expert held by one
  rank; EP4 == EP1 forward and backward for 8-1024 tokens; chunking,
  reordering, a rank without routes), and the full-model L2 on TP4 x EP4 is
  bitwise for all 16 cases: tokens, logits, logprobs and every layer, sub-node
  and state digest of vLLM rollout equal VIME recompute.

## Tests

| File | Device | Covers |
| --- | --- | --- |
| `tests/models/qwen3_next/test_qwen3_next_tp_reductions.py` | CPU | no raw NCCL reduction in Qwen3-Next forward paths; the TP reduction uses the fixed-order collective |
| `tools/validation/models/qwen3_next/qwen3_next_logprob_parity.py` + `..._compare.py` | 4 × CUDA | digests at every cross-rank point under each NCCL setting; TP4 == TP1 |
| `tests/distributed/tp/test_vocab_parallel_logp.py` (existing) | CUDA | WS2 TP = n == TP = 1 on synthetic shapes |
| `tests/models/qwen3_next/test_qwen3_next_ep.py` | CPU | expert placement and map, slot ownership, EP-dependent layer gates, accepted training and L2 topologies |
| `tools/validation/models/qwen3_next/qwen3_next_ep_moe_check.py` | 4 × CUDA | EP4 MoE gate on real weights (`layers` stage with `--ep 4`) |

## Known Limitations

- TP4 with EP = 1 or EP = 4 and CP = 1 only.
- EP4 is certified up to the full-model L2 (rollout vs replay). Training on EP4
  (Megatron expert-parallel optimizer state, checkpoint export and reload) is
  not supported yet: the training actor and HF export fail closed on EP4.
- EP4 and TP4 x EP1 are different profiles and do not give the same bits.
- Entropy merges per-shard contributions in rank order and therefore changes
  with the TP degree; rollout and replay run the same TP degree, so they still
  agree, but entropy is not TP-degree invariant.
