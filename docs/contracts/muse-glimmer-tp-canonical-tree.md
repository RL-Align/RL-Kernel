# Muse Glimmer WS2: Canonical TP Tree

Status: `tp_canonical_tree` row of [RFC #501](https://github.com/RL-Align/RL-Kernel/issues/501)

Code: `rl_engine/models/muse_glimmer/tp_canonical_tree.py`
Validation: `tools/validation/models/museglimmer_tp_canonical_tree.py`
Tests: `tests/models/muse_glimmer/test_museglimmer_tp_canonical_tree.py`

## Scope

Every TP reduction in a Muse Glimmer decoder layer, evaluated so that physical TP 2, 4 and 8
reproduce TP1 byte for byte:

| Projection | Parallel | Reduced dim | Canonical leaf (1 of 8) |
| --- | --- | --- | --- |
| `self_attn.o_proj` | row | 4,096 | 512 (4 Q heads) |
| `mlp.down_proj` | row | 19,968 | 2,496 |
| `self_attn.q_proj`, `self_attn.gate_proj` | column (input grad) | 4,096 | 512 |
| `mlp.gate_proj`, `mlp.up_proj` | column (input grad) | 19,968 | 2,496 |

Not covered: K/V input gradients and the KV ownership map (`kv_head_replication`), the vocabulary
tree (`vocab_parallel_softcap_logp`), and sequence-parallel reduce-scatter.

## Reduction order

Canonical TP is frozen at 8 (`workload.CANONICAL_TP`). The reduced dimension is cut into 8
contiguous leaves `s0 … s7`. Each leaf is one GEMM with **FP32 output**. Leaves are combined in
FP32 as one balanced tree:

```
((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
```

Physical TP rank `r` owns leaves `[r·8/tp, (r+1)·8/tp)`, folds them into its subtree root, and the
fixed-tree collective (`cuda_ipc_fixed_tree`, `rocm_ipc_fixed_tree` or
`torch_distributed_balanced_tree`) folds the rank roots in the same balanced order. TP1, 2, 4 and
8 evaluate the same expression; only where each node is computed differs.

A column group (attention `q_proj` + `gate_proj`; MLP `gate_proj` + `up_proj`) reduces each lane
on its own tree, packed into one collective call, then sums the lanes **in declared order in
FP32**.

## Precision and cast points

- Inputs and weights: BF16 (any floating dtype is accepted for testing).
- Leaf GEMM: exact upcast, FP32 accumulation, FP32 output. TF32 fails closed.
- Tree and collective: FP32.
- One rounding to the activation dtype, after the last tree node (or lane sum).
- Row-parallel input/weight gradients and column-parallel weight gradients do not reduce across
  TP; they are computed per canonical leaf (so a leaf's GEMM shape does not depend on TP) and
  rounded once.

Every leaf GEMM has the same shape at every physical TP, so cross-TP equality needs only a
run-to-run deterministic leaf GEMM. The default is the FP32 upcast reference; the qualified
[G] kernel (`dense_gemm_qualification`) plugs in through `gemm=`. Batch invariance of the leaf
GEMM is the [G] row's contract.

## Fails closed

- `RL_KERNEL_STRICT_CANONICAL_TP` set to anything but 8.
- Physical TP outside {1, 2, 4, 8}.
- A collective whose `backend_id` is not a fixed-tree collective (any native reducing
  collective).
- A fused projection (`qkv_proj`, `qkvg_proj`, `linear_qkv`, `gate_up_proj`, `linear_fc1`) or a
  projection that is not on the tree.
- A leaf GEMM that does not return FP32; shard shapes that are not the TP-local slice.

## Provenance

`tp_canonical_provenance(collective)` records physical and canonical TP, the TP rank and its
owned leaves, the collective backend, reduction order, partial dtype, rounding point, leaf GEMM
id, the column groups and the canonical shard width of each projection.

## Validation

```bash
# CPU, no GPU needed (one thread per rank on Windows, Gloo processes elsewhere)
python tools/validation/models/museglimmer_tp_canonical_tree.py --device cpu --shapes muse --tokens 8

# One 8-GPU node, with the strict vs native benchmark
torchrun --nproc-per-node 8 tools/validation/models/museglimmer_tp_canonical_tree.py \
    --device cuda --tp 2 4 8 --shapes muse --tokens 4096 --bench \
    --report museglimmer_tp_canonical_tree.json
```

For each TP, every rank compares its output, input-gradient and weight-gradient slices with the
locally recomputed TP1 result using `torch.equal`. TP1 is also checked against FP64, for
repeat-run determinism and for batch invariance.
