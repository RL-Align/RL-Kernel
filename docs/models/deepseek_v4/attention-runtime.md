# DSV4 attention mode gate (issue 81, item 27)

This opt-in request-level boundary composes the existing joint-attention
forward plan (item 24, PR #628) and native autograd adapter (item 25, PR
#629). It starts from RL-Kernel main `8588158650aff58c9c1f527674de94bad06c3d10`.
The separately merged item-22 branch supplies the versioned compressor-state
APIs, but this gate does **not** implicitly attach to their cache buffers.
No new numerical GPU kernel is introduced.

`AttentionRuntime(plan, version=..., autograd=...)` accepts bound
`JointMetadata`, Q, the canonically described KV rows, and sink. Train and
recompute share the same numerical plan; a gradient-bearing train call requires
the native `JointAttentionAutograd` instance bound to that plan. Recompute
compares BF16 O and FP32 LSE bytes per `(request, global_position)`, including
when rows are reordered. It first checks Q, canonically ordered KV, sink,
logical KV identities, and selected indices against owned training snapshots.
Generation requires contiguous logical positions:
prefill begins at zero, decode admits one next token per request. Stale
versions, duplicate tokens, incomplete prefill, uncaptured Graph use, and
gradient-bearing generation fail closed. Request position advances only after
the native call succeeds. A failed native launch poisons the underlying plan.

The caller owns the Indexer Top-K decision, cache write/read event ordering,
KV row materialization, and its model/scale version barrier. `JointMetadata`
checks ownership, complete compressed groups, window completeness, canonical
read order, and tensor mutation. Use the existing vLLM packed-cache reader
only with its explicit OCP/FNUZ ABI; do not equate lossy packed values with
unquantized training KV. The mode gate does not copy or replace those kernels.
The host-side metadata readback, byte comparison and saved input/output records
are correctness instrumentation, **not** a low-overhead serving path.

MI300X/gfx942 validation in the `rl-kernel-ljj` ROCm container:

```bash
export RLK_AITER_DSV4_SPARSE_KERNEL_SOURCE=/workspace/upstream/aiter-sparse-326c/kernels.py
export RLK_AITER_DSV4_SPARSE_WRAPPER_SOURCE=/workspace/upstream/aiter-sparse-326c/wrapper.py
PYTHONPATH=. pytest -q tests/models/deepseek_v4/test_attention_runtime.py \
  tests/models/deepseek_v4/test_joint_attention_rocm.py \
  tests/models/deepseek_v4/test_joint_backward_rocm.py
```

This is **not** item-27 end-to-end signoff: neither the Megatron/vLLM model
call sites nor verl/vime are wired to this opt-in boundary; full C4/C128 and
Indexer state are not combined into one live request; there is no whole-engine
Graph capture, CP/DCP, PP, multi-rank transport, or two-update weight-sync
test. Item 39 and items 40-42 require their own transport/parallel contracts
and cannot be inferred from single-card attention tests. The complete 0731
checkpoint and the two-node topology remain untested on this machine.
