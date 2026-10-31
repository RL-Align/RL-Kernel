# Item 25: native sparse-attention and sink VJPs

This extends item 24's **exact saved forward record** with native dQ/dKV,
fixed-order d_sink, a fixed-record GPU Graph path, and a canonical autograd
bridge. There is no new HIP/Triton numerical kernel.

## Reused arithmetic

| Stage | Provider |
|---|---|
| `delta = sum(O * dO)` | aiter MHA `_bwd_preprocess`, fixed 16-query tile |
| dQ and BF16 P/dP contributions | pinned aiter `_bwd_dq_store_dp_kernel`, H16/K32 |
| Shared K=V gradient contributions | pinned aiter `_bwd_dkv_interm_kernel` |
| KV fan-in | pinned aiter CSR helper and `_bwd_dkv_gather_kernel`, G64 |
| Sink contributions/reduction | Torch FP32 elementwise ops, explicit 32-query tiles, then tile merge |
| Cross-microbatch ordering | existing RLK `CanonicalBackwardSession.submit_rows` |

The upstream dQ kernel is called with **`HAS_ATTN_SINK=False` only to disable
its floating sink atomic branch**. It still receives the real **sink-inclusive
LSE from item 24**. The flag does not change its dQ/P/dP computation.
It would be incorrect to substitute a sink-free LSE.

KV contributions combine dK and dV in the upstream fixed head-group order
because the forward uses the same vector for K and V. CSR rows then gather
contributions in canonical query order. The reused CSR builder uses integer
counts and stable sorting, not floating gradient accumulation.

Sink contributions are `-delta * exp(sink - LSE)`. Non-finite disabled
sink/empty-attention entries contribute zero. Padded queries are removed
before the fixed 32-row reduction. Microbatch sink gradients are **not**
summed as independently rounded partial results.

## Provision the pinned sources

```bash
python3 scripts/prepare_dsv4_sparse_provider.py \
  --output /workspace/upstream/aiter-sparse-326c/kernels.py \
  --backward-wrapper-output /workspace/upstream/aiter-sparse-326c/wrapper.py
export RLK_AITER_DSV4_SPARSE_KERNEL_SOURCE=/workspace/upstream/aiter-sparse-326c/kernels.py
export RLK_AITER_DSV4_SPARSE_WRAPPER_SOURCE=/workspace/upstream/aiter-sparse-326c/wrapper.py
```

Both files come from aiter commit `326cffa38c9eda16bef19375ba55334e4d3acd59`.
The wrapper SHA256 is
`ede08814720b9c783c7375cab9f95ee38bf6212ca77ba23149e33309b72842ff`.
The adapter loads only its verified `_build_inverted_topk` AST function;
it does not monkeypatch installed aiter modules. Wrong content is rejected.
No runtime download or numerical fallback is provided.

## Autograd integration

```python
from rl_engine.integrations.engines.train.megatron.joint_backward import (
    JointAttentionAutograd,
    canonical_backward_session,
)

# plan and metadata use item 24's explicit geometry/ownership/version contract.
attention = JointAttentionAutograd(plan, layer_id="decoder.0")
with canonical_backward_session() as session:
    result = attention(
        q, kv, sink, metadata=metadata, version="optimizer-step-10",
    )
    result.output.backward(d_output)
    session.validate_complete()
```

For microbatches, register **all forwards before any backward** in that
session. Active query identities must be disjoint. A parameter's uses must
share the same KV tensor identity, logical KV/loader order, model version,
forward/backward implementation and event-owned forward plan.
The full active query set must fit the contract capacity and backward budget.
These restrictions prevent mixing stale state or unsynchronized plan storage.

Each microbatch computes independent dQ and per-query contributions through
the native kernels. The existing session orders the contributions by
`(request_id, global_position)` before the **one final** CSR/sink reduction.
The last submitted backward returns the aggregate KV/sink gradient; earlier
submissions return `None` for those shared gradients.

Q/KV input gradients are cast to **BF16**, matching the native training
boundary. Sink gradients are FP32. The bridge maps dKV back to the original
loader order automatically. LSE is explicitly non-differentiable diagnostic
state. Higher-order derivatives are rejected.

This is the upstream-style approximate training VJP with BF16 P/dP and saved
BF16 O. It is not a mathematical derivative of discrete BF16/FP8 rounding.
FP64 real-valued gradients are checked with tolerances; BI and repeated
execution are checked separately using exact bytes.

## Fixed-record backward Graph

```python
from rl_engine.models.deepseek_v4.joint_backward import JointBackwardPlan

saved = plan.forward_record(
    q, kv, sink, metadata=metadata, version="optimizer-step-10", mode="train",
)
backward = JointBackwardPlan(plan, saved)
backward.capture()
gradients = backward.backward(d_output, use_graph=True)
```

CSR preparation and canonical query sorting happen **outside** capture.
Q/KV/O/LSE/index state is frozen for this plan; dO may change between replays.
All returned tensors own storage. Shared forward-plan locks/events protect
workspace reuse across streams, and immutable state/CSR mutation is rejected.
A failed mutable workspace poisons its owning plan.

Manual results are FP32: dQ is in the original query order, while dKV is in
the **canonical logical KV order** of the saved record. To obtain the original
loader order, use `dkv.index_select(0, argsort(saved.metadata.permutation))`.
The autograd bridge performs this mapping and BF16 conversion for you.

The native-workspace budget includes delta, dQ, P/dP, per-query KV
contributions and gradient buffers. It excludes saved input/record clones,
CSR metadata, Graph pools and temporary Torch reduction allocations.

## Boundaries

Validated: item-24 MI300X profile, repeated/shared KV, partial head tiles,
empty/padded/disabled-sink cases, up to 640 selected rows / H128 / D512,
canonical microbatch and backward-order invariance, fixed-record native
backward Graph, and stream-safe result lifetime.

Not claimed: whole-training-step Graph capture, framework checkpoint/recompute
orchestration, distributed CP/TP exchange, FP8 cache-byte gradients, LoRA,
full model integration, higher-order derivatives or cross-parallel-degree
identity. The canonical autograd collector is eager; it does not pretend
dynamic CSR admission is Graph-capturable. Item 27 owns the broader runtime.

Forward/backward provider fingerprints include pinned source hashes, installed
MHA preprocessing source, Torch/Triton/HIP versions and RLK bridge/session code.
See `reports/joint-backward-mi300x-20261029.md`.
