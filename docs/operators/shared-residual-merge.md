# P6 / T04 Shared and Residual Merge Reference

T04 merges three token-local rows using exactly `(routed + shared) + residual`.
Both additions materialize FP32; only the final output is converted to BF16.
The routed row is already weighted by P5 and summed by T03. This operator never
multiplies route weights, runs a collective, computes gradients, or performs mHC
post-mixing. It is an eager **forward reference**, not a fused production kernel.

## Interface and scope

```python
from rl_engine.kernels.ops.pytorch.moe import shared_residual_merge_fwd

result = shared_residual_merge_fwd(
    routed, shared, residual, context=context, debug=True,
)
y = result.token_moe_row_bf16
receipt = result.receipt
```

| Value | Required shape / type |
| --- | --- |
| `routed` | Contiguous `[T,H]` FP32, already combined in canonical slot order |
| `shared`, `residual` | Same shape and device; BF16 or FP32 |
| `context` | `MergeContext`, carrying validated producer receipts and expected identities |
| `y` | `[T,H]` BF16, no autograd graph |

CPU and CUDA-device PyTorch eager execution are implemented; ROCm uses PyTorch's
CUDA device API. Hardware qualification requires running the named GPU tests.
Ascend, tensor subclasses, non-contiguous tensors, empty batches, broadcasting,
FP16 inputs, compiled execution, and CUDA Graph capture are rejected. A zero-count
EP rank does not call the local merge; its participation belongs to the P4/T08 gate.

The semantic registry exposes the explicit backend
`rlkernel.moe.shared_residual_merge.reference.v1` for `shared_residual_merge`.
It is marked `reference_only`, `forward_only`, and not production-certified. It
is not inserted into generic fallback priorities or a model's FFN path. A caller
requesting strict GPU launch observability cannot use this reference as proof.

## Evidence and exactly-once

`MergeIdentity` binds case/run/pass, checkpoint and weight fingerprints, opaque
RoutePlan/ExchangePlan/CombinePlan fingerprints, fixture checksum, and ordered
global token IDs. `MergeSource` attaches a role, a stable expected source ID,
the same identity, and a checksum over the tensor's dtype/shape/raw bytes.
The caller obtains these from validated producer artifacts. The reference does
not create a RoutePlan, ExchangePlan, CombinePlan, inverse map, or input adapter.

`MergeContext` requires one source for each of routed/shared/residual. It also
requires an upstream `route_weight` event exactly once, P5 ownership and weighted
input, and no prior shared/residual application or P6 local downcast. That local
downcast history does **not** count BF16 storage conversions inside P5. The fixed
local order is shared, residual, final BF16 cast. Unknown operations and mHC post
input roles fail closed.

All rows in a call have the same application history and must belong to the
executing token owner. Shared-expert replicas are recorded separately; their
number never scales the shared contribution. `rank` and replica coordinates are
caller-supplied P4 evidence, not measurements of a process group. The reference
cannot prove rank completeness, global uniqueness of ownership across independent
calls, or validate a false upstream claim. T08/live integration must establish
those facts. Repeating a pure reference call for comparison is allowed; the
output receipt's executed events prevent reusing its result as an unmerged input
when receipts are propagated correctly.

Identity and discrete checks precede tensor checksum and numerical checks. The
function then checks finite values at every boundary, including BF16 overflow.
Receipt counts derive from recorded local events and observed ATen dispatches;
they are not a certificate that upstream metadata was truthful.

The three boundary keys are:

- `P6.merge.after_shared`
- `P6.merge.after_residual`
- `P6.merge.final_bf16`

Every boundary records identity, rank, shape/dtype/layout, event index, backend,
implementation hash and tensor checksum. Slot is not applicable at this token-row
boundary; T03 owns the slot-order hash. `merge_order_hash` fingerprints only T04's
versioned three operations and is not a replacement for P6's full combine hash.
Debug off returns only hashes and the output. Debug on also returns intermediate
tensors; the execution and hashes are identical in both modes. Treat returned
tensors as immutable when associating them with receipts.

ATen adds and casts are observed through `TorchDispatchMode`. Actual GPU kernel
symbols, tile/warp/stage/vector width are not exposed by this layer and are
explicitly unobserved, never populated with guessed configuration values. The
reference synchronizes and copies tensor bytes to CPU for evidence, so it must
not be used as a production latency baseline.

## Validation and handoff

Use the existing pytest entry point from the repository root:

```bash
python3 -m pytest -q tests/test_shared_residual_merge.py

# Retain append-only local reference artifacts with the existing ArtifactStore.
RL_KERNEL_T04_ARTIFACT_DIR=/tmp/p6-t04-evidence \
  python3 -m pytest -q tests/test_shared_residual_merge.py \
  --junitxml=/tmp/p6-t04-tests.xml

# Run the named GPU slice on a GPU host; skips do not constitute a pass.
python3 -m pytest -q tests/test_shared_residual_merge.py -k gpu_golden
```

`tests/fixtures/p6_t04_merge.json` contains only synthetic inputs and manually
specified intermediate FP32 values and BF16 bit patterns. Its SHA256 is pinned
in the test. Goldens cover cancellation/order, early-rounding loss, signed zero,
FP32 and BF16 ties, largest finite BF16, and smallest normal BF16. Negative tests
exercise source duplication/missing data, identity drift, ownership, mHC misuse,
invalid types/layouts, non-finite inputs and intermediate/final overflow.
Batch/chunk/permutation/padding checks inspect raw bytes, preserving signed zero.

EP=1/2/4/8 cases simulate token ownership and shared replication on CPU. They
are local semantic tests, **not** a multi-GPU WS2 certificate. CUDA/ROCm golden
tests compare the same final and intermediate bit patterns when hardware exists.
No T05 fused-forward, T06 backward or live producer claim is made by this suite.

The artifact test uses the existing `ArtifactStore` and its completion seal,
checksum validation and immutable experiment identity. It writes T04 receipts
and debug bundles for the three synthetic golden cases. Retained files can be
read with `json` and `ArtifactStore.load_tensor_bundle`; each boundary checksum
is validated after reloading. A seal means complete local artifacts, not complete
P6 certification. Reusing an evidence root after a fixture, context or build
change fails resume; use a new experiment root for a new revision.

The broad `check_operator.py` dtype/tolerance harness is not modified: this
checked reference requires producer receipts and strict raw-byte tests, while
T01 has not yet frozen the P6 runner. T07 should consume these receipts in its
existing comparator/store. T05 should compare all three boundaries, not just the
final BF16 tensor. T03 is the designated reviewer for the T04 handoff.

## Contract delta for T01 review

| Field | Proposal |
| --- | --- |
| Current contract / ABI | `p6-task-contract.v1` / `foundation-moe-return.v1` |
| Proposed local receipt version | `p6.t04.merge-receipt.v1-proposal` |
| Change | Add `MergeIdentity`, `MergeSource`, `MergeContext`, `MergeResult` and keyword `context`/`debug` at the reference boundary |
| Compatibility | Does not change arithmetic, weight owner, return layout, gradient ownership or Foundation ABI |
| Affected consumers | T01, T03 reviewer, T05, T07, T08 |
| Input dtype profile | Routed FP32; shared/residual BF16 or FP32; T01 must approve this admitted subset |
| Fixture IDs | `T04-LOCAL-*` are provisional local IDs; T01 assigns canonical catalog IDs/checksums |
| Blocking scope | Formal P6 ABI certification and live integration; local reference can be tested independently |

No T01 start kit or public CombinePlan implementation was present in the base
checkout. These small receipt dataclasses are an explicit review proposal rather
than a claim to be that missing schema. Map them to the canonical start kit once
available; do not maintain a competing backend-specific protocol. Review,
approved fixture IDs, GPU evidence and live integration remain separate gates.
