# MoE Merge

`shared_residual_merge_fwd` adds the shared-expert output and residual to the combined routed-expert
output. Both additions use FP32 in the fixed order `(routed + shared) + residual`;
only the final result is converted to BF16. Fixing this order prevents different
rounding results between callers that otherwise implement the same formula.

## Interface

```python
from rl_engine.kernels.ops.pytorch.moe import MoeMergeOp, shared_residual_merge_fwd

y = shared_residual_merge_fwd(routed, shared, residual)

# The registry uses the same nn.Module / forward interface as other native ops.
op = MoeMergeOp()
y = op(routed, shared, residual)
```

| Value | Shape and dtype |
| --- | --- |
| `routed` | Contiguous, non-empty `[T, H]` FP32; routing weights and expert combination already applied |
| `shared`, `residual` | Same shape and device as `routed`; FP32 or BF16 |
| `y` | `[T, H]` BF16; no autograd graph |

The implementation supports CPU and CUDA-device PyTorch eager execution; ROCm
uses PyTorch's CUDA device API. It rejects broadcasting, non-contiguous tensors,
unsupported dtypes, compiled execution and CUDA Graph capture. This is a
forward-only reference. It performs no routing, collective or post-merge mixing.
Token ownership and source provenance are responsibilities of the caller.

The computation path performs tensor metadata checks and arithmetic. It does not
trace dispatches, hash tensors, build receipts or inspect source files. Values
follow PyTorch arithmetic, including non-finite propagation; the validation path
below rejects non-finite inputs and results.

## Intermediate results

The reference exposes the same computation with its two FP32 intermediates:

```python
after_shared, after_residual, output = op.forward_with_intermediates(
    routed, shared, residual,
)
```

`forward` and `forward_with_intermediates` share the arithmetic implementation.
The validation helper names these tensors `after_shared`, `after_residual` and
`final_bf16`. Collecting them does not change the output bytes.

## Registration

The existing semantic registry resolves `shared_residual_merge` to the explicit
backend `rlkernel.moe.shared_residual_merge.reference.v1`. Its factory constructs
`MoeMergeOp`; implementation provenance comes from `OperatorSession`.

The descriptor declares deterministic, batch-invariant, forward-only reference
arithmetic. Its dtype is the routed input's FP32 dtype; shared and residual dtypes
are validated by the operator. The empty topology capability requires a local
request. Distributed callers validate their topology and invoke the operator on
each token owner's rows. GPU launch parameters are not observable at this layer.

## Validation

Run the focused tests from the repository root:

```bash
python3 -m pytest -q tests/test_moe_merge.py

# Retain receipts and tensor snapshots through the existing ArtifactStore.
RL_KERNEL_MOE_MERGE_ARTIFACT_DIR=/tmp/moe-merge-evidence \
  python3 -m pytest -q tests/test_moe_merge.py
```

`tests/fixtures/moe_merge.json` contains synthetic inputs, manually specified FP32
intermediates and exact BF16 words. Tests cover addition order, early rounding,
signed zero, ties, dtype limits, autocast, input immutability and batch/chunk/
permutation/padding invariance. Comparisons use raw bytes, including signed zero.

`rl_engine.alignment.testing.moe_merge` holds the identity/count gate and evidence
collection used by these tests. It validates source identities, checksums, token
ownership, weighted routed inputs and application history before numeric checks.
Missing or repeated sources, prior shared/residual application, early downcasts,
and post-merge mixing inputs are rejected. This metadata is not part of the tensor
operator's API.

The receipt records the validated upstream history, reference schedule, actual
intermediate tensor checksums and the shared registry's implementation provenance.
Application counts describe that reference schedule; they are not measured ATen
or GPU launch counts. Fixed numerical fixtures validate the arithmetic. A receipt
cannot establish that an upstream producer's claims are truthful.

CPU tests simulate expert-parallel replication at degrees 1, 2, 4 and 8. GPU tests
start one NCCL/RCCL process per device at those degrees, merge uneven owner-local
rows and compare gathered token IDs and output bytes against an independent
integer oracle. Insufficient GPU counts are skipped. Communication exists only in
the test harness; these checks do not establish full model integration.

Artifact tests use the existing `ArtifactStore` for append-only writes, completion
seals, reload checks and corruption detection. Receipts and boundary tensors use
`merge_receipt.json` and `merge_debug.pt`. Use a new evidence directory after
changing fixtures, metadata or implementation; completed artifacts remain immutable.
