# Task 29: MI300X router sqrt-softplus

Task 29 is implemented on the original `dsv4-rocm` baseline
`5ba769f681b7fa765ea6b29f9939454a858390ba`. It is independent of the unmerged
task-28 Gate PR and consumes raw FP32 logits. No complete routing or model
integration is claimed.

The AITER companion patch reuses `_apply_score_mode` from commit
`326cffa38c9eda16bef19375ba55334e4d3acd59`, corrects negative-tail precision,
adds a standalone score/VJP entry and fixes FP-fusion options for sqrt-softplus
in existing Top-K/Hash/Grouped callers. Numerical kernels remain in the AITER
patch; RLK binds autograd, validates requests and records the actual execution.
The patch has been applied cleanly to the pinned original files and all three
resulting source hashes match the packaged manifest. It is not an upstream
AITER release, and no installed AITER package was replaced for validation.

## Numerical contract

The admitted domain is finite FP32 logits and upstream gradients. Output and
VJP are FP32. The normal branch uses OCML/libdevice exp/log1p and sqrt; a fixed
negative-tail approximation avoids intermediate softplus underflow. A deeper
negative-tail VJP rescales in log space so large finite upstream gradients can
produce representable derivatives even when the FP32 score is zero.

The independent CPU FP64 mathematical oracle uses logaddexp/sqrt and an analytic
VJP. Its conservative tail cutoff is -36 rather than the production -16: it
retains the mathematical formula on [-36,-16) to measure the production
approximation independently. It is not a production-arithmetic emulator.
Separate FP32-formula/ULP tests check the production -16 branch and neighbors.
Forward tolerance is `rtol=5e-7`; VJP tolerance is `rtol=2e-5`; both admit an
absolute difference of eight minimum FP32 subnormal quanta. BI checks require
zero differing raw bits independently of those numerical tolerances.

## Verification

```bash
AITER_TRITON_ONLY=1 PYTHONPATH="$PWD/artifacts/task29/aiter:$PWD" \
  python -m pytest tests/ops/moe/test_router_sqrt_softplus.py \
  tests/runtime/test_router_sqrt_softplus_dispatch.py tests/runtime/test_kernel_registry.py -q
```

| Suite | Result |
| --- | ---: |
| MI300X accuracy, score/VJP BI, upstream score reuse and runtime rejection | 41 passed |
| CPU-safe input/math, dispatch and source guards | 20 passed |
| Existing registry regression | 12 passed |
| **Total** | **73 passed; no failures or skips** |

The expected empty-graph warning comes from explicitly rejecting capture.
Ruff check/format, strict MkDocs build, patch application and whitespace checks
passed.

Coverage includes ten random seeds; E=1/6/255/256/257;
M=0/1/16/17/32/65/257; changed neighboring rows, row placement, padding and
transposes; full finite FP32 endpoints; nextafter values at the branch cutoffs;
large/small/signed-zero upstream gradients; repeated VJPs; saved-tensor mutation;
and rejection of unsupported dtypes/devices, NaN/Inf, Graph and higher derivatives.

mLite's pinned FP32 expression is compared on [-80,80]. Its -120 underflow and
NaN derivative are explicit negative controls. The corrected score and derivative
remain positive there. Existing AITER callers return exactly the standalone
score at each selected index; this test does not certify their selection/tie,
normalization, capacity or communication semantics.

`task29-mi300x.json` records the GPU/software/source fingerprints, synthetic
input/upstream hashes, independent numerical errors, BI results and performance.
All six replay batches M=1/16/17/32/65/257 have zero score and VJP bit mismatches
for the fixed row, and zero differences between training and inference forwards.
The largest measured normal-value relative errors are approximately **1.16e-7**
for scores and **5.88e-6** for VJPs. The fixture uses synthetic FP32 logits;
checkpoint weights, tokenizer processing and Preview data are not used.

Two independent processes with distinct Triton JIT cache directories produced
identical input, upstream-gradient, score and VJP hashes for all six batches.
`fresh-process-hashes.json` records those comparisons.

## Performance disclosure and limits

Each latency uses 10 warmups and 30 GPU-event samples around Python calls.
The full RLK adapter includes finite checks, synchronization, provenance and
autograd setup. The raw AITER calls include their Python allocation/launch costs;
they are not isolated GPU instruction timings. Across the checked-in replay,
median full-adapter forward latency is approximately **116–141 µs** and raw
AITER forward latency **28–34 µs**; per-case backward timings are in the JSON.
No full-model throughput or speedup is claimed.

Only MI300X/gfx942 is validated. MI355X/gfx950, NVIDIA, Graph, distributed
execution, complete routing, framework hooks, cross-microbatch accumulation
and full-model raw-logprob acceptance remain unvalidated or separate tasks.

See [the operator contract](../../../docs/operators/router-sqrt-softplus.md) for
the exact arithmetic/VJP order and isolated-checkout reproduction commands.
