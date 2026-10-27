# Task 34: optional router auxiliary loss

Contract: [P3-B RFC #516](https://github.com/RL-Align/RL-Kernel/issues/516), item 34.
Base: `RL-Align/RL-Kernel:dsv4-rocm` at
`5ba769f681b7fa765ea6b29f9939454a858390ba`.

The RFC specifies deterministic denominators/reductions and overflow gradients,
but does not prescribe a specific formula. The explicitly confirmed profile is:

`L = alpha * 256 * sum_e(f_e * p_e)`, where
`f_e = pre-capacity Top-6 count[e] / (6*N)` and
`p_e = post-capacity unscaled normalized mass[e] / N`.

`N` counts all valid tokens in the complete domain, including all-dropped rows.
Frequency is discrete; probability retains the overflow-denominator derivative.
Default configuration is disabled and records that choice without invoking
routing, kernels or collectives. Enabling requires explicit alpha/version.

## Validation

|Check|Result|
|---|---|
|MI300X numerical/training/adversarial tests|50 passed|
|CPU optional-config tests|8 passed|
|Exact dispatch and source-failure tests|11 passed|
|Existing registry regression|12 passed|
|Total pytest|81 passed, no skips|
|Real fixed checkpoint-bias normal routing|24 cases, four profiles × six compute microbatch sizes|
|Independent FP64 loss/dlogits|Normal discrete choices checked first; mathematical checks passed|
|Overflow finite differences|Recompute normal selection/admission for each perturbation|
|Negative controls|Post-capacity frequency and detached capacity denominator detected|
|Route scale 0/1/1.5/2|Unscaled auxiliary loss and dlogits remain byte-identical|
|Padding/valid-token denominator|Invalid padded rows excluded, zero gradient, global N retained|
|Fresh-process raw bytes|24 cases; 147 tensors; 7,100,200 raw payload bytes equal|
|Two MI300X, automatic P4/HIP IPC|Five scenarios per rank passed|
|Two MI300X, explicit P4/RCCL tensor transport|Same five scenarios per rank passed|
|Rank config/upstream mismatch|Both rejected by collective metadata checks|
|Two streams / 50 calls|Independent workspace/results; no live-allocation growth|
|Patch portability|Four clean applications; all seven source hashes verified|
|Ruff check/format and strict MkDocs|Passed|

The pytest warning is from the intentional unsupported-Graph test ending an empty
capture. Arithmetic evidence is from MI300X gfx942, Torch
`2.12.0+rocm7.14.0a20260608`, ROCm `7.14.60850`, Triton `3.7.0`, and isolated AITER
base `326cffa38c9eda16bef19375ba55334e4d3acd59` plus the pinned companion patches.
The installed AITER package is unchanged. Task-29/31/32 numerical wrappers and
companion patch/manifests are unchanged.

`task34-mi300x.json` uses real fixed layer-3 bias from Flash-0731 revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`, verified against raw SHA256
`34d941394db670f6de58ccb9bb781158d12407cc27613f8ae90756e2623fb236`, with synthetic
FP32 N=257×256 logits (CPU seed 34). Profiles are unbounded capacity, capacity 2,
all drop, and capacity 2 with main route scale zero. Compute microbatch sizes are
1/16/17/32/65/257; each call still evaluates the same complete-domain objective.
No full checkpoint shard or full-model activation trace is used.

Maximum absolute errors versus the independent mathematical FP64 oracle are
`9.17e-9` for loss and `8.10e-13` for dlogits in these four replay profiles. Tests
also cover ten normal-route random seeds, finite differences, very negative and
positive tails, zero coefficient, FP32 overflow, stale ownership/versions,
higher-order/Graph rejection and mutation between forward/backward.
The mathematical oracle uses a conservative -36 tail; production retains the
task-29 -16 cutoff. Tiny representable final gradients are checked with an
appropriate explicit alpha; underflow at smaller coefficients is not treated as
a mathematical mismatch. Numerical tolerances and payload invariance are
separate criteria.

`fresh-process-bytes.json` compares original contiguous uint8 payloads, including
signed zero, for logits, logical IDs, bias, loss, counts, masses, selected IDs,
unscaled probabilities and dlogits. Both runs use distinct Triton cache
directories. Hashes locate cases; acceptance also compared actual raw bytes.
Raw snapshots (~7 MB per process), fixtures, sources and logs remain ignored.

`two-mi300x-p4.json` records automatic P4 with active HIP IPC. The separate
`two-mi300x-rccl.json` records a transport-only subclass that disables IPC to
exercise P4's existing NCCL-API/RCCL tensor all-gather. Neither path performs a
floating sum in transport. Both restore global logical token order and compare
loss and each rank's local dlogits byte-for-byte with the single-device complete
domain. This catches rank-local means and accidental world-size multiplication.
Scenarios cover unequal lengths, empty rank, overflow/all drop, empty global
domain, and unbounded capacity. Empty domains are boundaries, not acceptance
samples. Shared-parameter DDP averaging is not tested or silently compensated.

## Reproduction and timing

API, formula, P4 participation/scaling and setup are documented in
[`docs/operators/router-aux-loss.md`](../../../docs/operators/router-aux-loss.md).
The two-process commands used the already verified task-32 bias artifact:

```bash
export AITER_TRITON_ONLY=1
export PYTHONPATH=artifacts/task34/aiter
TRITON_CACHE_DIR=/tmp/rlk-task34-cache-a python -m tools.validation.operators.check_router_aux_loss \
  --bias-fixture artifacts/task32/0731-layer3-bias.pt \
  --raw-output artifacts/task34/raw-a.pt \
  --output reports/experiments/router-aux-loss/task34-mi300x.json
TRITON_CACHE_DIR=/tmp/rlk-task34-cache-b python -m tools.validation.operators.check_router_aux_loss \
  --bias-fixture artifacts/task32/0731-layer3-bias.pt \
  --raw-output artifacts/task34/raw-b.pt --output artifacts/task34/fresh-process-b.json
python -m tools.validation.operators.compare_router_aux_processes \
  --first reports/experiments/router-aux-loss/task34-mi300x.json \
  --second artifacts/task34/fresh-process-b.json \
  --raw-a artifacts/task34/raw-a.pt --raw-b artifacts/task34/raw-b.pt \
  --output reports/experiments/router-aux-loss/fresh-process-bytes.json
```

The committed single-device report measures full auxiliary forward with a
prepared N=257/capacity-2 plan, including normal routing, scale-one normal
probabilities, content hashes, host validation and fixed-order local reduction.
It excludes plan preparation, Gate projection, backward, transport and explicit
file I/O. Five warmups and twenty wall-clock samples give a **4.592-ms wall median**.
This is a correctness profile with substantial host and
duplicate-routing cost, without a speedup or full-model throughput claim.

Global gathering costs O(N×6) storage per rank and requires complete logical
ownership metadata. The optional API is first-order, learned layers 3–42, FP32,
ROCm TP1 and Graph disabled. Hash-layer auxiliary objectives, alternative loss
formulas, MI355X/multi-node measurement, complete framework wiring, distributed
model/shared-parameter autograd and full Flash-0731 raw-logprob signoff remain
outside this task's evidence.
