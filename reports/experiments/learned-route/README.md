# Task 32: learned router, capacity and overflow

Contract: [P3-B RFC #516](https://github.com/RL-Align/RL-Kernel/issues/516), item 32.
Base: `RL-Align/RL-Kernel:dsv4-rocm` at
`5ba769f681b7fa765ea6b29f9939454a858390ba`.

The frozen capacity policy, confirmed for this task, is
`complete_domain.global_logical_token_id_ascending.v1`: within each complete
layer/domain retain tokens by global logical token ID, mask overflow in the
original Top-6 slots, then normalize surviving **pre-bias** scores in slot order.
Changing the actual competing domain may change admission; changing compute
microbatch shape while retaining the same domain/plan does not.

## Validation

|Check|Result|
|---|---|
|Operator, dispatch, fixture and registry pytest|122 passed, no skips|
|Normal routing with fixed real checkpoint bias|18 cases: capacities None/2/0 × M=1/16/17/32/65/257|
|Microbatch/inference-training consistency|Exact FP32 forward and VJP payload bits; exact IDs/masks/positions|
|Fresh processes and separate Triton caches|18 cases, 126 tensors, 2,539,848 raw payload bytes equal|
|Independent normal-routing oracle|Exact discrete selection/admission; FP64 weight/VJP checks passed|
|Shared plan across two CUDA streams on ROCm|Exact forward/VJP/dispatch results; stream ownership check, no overlap claim|
|Repeated normal routing|50 calls, unchanged plan identity and live GPU allocation|
|Real two-MI300X RCCL|5 cases per rank: overflow, empty rank, all drop, empty domain, unbounded|
|Companion patch portability|Three clean applications; all six source fingerprints match|
|Prerequisite profiles|Task-29 and task-31 patch/manifest bytes unchanged|
|Ruff and strict MkDocs|Passed|

The pytest warning comes from the intentional Graph-rejection test ending an
empty capture. MI300X gfx942, Torch `2.12.0+rocm7.14.0a20260608`, ROCm
`7.14.60850`, Triton `3.7.0` were used. Source/API/launch fingerprints and precise
execution details are embedded in the JSON reports.

`task32-mi300x.json` uses synthetic FP32 gate logits with the real, fixed
`layers.3.ffn.gate.bias` from Flash-0731 revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`. Its 1-KiB raw SHA256 is
`34d941394db670f6de58ccb9bb781158d12407cc27613f8ae90756e2623fb236`.
The downloader and independent fixture loader check that constant, not merely a
checksum supplied by the fixture. No full checkpoint shard is downloaded.

Across these three complete-domain cases, maximum absolute weight error versus
the mathematical FP64 oracle is `9.13e-8`; maximum VJP error is `1.37e-8`.
The mathematical oracle uses a conservative -36 tail; production uses the pinned
task-29 -16 cutoff. Unit tests separately exercise extreme tails and gradients,
including the denominator clamp. Mathematical tolerances do not substitute for
the exact microbatch and repeated-execution checks.

`fresh-process-bytes.json` records actual uint8 payload equality, including signed
zero, for logits, logical IDs, weights, expert IDs, masks, positions and dlogits.
Hashes are included for locating inputs and outputs; equality was checked against
the original tensors saved by both processes. These ~2.5-MiB raw files and the
bias/source downloads remain in ignored `artifacts/task32`.

`two-mi300x-capacity.json` covers real RCCL all-gather of complete domains with
different row assembly orders per rank. Both ranks produce the same logical
capacity plan and independently validate normal local routing and local VJPs.
It does not validate distributed autograd or expert dispatch communication.

## Timing disclosure

Capacity 2, N=257, 256 experts; isolated process, ten warmups. Wall medians from
the committed primary replay:

|Operation|Wall median|GPU-event median|Samples|
|---|---:|---:|---:|
|Complete-domain plan preparation|2.826 ms|2.807 ms|10|
|Full RLK normal forward|3.176 ms|3.159 ms|30|
|AITER normal selection + post-mask weights|0.147 ms|0.133 ms|30|
|AITER local VJP|0.054 ms|0.041 ms|30|

Host validation, CPU admission, content hashing, copies and metadata handling
make the adapter substantially more expensive than the arithmetic kernels.
GPU events can include gaps while the host submits work. These are operator
measurements, with no full-model throughput or latency claim. The companion
profile is correctness focused and has no M-dependent autotuning.

Reproduction commands, API example, semantics and limitations are in
[`docs/operators/learned-route.md`](../../../docs/operators/learned-route.md).
Full-model logprob acceptance, framework integration, MI355X measurement,
distributed autograd, task-33 replay integration and task-34 auxiliary losses
remain outside this task's results.
