# Task 33: router backward and result records

Contract: [P3-B RFC #516](https://github.com/RL-Align/RL-Kernel/issues/516), item 33.
Base: `RL-Align/RL-Kernel:dsv4-rocm` at
`5ba769f681b7fa765ea6b29f9939454a858390ba`.

Discrete expert IDs have no derivative. Normal selection is checked independently
before numerical weights and gradients; saved-route replay remains diagnostic.
The task-32 complete-domain/global-logical-token-ID capacity policy is preserved.

## Validation

|Check|Result|
|---|---|
|MI300X operator/training tests|72 passed|
|CPU record/adversarial tests|25 passed|
|Exact dispatch/source-failure tests|10 passed|
|Existing registry regression|12 passed|
|Total pytest|119 passed, no skips|
|Real-bias/table normal routing|54 cases: nine profiles × M=1/16/17/32/65/257|
|Fresh processes/separate Triton caches|165 tensors; 4,098,624 original payload bytes equal|
|RCCL two-MI300X records|Five scenarios per rank passed, with rank and model/route identity checked|
|Normal-route finite differences|Recomputed selection/capacity; selected, overflow and unselected logit coordinates|
|Frozen Gate → BF16 upstream LoRA|K=32 and actual K=4096, rank 4, scale 0.5; two SGD updates with old-plan rejection|
|Two ROCm streams|Shared plan, independent result/VJP records and exact payload agreement; no overlap claim|
|50 repeated calls|Unchanged live GPU allocation and matching normal records|
|Companion applicability|Five clean patches; all eleven AITER source hashes verified|

The pytest warning is from the intentional unsupported-Graph test ending an empty
capture. Ruff check/format, patch whitespace and strict MkDocs validation are part
of the final checks.

Hardware/software: MI300X gfx942, Torch `2.12.0+rocm7.14.0a20260608`, ROCm
`7.14.60850`, Triton `3.7.0`. The isolated AITER base is
`326cffa38c9eda16bef19375ba55334e4d3acd59`; no installed package is modified.
Task-29/30/31/32 companion patches/manifests are byte-identical to their existing
PRs. Task 33 adds only an AITER replay binding; GPU numerical work reuses existing
selected-weight forward/VJP implementations.

The pinned mLite source `327a238239448de6771beea185dc6902a8de3eaa` uses
`_VLLMFixedRouteFunction` to save discrete IDs for conditional weight backward.
Its source SHA256 is
`e9b5bb09a083a086775f13de333ceb64a4146dd1d09b42c3c3c65e43955958d2`.
This structural reference does not replace normal selection or claim execution
of the CUDA/mLite runtime on this ROCm host.

## Evidence boundaries

`task33-mi300x.json` covers normal learned routing with unbounded/2/zero capacity,
and all three Hash layers with and without normalization. It uses synthetic FP32
logits and the real fixed Flash-0731 layer-3 bias and three Hash tables, verified
against frozen tensor checksums before GPU use. The checkpoint revision is
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`. No checkpoint shard or full-model
activation trace is used. Capacity-2 microbatch cases explicitly include a token
with both surviving and overflowing slots.

Every profile checks normal global expert IDs before its independent FP64 weight
and analytic VJP reference. Maximum absolute errors across the nine profiles are
`4.40e-7` for weights and `2.00e-7` for dlogits. The mathematical oracle uses a
conservative -36 tail, while the production score contract uses -16. Repeated,
inference/training and microbatch byte tests are separate from mathematical
tolerances. BF16 LoRA gradients have a separate dtype-aware comparison.

The new source-checkout Gate compatibility handling reports missing generated
AITER version metadata honestly through source hashes; numerical kernels and
Gate arithmetic are unchanged. The model bridge freezes base Gate/bias and
checks their content. This validates a local upstream LoRA boundary, not a full
framework LoRA attachment/topology or parameter-accumulation signoff.

`fresh-process-bytes.json` compares actual uint8 payloads, including signed zero,
for input logits/logical IDs/upstreams and normal output weights/experts/dlogits.
Forward and VJP record hashes also agree. The two runs used distinct Triton cache
directories. Raw snapshots are approximately 4 MiB each in ignored artifacts.

`two-mi300x-records.json` uses real RCCL all-gather with interleaved logical IDs
and differing complete-domain assembly orders. Both ranks agree on plans and
independently validate normal local choices/weights/VJPs. It includes overflow,
empty rank, all drop, empty domain and unbounded capacity. Empty records are
explicit boundary cases without normal-acceptance samples. Replayed records
never receive acceptance, including bitwise matches.

## Reproduction and costs

Use the commands and API descriptions in
[`docs/operators/router-training.md`](../../../docs/operators/router-training.md).
The exact commands used here reused verified artifacts from tasks 30/32:

```bash
export AITER_TRITON_ONLY=1
export PYTHONPATH=artifacts/task33/aiter
TRITON_CACHE_DIR=/tmp/rlk-task33-cache-a python -m tools.validation.operators.check_router_training \
  --bias-fixture artifacts/task32/0731-layer3-bias.pt --hash-fixture-dir artifacts/task30 \
  --record-dir artifacts/task33/records-a --raw-output artifacts/task33/raw-a.pt \
  --output reports/experiments/router-training/task33-mi300x.json
TRITON_CACHE_DIR=/tmp/rlk-task33-cache-b python -m tools.validation.operators.check_router_training \
  --bias-fixture artifacts/task32/0731-layer3-bias.pt --hash-fixture-dir artifacts/task30 \
  --record-dir artifacts/task33/records-b --raw-output artifacts/task33/raw-b.pt \
  --output artifacts/task33/fresh-process-b.json
python -m tools.validation.operators.compare_router_training_processes \
  --first reports/experiments/router-training/task33-mi300x.json \
  --second artifacts/task33/fresh-process-b.json \
  --raw-a artifacts/task33/raw-a.pt --raw-b artifacts/task33/raw-b.pt \
  --output reports/experiments/router-training/fresh-process-bytes.json
```

The recorded learned **logit-route** forward at M=17, using a prepared N=257
capacity-2 domain, has a **2.583-ms wall median** (five warmups, twenty samples).
This includes normal routing, CPU snapshot/validation and backward-guard content
hashes. It excludes Gate projection, VJP serialization and explicit file I/O.
There is no end-to-end throughput or speedup claim. VJP records store O(M×256)
payload bits and should be scoped explicitly by the caller. Source downloads,
fixtures, detailed route/VJP records, raw outputs and logs remain ignored.

No distributed autograd, expert dispatch, task-34 auxiliary loss, complete
Megatron/vLLM/verl/vime wiring, MI355X/multi-node/full-model raw-logprob acceptance
is claimed by these operator results.
