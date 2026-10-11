# Qwen3 Dense refactor acceptance: CUDA and ROCm

Baseline: upstream `main` at `bea224b`. Candidate: the exact commit of the
`refactor` PR under review. Run both commits on each target platform before
approving the refactor. No new model or native MUSA implementation is certified
by this directory change.

## Required result

Within each supported configuration, independently recomputed training and
rollout selected logprobs must have **zero raw-byte/bit mismatches**. Compare
logical active tokens after unpadding, with token/sample IDs aligned. Positive
and negative zero are distinct. NaN/Inf, missing tokens, missing provenance,
skipped required cells, and silent fallback cannot count as a pass. The training
score must not reuse the rollout logprob value.

Compare each commit's WS1/WS2 outputs, chain intermediates, loss, gradients and
updates for the scope already supported by the baseline contract. Preserve the
baseline's accuracy tolerances against FP32 references; invariance remains zero
tolerance. Record unsupported cells explicitly instead of inventing new coverage.
CUDA and ROCm are separate acceptance matrices; this does not claim byte equality
between vendors or automatically certify every topology on either vendor.

Performance is **report only**: report latency/throughput/memory differences and
let reviewers decide whether they are acceptable. There is no agreed automated
percentage threshold. Numerical failure is not waived by a performance gain.

## Freeze the comparison environment

Use separate clean checkouts and separate extension builds for baseline and PR.
Record both Git SHAs, GPU model/count, driver, CUDA/ROCm version, PyTorch, Triton,
AITER/FlashAttention, vLLM, Megatron, VIME and companion patch hashes. Match build
flags, clocks/power limits, dtype/TF32 policy, allocator/graph/cache settings,
seeds, weights and optimizer state. Do not let an editable install or stale
`rl_engine._C` resolve to the other checkout.

Use the pinned Qwen3-8B revision, config, tokenizer and verified real weight
snapshot. Match token IDs, masks, sequence lengths, packing, sampling support,
temperature/top-p/top-k, training TP/CP and rollout TP/CP. Keep machine-local
profiles in `.rlk-profile.json` or `configs/local`, with secrets untracked.
The original pinned precision/workload JSON bytes are unchanged by this PR.

## Execution matrix

| Scope | CUDA | ROCm | Evidence |
|---|---|---|---|
| WS1 operators | Existing CUDA and Triton profiles | Existing ROCm backend/operator cases | Four judgments, raw mismatches, actual backend and configuration |
| WS2 distributed | Supported TP/CP and collective cases | Supported TP/CP and RCCL cases | Rank/topology, partition layout, communication and reduction order |
| Dense chain | Full Qwen3-8B, real weights | Full Qwen3-8B, real weights | Intermediate/score/loss/gradient scope supported by baseline |
| Integration | VIME + vLLM rollout + Megatron train | Same supported engine configuration | Independent scoring, integration readbacks, weight publication/version |
| Ablations | P/P, R/R and supported mixed arms | Same supported arms | Positive controls plus mismatch/negative controls |
| Performance | Fixed replay and actual training separately | Fixed replay and actual training separately | Raw repeated samples, warmup count, median and spread |

The existing CUDA WS1 scripts retain their profiles and fail-closed behavior:

```bash
PY=python python ci/run.py ws1
PY=python python ci/run.py ws1-chain
```

Do not run the CUDA WS1 script on ROCm and treat skipped CUDA cases as ROCm
coverage. ROCm operator/distributed entry points are in
`tools/validation/distributed`, `tests/backends/rocm`, and
`benchmarks/distributed`; use the same cases supported by the baseline.

The shared Dense launcher remains available on both platforms. First configure
`.rlk-profile.json` for the machine and inspect the command:

```bash
./rlk plan --tp 2 --cp 4 --rollout-tp 4 --temperature 0.7 --top-p 0.95 --steps 200
./rlk run  --tp 2 --cp 4 --rollout-tp 4 --temperature 0.7 --top-p 0.95 --steps 200
```

These are the existing eight-GPU example settings, not a new claim of support
for an untested machine. Use a baseline-supported topology on the actual hardware.
CUDA's `verify` command and ROCm's `run` validation retain their existing scope;
ROCm `verify` is still unsupported. The
[existing platform audit](../usage/cuda-rocm-consistency-audit.md) explains the
limits of older evidence. Historical results do not validate this PR.

## Performance report

Warm up both revisions using the same count, synchronize measurement boundaries,
then retain multiple repeated samples in the same workload/environment. Compare
median milliseconds using `(candidate / baseline - 1) * 100`; higher is slower.
For throughput, report `(candidate / baseline - 1) * 100`; higher is faster.
Include dispersion, peak allocated/reserved memory, graph compilation and setup
cost separately. Do not mix synthetic replay latency with end-to-end throughput.

Use `benchmarks/operators`, `benchmarks/layers`, `benchmarks/distributed`, and
`benchmarks/e2e` for the corresponding scopes. Save local files under
`artifacts/refactor/<platform>/<sha>/`. Attach a report with:

| Platform / GPU | Workload / topology | Baseline SHA | Candidate SHA | Bit mismatches | Median baseline / PR | Difference | Memory | Decision |
|---|---|---|---|---:|---|---|---|---|
| CUDA | Pending | bea224b | PR SHA | Pending | Pending | Pending | Pending | Reviewer |
| ROCm | Pending | bea224b | PR SHA | Pending | Pending | Pending | Pending | Reviewer |

The PR can be used for these runs while open. Merge acceptance remains pending
until the required GPU artifacts and reviewer decisions are attached.
