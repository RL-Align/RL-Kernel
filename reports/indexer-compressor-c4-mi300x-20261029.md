# Item 20 — independent Indexer C4 acceptance, 2026-10-29

Parent task-19 commit: `879305251c37e1b8bb0cb44409f1762752d03561`.
PR target: `dsv4-rocm` (`5ba769f681b7fa765ea6b29f9939454a858390ba`).
The member environment bootstrap built RLK main
`8588158650aff58c9c1f527674de94bad06c3d10` with ROCm Clang and verified
`rl_engine._C.fused_logp`; task tests ran against the task checkout.

## Environment

MI300X `gfx942`; Torch `2.11.0+gitd0c8b1f`; HIP `7.2.53211`; Triton 3.6.0;
aiter 0.1.16.post3; FlyDSL 0.2.0; vLLM 0.26.0+rocm723.

Existing aiter C4/Hadamard/MXFP4 and vLLM FP8 operators are called directly.
Task 19's grouping/state descriptors and canonical backward session are shared.
There is no newly written numerical GPU kernel, duplicate C4 state machine or
generic matrix-multiply Hadamard fallback.

## Acceptance

- **296 tests passed twice**, zero failures/errors/skips.
  First run with coverage: 57.56 s; independent repeat: 44.91 s.
- 25 item-20 CPU tests, 44 item-20 real GPU tests, 85 main-C4 regression tests
  and 142 existing operator-contract tests.
- Each full run reported 16 warnings; warnings are not counted as passed tests.
- Exact byte comparisons cover BF16 output, packed payload and scales across
  chunks, row permutations, padding, CP ownership materializations, training/
  rollout, page boundaries and independent processes.
- Real mLite VJP, native H-transpose, exact gradient comparisons across
  microbatch/reverse orders, genuine FP64 compressor reference and parameter
  finite differences. Quantized payloads deliberately have no gradient.
- Shared-session main/Indexer gradient isolation and cross-branch misuse
  rejection, both CPU and native GPU.
- Native Hadamard tail allocations, previous-output/canary preservation,
  empty/inference inputs, streams, mutation/version/Graph rejection.
- 50 repeated calls: live GPU allocation stayed **134,541,824 bytes** before
  and after; this is live allocation, not a total process-memory bound.
- Ruff lint and format checks passed for the 14 changed/new Python files.

Python statement coverage: **463/505 (91.68%)**; branch coverage **82/92
(89.13%)**, across shared C4 and Indexer wrappers. Native compiled instruction
coverage is not measured. Coverage/autograd tracing gaps are not classified as
all constructor-failure paths.

## Performance (microseconds)

Median of five synchronized 30-repetition wall-time loops after warmup.
Python dispatch and validation are included; binding and quantization are
reported separately. These are not kernel-only latencies or speedup claims.

| Precision | Groups | Bind | Native compress+rotate | Quantize | Full operator |
|---|---:|---:|---:|---:|---:|
| MXFP4 | 1 | 352.45 | 125.13 | 35.04 | 562.92 |
| MXFP4 | 8 | 511.48 | 110.64 | 42.62 | 589.46 |
| MXFP4 | 64 | 1674.79 | 237.77 | 44.43 | 714.58 |
| MXFP4 | 128 | 2997.66 | 381.71 | 43.75 | 890.14 |
| FP8 | 1 | 352.03 | 113.08 | 15.54 | 508.78 |
| FP8 | 8 | 508.17 | 110.00 | 15.53 | 505.23 |
| FP8 | 64 | 1696.73 | 236.97 | 15.96 | 647.97 |
| FP8 | 128 | 2951.27 | 364.55 | 15.74 | 792.66 |

The small-row operator is Python/validation/launch dominated; no end-to-end
training or generation throughput improvement is claimed.

Implementation fingerprints:

- MXFP4: `f4d231272cbbf4d351ddc83244f26fb8a4af20799e3c84b21526b11a7f53df04`
- FP8: `c57b8cf88f38f0911eeddc949678cc1f6f0d3def5e11f366261765c61c666e5d`

The report/log/JUnit/coverage/provenance and benchmark evidence are stored
under the member's mounted `/outputs/item20-20261029/`.

## Limits

No live request/cache commit (item 22), quantized STE/full Indexer training
chain (item 26), Graph/runtime integration (item 27), actual RCCL/GPU CP,
multi-node or full-model acceptance is claimed. The opt-in adapters share the
operator boundary but are not full vLLM/Megatron model registration.
