# Task 21 — MI300X state/cache acceptance, 2026-10-29

Parent: `879305251c37e1b8bb0cb44409f1762752d03561` (task 19 / PR #618).
Target: `dsv4-rocm`. Source lives in the member workspace; logs and evidence
are under the mounted `/outputs/item21-20261029/`.

## Environment and reuse

MI300X `gfx942`; Torch `2.11.0+gitd0c8b1f`; HIP `7.2.53211`; Triton 3.6.0;
FlyDSL 0.2.0; aiter 0.1.16.post3; vLLM 0.26.0+rocm723.

- Existing aiter compression/RMSNorm/RoPE/cache scatter.
- Existing vLLM FP32 packed partial-state writer.
- Preallocated layout-only reader copies and Torch Graph capture/replay.
- No new numerical GPU kernel.

The explicit single-wave C128 profile is distinct from task 18's eight-wave
HCA profile. An arithmetic handshake rejects mixing them. The tests below do
not claim equivalence to a different reduction recipe.

## Test results

- Full targeted contract + DeepSeek-V4 suite: **316 passed twice**, zero
  failures/errors/skips; 54.58 s with coverage, 39.50 s independent repeat.
- 39 new CPU tests and **50 real GPU tests**; 85 C4 prerequisite regressions
  and 142 existing operator-contract regressions.
- Each full run reported 16 warnings.
- Genuine FP64 pool/norm/rotation numerical reference; exact FP32 ring bytes.
- Exact cache/state bytes across full input, uneven chunks, single-token decode,
  real Graph replay, multiple seeds, interleaved requests and page boundaries.
- Request generation/reuse, writer ownership, version replacement, duplicate
  lease rejection, incomplete groups, original input alias reuse and output
  snapshot isolation.
- Thread races, cross-stream staging handoff, no writes during Graph warmup,
  stable Graph addresses and 40 release/recreate cycles without live-allocation
  growth.
- Fault injection after mutable work, release and model replacement verifies
  fail-closed poisoned arenas rather than unsafe retry.
- Ruff lint and format passed for all ten new Python files.

Python statement coverage: **316/320 (98.75%)**; branch coverage **27/28
(96.43%)**. Native GPU instruction coverage is not represented by this report.

## Stateful append latency

Microseconds; median of five synchronized loops of 30 appends after warmup.
Includes host validation, staging, native execution and device completion.
The table is not raw kernel latency, full-model throughput or an overlap claim.

| Ratio | Graph | Tokens/append | Append (us) |
|---:|:---:|---:|---:|
| 4 | off | 1 | 369.13 |
| 4 | off | 256 | 449.62 |
| 4 | on | 1 | 244.97 |
| 4 | on | 256 | 351.54 |
| 128 | off | 1 | 354.77 |
| 128 | off | 256 | 426.40 |
| 128 | on | 1 | 251.08 |
| 128 | on | 256 | 354.24 |

The prefill cases process 39,168 consecutive tokens in each arena, validating
capacity/page traversal beyond the shorter numerical test fixtures. The report
does not claim arbitrary shapes or precision profiles are validated.

## Remaining scope

No actual distributed CP/RCCL transport, multi-node, full framework registration,
full model, cache-write autograd or compatibility with the eight-wave C128
profile is claimed. Those require their own integration/acceptance. This PR
implements the explicit state/cache boundary described in its contract.
