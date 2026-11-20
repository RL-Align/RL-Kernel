# Task 22 — MI300X Indexer MXFP4 state/cache acceptance, 2026-10-29

Prerequisites: task 20 (`e61f5ee`) and task 21 (`49dc179`), combined at parent
`c966af9a2ede87b803e12b63f60424e8c51b311d`. Target: `dsv4-rocm`.

## Environment and implementation

MI300X `gfx942`; Torch `2.11.0+gitd0c8b1f`; HIP `7.2.53211`; Triton 3.6.0;
aiter 0.1.16.post3; FlyDSL 0.2.0; vLLM 0.26.0+rocm723.

Reuses the task-21 lifecycle and native state/compression backend, task-20
native Hadamard/MXFP4 postprocessor, and Torch's existing masked scatter.
The new Indexer state backend/model/contract are small specializations; no
numerical GPU kernel or duplicate lifecycle implementation was added.

## Acceptance

- **434 tests passed twice**, zero failures/errors/skips.
  Coverage run: 75.32 s; independent repeat: 61.35 s.
- New task-22 tests: 19 CPU and **30 real GPU tests**.
- The combined run also includes task 20, task 21, main C4 and existing
  operator-contract regressions.
- Each run reported 17 warnings.
- Payload and E8M0 scale bytes match the standalone task-20 native pipeline
  across prefill/chunks/decode, multiple seeds, request interleaving, page
  boundaries, model replacement and real Graph replay.
- Final FP32 partial-state bytes are checked separately against direct
  `[kv,score+ape]` construction.
- Incomplete/invalid rows retain payload 0 and scale 127. Graph warmup never
  commits a row; released pages/scales are reinitialized before reuse.
- Thread races, original-input alias reuse, cross-stream handoff, private
  snapshots, stale handles and duplicate leases are covered.
- Failure injected between payload and scale writes poisons the arena without
  advancing the cursor; partial results cannot be read/retried through the API.
- Fixed Graph buffer addresses and 40 release/recreate cycles show no
  live-allocation growth. A decrease due to delayed allocator cleanup is allowed,
  not misclassified as a leak.
- Changed-file Ruff lint and format pass.

Scoped Python state-wrapper coverage: **451/458 statements (98.47%)** and
**29/30 branches (96.67%)**. Native instruction coverage is not measured.

## Stateful append latency

Microseconds, median of five synchronized 30-append loops after warmup.
Includes validation, staging, compression, Hadamard, quantization, payload/scale
commit and device completion; not raw kernel latency or model throughput.

| Graph | Tokens/append | Append (us) |
|:---:|---:|---:|
| off | 1 | 687.29 |
| off | 256 | 789.91 |
| on | 1 | 300.93 |
| on | 256 | 402.57 |

The 256-token cases traverse 39,168 tokens / 9,792 complete rows. General
end-to-end speedup is not claimed.

Evidence, source hashes, JUnit, coverage and benchmark JSON are in the member's
mounted `/outputs/item22-20261029/`.

## Scope

This is an explicit unshuffled row-major MXFP4 cache boundary, not an implicit
swizzled cache replacement. No cache-write autograd, full quantized VJP,
distributed CP/RCCL, multi-node, full framework registration or full model
acceptance is claimed.
