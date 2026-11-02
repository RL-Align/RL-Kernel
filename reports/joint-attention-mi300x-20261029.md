# Item 24 MI300X acceptance — 2026-10-29

Original October 29 base: main `8588158650aff58c9c1f527674de94bad06c3d10`.
This task does not depend on pending item-13/23 implementation commits.
The requested PR target remains `dsv4-rocm`, which predates that main.

Environment: MI300X/gfx942; Torch `2.11.0+gitd0c8b1f`, HIP `7.2.53211`,
Triton `3.6.0`, aiter `0.1.16.post3`, vLLM `0.26.0+rocm723`.
The pinned external aiter source and installed native vLLM cache source are
fingerprinted in the JSON report. No new numerical GPU kernel was written.

November 3 forward-only refresh: merged main `d00d444b46858b5109acafd29b5e6af70ac73370`
and ported the forward fixes also used by the stacked item-25 branch. The
target remains `dsv4-rocm`; these changes do not depend on item-25 backward.

## Acceptance

**236 passed, zero skipped, 14 warnings**, twice in fresh processes:
28.00 seconds with coverage, 12.57 seconds without.
Includes **94 new task tests** and 142 contract regressions.
Ruff format/check passes for all new Python files. The explicit provider
provisioning helper verified the installed source hash.

```bash
export RLK_AITER_DSV4_SPARSE_KERNEL_SOURCE=/workspace/upstream/aiter-sparse-326c/kernels.py
python3 -m pytest tests/contracts/operators tests/models/deepseek_v4 -q \
  --disable-warnings --basetemp=/outputs/item24-20261029/repeat-temp \
  --junitxml=/outputs/item24-20261029/repeat-tests.xml
```

Tests cover:

* FP64 independent softmax O/LSE oracle with a shared denominator and no
  sink V, using numerical tolerances appropriate to the native BF16 P cast.
* Exact bytes across forward modes, inference mode, chunking, query
  permutation, compressed Top-K permutation and physical KV loader order.
* Empty input and segments, disabled `-inf` sink, sink-only attention,
  padded queries and padded heads, D128/D512 and C4/C128 boundaries.
* The full 512-compressed + 128-window selected capacity at H128/D512.
* Real captured-kernel replay with changing Q, output/saved-state ownership,
  stable workspace addresses, two CUDA streams, independent plans and
  concurrent host threads.
* vLLM native OCP/FNUZ packed-cache encode/decode, block sizes 64/128,
  scattered physical slots, and byte-identical forward using the same
  decoded BF16 data through contiguous and packed ingress.
* Index/owner/causality/window completeness, source hash, version,
  metadata mutation, buffer failure poisoning and packed-cache ABI guards.

Coverage: **294/300 production statements (98.00%)**, **27/30 branches
(90.00%)**, across the five new runtime modules. Native instructions,
benchmark and provisioning script are not counted as runtime coverage.
The FP64 backend is explicitly test-only; production fails closed.

On November 3, **254 passed, 14 warnings, zero skipped** for
`tests/contracts/operators` and the two forward test modules, with
`RLK_AITER_DSV4_SPARSE_KERNEL_SOURCE=/cache/dsv4/sparse_mla_dsv4_train.py`.
Ruff check and format passed. This run covers the pinned-source TOCTOU guard,
packed D512 admission, independent caller-key selection oracle with an
intentionally incorrect window-row regression, and forward buffer geometry
added after the original coverage measurement; the October 29 coverage
percentage is historical.

## Timings

November 3: 30 repetitions after warmup. Eager measurements precede Graph
capture; captured measurements follow it. HIP events and synchronized wall
time are reported from the same pass. These are provider-call measurements,
not isolated instruction timing; host/launch gaps can appear inside event
intervals. The eager and captured core measurements now do equal work,
including the same padding masks; the original October 29 upstream-only
measurement was not an equal-work comparison.
Metadata binding and cache dequantization are excluded. Forward-record
timings include finite admission checks, copies, event ownership and clones.

| M | H | D | Selected | Eager core event µs | Captured core event µs | Checked eager wall µs | Checked Graph wall µs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 128 | 6 | 64.34 | 19.61 | 476.19 | 389.74 |
| 32 | 64 | 512 | 6 | 63.36 | 32.56 | 533.71 | 442.26 |
| 128 | 64 | 512 | 6 | 62.76 | 46.09 | 499.85 | 458.21 |
| 1 | 128 | 512 | 640 | 95.93 | 99.23 | 591.24 | 473.84 |

No baseline-engine speedup is claimed. Python/admission/snapshot costs
dominate the checked API; the measurements deliberately keep those costs
visible instead of reporting the native kernel as the full integration.

Evidence: `/outputs/item24-20261029/` contains JUnit, two complete test logs,
coverage JSON, benchmark JSON, source SHA256 manifest and the initial
native/Graph probe. A compact machine-readable report is committed beside
this document. The November 3 JSON records the rerun and its tested source
hashes, independently of the October 29 evidence directory.

## Non-claims

This is the **forward and saved-state** task. No backward/autograd signoff
is inferred from it; that remains item 25. No whole-engine Graph capture,
CP/DCP, distributed transport, full model, cross-parallel-degree equality,
lossless FP8 cache equivalence, or larger-than-640/split-K profile is claimed.
