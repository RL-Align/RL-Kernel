# Item 25 MI300X acceptance — 2026-10-29

Prerequisite: item 24, commit `159b0592be933b65aac76ff8b9fe411fa3f27456`
(PR #628), on main `8588158650aff58c9c1f527674de94bad06c3d10`.
The requested target is `dsv4-rocm`; the pending forward prerequisite appears
in the stacked PR diff.

Environment: MI300X/gfx942, Torch `2.11.0+gitd0c8b1f`, HIP `7.2.53211`,
Triton `3.6.0`, aiter `0.1.16.post3`.
Numerical kernels are reused aiter kernels. Fixed sink tiles use native Torch
operations; no new numerical HIP/Triton kernel was written.

## Tests

**306 passed, zero skipped, 14 warnings**, in two fresh processes:
26.45 seconds under coverage and 13.36 seconds under default autograd threading.
There are **70 new tests: 26 CPU and 44 real ROCm**, plus 236 prerequisite/
contract regressions. Ruff checks pass on every new/changed Python file.

Tests cover:

* Separate dQ, shared K=V dKV, and sink gradients against independent FP64
  autograd math, using explicit native-VJP numerical tolerances.
* Exact gradient bytes under changed microbatch sizes, query permutation,
  reversed backward arrival, padded rows, and shared repeated KV.
* A 65-query fan-in crossing both CSR64 and sink32 boundaries.
* H8/H13/H64/H128, D128/D512, and 640 selected entries.
* Empty input, truly empty KV storage, all-padding, disabled sink and
  no-attention zero VJPs.
* Real fixed-record backward Graph replay, changing dO, two streams,
  saved-state/CSR mutation guards and result ownership.
* Q-only, KV-only and sink-only autograd; provider/version/identity/session
  barriers; distinct event-owned plan and mixed-VJP-profile rejection.
* Upstream source hashes, native workspace accounting, explicit provisioning
  behavior and no-overwrite-on-hash-mismatch.

Default-threading repeat:

```bash
export RLK_AITER_DSV4_SPARSE_KERNEL_SOURCE=/workspace/upstream/aiter-sparse-326c/kernels.py
export RLK_AITER_DSV4_SPARSE_WRAPPER_SOURCE=/workspace/upstream/aiter-sparse-326c/wrapper.py
python3 -m pytest tests/contracts/operators tests/models/deepseek_v4 -q \
  --disable-warnings --basetemp=/outputs/item25-20261029/repeat-temp \
  --junitxml=/outputs/item25-20261029/repeat-tests.xml
```

## Coverage instrumentation caveat

The ordinary coverage tracer missed Python VJP callbacks launched by native
autograd worker threads, even though their numerical tests passed. For the
**coverage-only run**, autograd worker multithreading was disabled so callbacks
ran on the calling Python thread. Native GPU kernels, numerical assertions
and Graph tests were unchanged. The independent repeat and all benchmarks
used default autograd threading.

Measured scope: the four new runtime modules, **306/306 statements (100%)**
and **42/42 branches (100%)**. This does not measure native GPU instructions
and does not imply that all possible inputs or errors are covered. No source
line was excluded.

Equivalent coverage invocation:

```python
import coverage
import pytest
import torch

cov = coverage.Coverage(
    branch=True, include=["*/joint_backward.py", "*/joint_autograd.py"],
)
cov.start()
with torch.autograd.set_multithreading_enabled(False):
    status = pytest.main([
        "tests/contracts/operators", "tests/models/deepseek_v4", "-q",
    ])
cov.stop()
cov.save()
cov.report()
raise SystemExit(status)
```

## Timings

20 repetitions after warmup. Native pipeline/core timings exclude admission,
CSR preparation and record construction. Checked VJP timings include input
checks, dO copies, events and result clones. Canonical forward/backward timing
includes the full checked forward and autograd collection/allocation/sorting.
Event measurements can include host-launch gaps; they are not instruction-only
latencies or baseline-engine comparisons.

| M | H | D | Selected | Graph core event µs | Checked Graph VJP wall µs | Canonical forward+backward wall µs |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 13 | 128 | 6 | 67.83 | 153.87 | 3206.02 |
| 7 | 64 | 512 | 6 | 84.91 | 165.34 | 3294.05 |
| 65 | 13 | 128 | 6 | 75.87 | 173.87 | 3370.60 |
| 1 | 128 | 512 | 640 | 442.07 | 523.88 | 3356.69 |

The checked canonical adapter has material Python/admission/allocation overhead.
No full-engine speedup is inferred from the native-kernel measurements.
Machine-readable timings, provider fingerprints and tested source hashes are
committed in the adjacent JSON report.

## Explicit non-claims

Fixed-record native backward Graph is not a captured whole autograd training
step. CP/TP exchange, checkpoint orchestration, full framework/model execution,
higher-order derivatives and gradients through quantized cache bytes are not
signed off. This report does not complete the remaining runtime or model tasks.
