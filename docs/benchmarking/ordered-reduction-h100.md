# H100 ordered-reduction results

The [ordered-reduction helpers](../architecture/ordered-reduction.md) preserve a
fixed arithmetic tree in ordinary Triton and optional Gluon enclosing kernels.
The benchmark compares them with **ordinary `torch.sum`** and **native `tl.sum`**;
the explicit same-tree PyTorch implementation is a correctness reference, not a
speedup baseline.

This page reports the test-suite cleanup regression committed as
`c77d2edaf50fac604e3cf79444b72ffc055876b9`, also reported in
[PR #595](https://github.com/RL-Align/RL-Kernel/pull/595). It supersedes the earlier
confirmation run. The A40 pytest suite and H100 benchmark ran against the same
cleanup snapshot before commit. The attachment maps that snapshot to this
commit: production and benchmark sources are byte-identical; two test files
received formatting-only changes with identical Python ASTs.

## Environment and scope

- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0, 132 SMs; PCI bus `00000000:E5:00.0`.
- NVIDIA driver: 595.91.07; PyTorch CUDA runtime: 13.0.
- Python 3.12.3, PyTorch 2.13.0+cu130, Triton 3.7.1.
- Run completed on 2026-10-31 at 16:34:50 UTC (2026-11-01 00:34:50 Asia/Shanghai).

The run covered **39 shape/dtype cases and 159 provider measurements**. Inputs
are FP16/BF16/FP32; all providers accumulate and output FP32. Arithmetic tiles
contain 1024 elements, with initial groups of 8. The reduction operates on the
final dimension of `[rows, length]`. Cases include small and intermediate
widths, non-power-of-two tails, large matrices and physical suffix padding.

Provider settings were frozen from earlier H100 tuning before this confirmation;
there was no parameter search during the run. The comparisons below use the
preselected parallel Gluon caller for `[16,262144]`, and row-only Gluon elsewhere.
The extra row-only Gluon measurements for the long-row cases remain in the raw
report. This is not an automatic library dispatch policy or a universally optimal
configuration.

Product and custom callbacks passed correctness tests; their performance was not
tuned in this run. Results do not establish ROCm, other GPU models, arbitrary
fusion, full-model training/rollout parity, or cross-compiler reproducibility.

## Measurement method and correctness

These numbers are **CUDA Graph kernel timings**. Inputs, outputs and scratch are
preallocated. Compilation, input creation, accuracy checks and Graph capture are
excluded. Providers share at least 250 ms of warmup and run in shuffled order,
recorded per round. Inputs are reused without explicit cache eviction. No
`torch.compile` is used.
Both stages of a parallel implementation are included. Eager host dispatch and
allocation costs are not represented by these figures.

Each provider uses six rounds of 40 samples, with 64 invocations per replay for
small/medium cases and 16 for `[1024,262144]`. Reported latency is the median of
the round medians. Throughput inputs are uniform in `[0,1)`, seed 314159;
separate tests cover cancellation and other numerical stress cases. Every
comparison within a table row comes from this same confirmation run. The largest
maximum/minimum round-median ratio was 1.0272. All measurements stayed below the
1.15 spread warning threshold.

Before timing, candidates passed independent FP64 accuracy, sampled fixed-tree
PyTorch bitwise checks and all-row equality across ordered schedules. Graph
replay overwrote NaN-poisoned output and was checked bitwise against the eagerly
launched result.

The full regression ran on **A40: 1455 tests passed, with zero skips, failures
or errors**, in 639.08 seconds. Python, PyTorch and Triton versions match the
H100 environment above. The suite consolidates duplicate parameter combinations
and keeps backend launchers beside their tests. It retains fixed-tree and
execution/prefix invariance, independent accuracy, FP64 constexpr/runtime merge
seeds, empty-fold storage/autograd, interpreter/compile and Graph replay checks.
Benchmark tests also reject a no-op replay and verify recorded execution order.

The H100 step reran all 39 benchmark cases, including their correctness gates;
it did not repeat the complete pytest suite. Production reduction arithmetic
and public APIs are unchanged by this cleanup.

## Representative results

Latency in microseconds; **lower is better**. The native Triton provider and
ordered execution settings for each case are recorded in the frozen plan.
Long-row Gluon uses tile producers followed by the ordered merge; other Gluon
rows use one program per input row.

| Shape | Input | torch.sum | Native tl.sum | Ordered Triton | Ordered Gluon |
| --- | --- | ---: | ---: | ---: | ---: |
| 128×8193 | FP16 | 3.046 | 1.887 | 3.117 | 2.034 |
| 128×8193 | BF16 | 3.034 | 1.861 | 3.137 | 2.014 |
| 128×8193 | FP32 | 3.306 | 2.000 | 3.287 | 2.259 |
| 16×262144 | FP16 | 8.996 | 4.602 | 5.127 | 5.007 |
| 16×262144 | BF16 | 8.933 | 4.583 | 5.131 | 5.040 |
| 16×262144 | FP32 | 10.077 | 4.711 | 5.416 | 5.197 |
| 1024×262144 | FP16 | 171.630 | 173.259 | 173.143 | 169.158 |
| 1024×262144 | BF16 | 171.638 | 173.167 | 173.198 | 169.214 |
| 1024×262144 | FP32 | 338.080 | 338.987 | 336.998 | 335.598 |

Percent overhead means `100 × (ordered latency / baseline latency - 1)`.
The figures and raw report include every case, including slower results.
Gluon is not consistently faster for every small input.

Few rows and a very long reduction require enough independent programs. For
`[16,262144]`, a row-only Gluon launch creates just 16 programs; the parallel caller
creates 4096 fixed-tile producers. This changes the execution schedule without
changing the tree or merge order. The grid and measurements support this
explanation; instruction-level hardware counters were not used to attribute all
remaining latency to a particular instruction.

The [PR figures](https://github.com/RL-Align/RL-Kernel/pull/595) and attachment use
these same final measurements. The latency figure shows microseconds on a log
axis, where lower is better. The speedup figure uses **baseline latency divided
by ordered latency**: above 1 is faster, below 1 is slower. Each dtype is shown
separately.

## Acceptance and remaining overhead

The agreed tolerance was no more than 10% slower than each native baseline,
allowing up to 0.5 µs absolute excess when that baseline is at most 10 µs. The
wider allowance applies. Each baseline is assessed separately.

| Implementation | Against torch.sum | Against native tl.sum |
| --- | ---: | ---: |
| Ordered Triton | 39/39 | 16/39 |
| Ordered Gluon | 39/39 | 39/39 |

The previously borderline Gluon FP32 `[16,262144]` case measured 5.197125 us
versus native Triton's 4.711250 us: +10.31%, +0.485875 us. It is inside the
0.5 us allowance by 0.014125 us and remains close to that boundary. Ordinary
Triton retains its previously documented misses against native `tl.sum`.

With identical frozen settings, the largest per-case latency increase from the
preceding run was 1.84% for ordinary Triton and 1.24% for Gluon. No ordered case
exceeded the same regression allowance against its previous latency. Both native
and ordered timings are retained in `comparison.json`. Timed kernel arithmetic
is unchanged: the near-boundary gate crossing is measurement variation, not
evidence of a cleanup-induced optimization.

Upper layers can keep ordinary Triton or choose a separately qualified Gluon
kernel. The library does not select a backend, promise these speedups for every
input, or substitute a different arithmetic tree for performance.

## Evidence and reproduction

Download the [A40 test and H100 benchmark bundle](https://github.com/user-attachments/files/34221795/ordered-reduction-cleanup-20261101.zip).
Its ZIP SHA256 is
`b5b62c5e5d02e94ff1a84e0ed7b27be057a7f1f1283718f4dd0ea5847b1d4245`.
Measurements, figures and archives are PR attachments, outside the source diff.

The bundle contains:

- `pytest-a40/`: complete pytest log, JUnit XML, exit codes and environment.
- `benchmark-h100/results.json` and `benchmark-h100/cases/*.json`: all 159
  measurements, raw samples, round medians, actual execution orders, correctness
  gates, hardware/software identity and provider settings.
- `benchmark-h100/summary.json` and `benchmark-h100/comparison.json`: acceptance
  calculations and per-case changes from the preceding published run.
- `source/`: the exact measured patch, new test files, full source manifest and
  original bounded runners. Raw provenance retains the pre-commit source identity.
- `committed-source-mapping.json`: verifies 1426 of 1428 files byte for byte;
  the two remaining test files have formatting-only changes with identical ASTs.
- `reproduction/`: drivers and the same frozen 39-case plan, pinned to the
  committed cleanup revision rather than the moving branch head.
- `latency.csv`, `figures/`, `plot_results.py` and `SHA256SUMS`: chart data,
  PNG/SVG figures, plotting script and attachment checksums.

On separate fresh hosts with Git, Python 3.12 and uv installed, run from the
extracted bundle directory:

```bash
# A40 or another qualified NVIDIA test GPU
bash reproduce.sh pytest

# H100 SXM 80GB, 132 SMs
bash reproduce.sh benchmark
```

Each command fetches the pinned commit and verifies its manifest before running.
It refuses to overwrite `/root/ordered-regression`. Setup has a 900-second
timeout; pytest has 1200 seconds overall and 90 seconds per test; the benchmark
has 2400 seconds overall and 180 seconds per case. Results are packaged as
`/root/ordered-regression-pytest.tar.gz` or `/root/ordered-regression-h100.tar.gz`.
The benchmark driver checks completion, sample counts and actual round orders.

The maintained benchmark CLI uses one explicit configuration per schedule. A
single invocation with its defaults does **not** reproduce the per-case settings
in this table; use the frozen plan and attachment driver for that comparison.
The earlier parameter-grid search and diagnostic candidates are not retained in
the source tree. New measurements can use the documented
[benchmark commands](../architecture/ordered-reduction.md#performance-comparison)
with explicit settings and keep their results separate from this recorded run.
