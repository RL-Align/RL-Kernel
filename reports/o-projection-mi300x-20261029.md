# Item 23 MI300X acceptance — 2026-10-29

## Revision and environment

Based on item-13 commit `f9c8b27d78b620e3a29c9125f108af631bd63699`
and current `main` `8588158650aff58c9c1f527674de94bad06c3d10`,
merged in `1becef6d` before the final two test runs.
PR target remains `dsv4-rocm`; item 13 is an explicit prerequisite.

* One MI300X/gfx942 device; no other member container was changed.
* Torch `2.11.0+gitd0c8b1f`, HIP `7.2.53211`, Triton `3.6.0`.
* aiter `0.1.16.post3`, vLLM `0.26.0+rocm723`.
* Native aiter RoPE shared objects, batched BF16 GEMM and item-13 BF16/FP8
  providers. No replacement arithmetic kernel was authored.
* Artifacts live under `/outputs/item23-20261029/`, including the tested
  source SHA256 manifest, JUnit, coverage, benchmark and provider fingerprints.

## Correctness

Final command (with the pinned mLite source enabled):

```bash
RLK_MEGATRON_DENSE_SOURCE=/workspace/upstream/megatron_dense.py \
python3 -m pytest tests/contracts/operators tests/models/deepseek_v4 -q \
  --disable-warnings --basetemp=/outputs/item23-20261029/repeat-temp \
  --junitxml=/outputs/item23-20261029/repeat-tests.xml
```

**316 passed, zero skipped, 14 warnings**, in two fresh processes:
24.31 seconds with branch coverage and 12.81 seconds without coverage.
This includes **110 new item-23 tests** plus 206 prerequisite/regression tests.
Ruff formatting and checks pass on all new Python files.

Coverage is measured only over the six new production Python modules, not
native library instructions. Detailed statement/branch counts are retained
in `coverage.json`. Missing paths include dependency-import failure and
defensive impossible canonical-group completion guards.

Acceptance covers:

* Each intermediate: suffix inverse RoPE, unrotated prefix, group outputs,
  concatenation and actual FP32 `wo_b` partials against independent FP64 math.
* BF16 dX, dA and dB against an independent manually staged VJP oracle;
  inverse-rotation finite differences/adjoint identity.
* Exact bytes for one-row and mixed microbatches, permutations, reverse
  backward order, padded rows, train/no-grad/inference.
* Both independent FP8 stages and their combination; OCP and normalized FNUZ;
  real native activation quantization plus dequantized FP64 matmul oracle.
* FP8 BF16-master surrogate VJPs, canonical gradient byte equality across
  microbatches, and rejection of missing masters for differentiation.
* M=0/1/3/8/31/32/33/65, width/rank tails, full/partial RoPE suffixes,
  and synthetic H64/D512/rank1024/hidden4096.
* Local TP1/2/4 rank-ordered sums and gradients; missing/duplicate ranks,
  wrong versions/rows/angles/layer/provider rejected.
* Stale source/master/scale/metadata mutation, malformed checkpoint and
  layout, duplicate token identities and nondefault CUDA stream use.

The first exploratory tests incorrectly demanded bitwise equality between
FP32 native RoPE and a FP64 oracle at BF16 rounding midpoints. Numerical
acceptance now uses BF16-appropriate tolerances; **native-vs-native BI checks
remain exact bytes**. This is not an arithmetic fallback or a relaxed BI
criterion.

## Timing

40 repetitions after warmup, separate HIP events and synchronized wall time.
These are **eager provider-call timings**, including launch/host gaps, not
isolated kernel-only latency or a comparison against a baseline engine.
Row binding and weight preparation are outside the timed full operator.
The full operator includes validation, masking, layout and Python overhead.

H64 / D512 / rank1024 / hidden4096, BF16:

| M | Inverse RoPE event µs | Grouped A event µs | B event µs | Full event µs | Full wall µs |
|---:|---:|---:|---:|---:|---:|
| 1 | 124.65 | 48.90 | 115.29 | 371.27 | 371.44 |
| 32 | 144.48 | 52.94 | 123.93 | 404.56 | 404.96 |
| 256 | 149.90 | 63.67 | 189.56 | 494.25 | 495.48 |

Smaller H16/D128/rank128/hidden128 full wall times were 271.56, 289.72,
289.63 µs for the same M values. FP8 throughput was not benchmarked.
No speedup claim is inferred from these single-provider measurements.

## Explicit non-claims

Graph is rejected rather than silently falling back. TP tests are local
partial assembly, not real RCCL multi-rank tests. This report does not
establish full-model vLLM/Megatron/verl/vime execution, CP/DCP, equality across
parallel degrees, native FP8 quantization derivatives, or a model checkpoint
equivalence test. The large geometry is synthetic, not a loaded model.
