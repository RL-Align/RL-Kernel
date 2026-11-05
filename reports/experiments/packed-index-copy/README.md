# Packed index copy: local SM75 optimization experiment

The optimized CUDA wrapper was faster than the frozen original CUDA wrapper in
all 48 measured cases on this RTX 2060 run. This is local operator evidence from a
focused extension containing the unchanged shared declarations/bindings and the
operator source. It is not a normal full shared-extension runtime result or an
optimized H200 result.

| Mode | Original CUDA / optimized CUDA, min / median / max | Eager / optimized CUDA range |
|---|---:|---:|
| Forward | 1.445 / 2.020 / 2.635x | 2.469–3.376x |
| Backward | 1.449 / 1.865 / 2.730x | 1.005–3.309x |
| Forward + backward | 1.343 / 1.936 / 2.325x | 1.341–3.519x |

Each mode covers 16 workloads. All 16 optimized medians were below eager in this
run, but BF16 `1x257x5376` backward was only 1.004879x: approximately break-even.
The table does not establish statistical significance. Ratios summarize the 16
per-workload medians, not pooled timing samples.

Three source stages isolate the changes:

- Original: clean source commit `32fce8bb9207b7297f9dbb4ab5db0d8b0dccab45`, frozen
  before edits; sort-based validation and scalar copy.
- Bitset only: an uncommitted intermediate based on `32fce8bb`, CUDA source SHA256
  `a4b0fd81c632013e7d68971e9213e948fcf0eaee91649e6e0928c6964108d8af`.
- Optimized: clean commit `2f67bc3e5409e7d366f8ec6e2a10bac1117cedf6`, integer bitset
  validation and vector row copies. All six measured implementation/benchmark
  file hashes matched that commit before timing; CUDA source SHA256
  `82129da88e34242f5d8db2ca8996dc4b6a78969776eef89b117724e302a7c263`.

The [comparison JSON](sm75-comparison.json) contains complete source/binding and
binary hashes, build flags, environment, and all three stages' 48 per-case
median/standard-deviation and allocation summaries for both eager and CUDA.
The bitset intermediate is identified by its snapshot hash rather than a commit.

Measurements ran on NVIDIA GeForce RTX 2060 (SM75, 6 GiB), driver 610.88,
PyTorch 2.9.1+cu130 and Python 3.12.3. Exact nvcc/GCC versions are in the JSON.
Focused extensions used GCC/G++ 13, `-O3`, CUDA `-O3 -Xfatbin -compress-all`,
`TORCH_CUDA_ARCH_LIST=7.5`, and no `--use_fast_math`. Each stage used the same
25-warmup/100-call shared `PerformanceProfiler` method, seed 420, four shapes,
FP16/BF16/FP32/mixed profiles, and three modes.

Public wrappers are timed, including index checking and host synchronization,
allocation and autograd dispatch. Forward uses `no_grad`; backward uses a
prebuilt retained graph; combined execution creates fresh leaves and a graph.
All three source gradients are requested without `.grad` accumulation.
Identical resident inputs use reversed shared modality indices and zero holes.
All stages passed output, three-gradient, and inference byte checks before
timing. Those checks are reused for the three modes of each dtype/shape.
Input generation and correctness checks are outside timing.

Separate stage runs permit scheduling and clock drift; eager measurements also
varied. Individual timing samples were not retained: the JSON reports medians,
sample standard deviations, and memory summaries, not 100 raw samples. Case
rows round milliseconds/ratios to nine decimal places and memory to six places
in MiB or nine in GiB.
Extra peak MiB excludes inputs and any retained graph; total allocated GiB
includes live inputs/graphs and is not reserved memory or device usage.

Diagnostic profiler measurements explain why both changes matter. BF16
`1x8193x5376` backward used about 68.9 us for sorting and 1543.5 us for the old
gather kernel. Bitset validation reduced its validation kernel to about 2.1 us
and the backward path to three kernel launches; the scalar gather remained
about 1551.1 us. With vector copies the gather was about 565.2 us, with the same
three launches. These profiled timings include instrumentation overhead and
are separate from the benchmark medians. Host synchronization waits can include
previously queued kernels; they are not pure validation cost. The first small
bitset profile lacked CUDA events and its device metrics are unavailable.

Re-run the optimized public benchmark from the repository after building the
normal CUDA extension, following the operator page's build instructions:

```bash
python -m benchmarks.operators.packing.benchmark_packed_index_copy \
  --device cuda:0 --dtypes bf16 fp16 fp32 mixed \
  --shapes 1x257x5376 1x1025x5376 1x8193x5376 2x257x5376 \
  --warmup 25 --repeat 100 --seed 420
```

That command uses the installed shared extension; these recorded experiments
used isolated frozen source/binary snapshots with extracted bindings. The
reported pre-optimization H200 results remain separate external evidence and do
not validate the optimized commit. No full H3 performance claim is made.
