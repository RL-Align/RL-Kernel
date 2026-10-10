# Issue #50 persistent auto-tuning: B200 evidence

Auto-tuning is opt-in; with `RL_KERNEL_AUTOTUNE` unset or `0`, operators launch
their existing fixed configurations. All runs used one NVIDIA B200 (driver
580.126.20, CUDA 13.0) with no other compute process at start, on top of upstream
`main` at `16b81175`. [source-sha256.json](source-sha256.json) fingerprints every
Python file this change adds or modifies.

Two environments were used:

| Environment | Python | PyTorch | Triton | Native `_C` | Results |
| --- | --- | --- | --- | --- | --- |
| A | 3.13.13 | 2.14.0+cu130 | 3.8.0 | not built (PyTorch 2.14 headers require C++20; the extension builds with C++17) | this directory |
| B | 3.12.14 | 2.13.0+cu130 | 3.7.1 | built in place from this tree | [native/](native/) |

Environment B ran the exact fingerprinted source. Environment A ran an earlier
snapshot that differs only in `tests/ops/gemm/test_det_gemm.py` and one import line
in `tests/benchmarks/test_autotuning_benchmark.py`; GEMM results therefore come
from environment B only, and the benchmark row was rerun with the final file.

## Numerical and functional results

| Suite | Env | Passed | Skipped | Failed | Evidence |
| --- | --- | ---: | ---: | ---: | --- |
| GEMM, `RL_KERNEL_MODE=auto RL_KERNEL_AUTOTUNE=0` | B | 73 | 9 | 0 | [xml](native/gemm-off.xml), [log](native/gemm-off.txt) |
| GEMM, `RL_KERNEL_MODE=auto RL_KERNEL_AUTOTUNE=1` | B | 73 | 9 | 0 | [xml](native/gemm-on.xml), [log](native/gemm-on.txt) |
| HAL and registry | A | 31 | 0 | 0 | [xml](hal.xml), [log](hal.txt) |
| Operator, cache and graph regressions | A | 60 | 2 | 0 | [xml](gpu.xml), [log](gpu.txt) |
| | B | 60 | 2 | 0 | [xml](native/gpu.xml), [log](native/gpu.txt) |
| Candidate qualification | A | 402 | 27 | 0 | [xml](qualification.xml), [log](qualification.txt) |
| | B | 402 | 27 | 0 | [xml](native/qualification.xml), [log](native/qualification.txt) |
| Benchmark functionality | A | 18 | 0 | 0 | [xml](benchmark-functional.xml), [log](benchmark-functional.txt) |

Commands, from the repository root:

```bash
RL_KERNEL_MODE=auto RL_KERNEL_AUTOTUNE={0|1} python -m pytest -q -rs tests/ops/gemm/test_det_gemm.py
python -m pytest -q -rs tests/platforms/test_persistent_autotuner.py tests/runtime/test_kernel_registry.py
python -m pytest -q -rs tests/ops/test_autotuning_gpu.py
python -m pytest -q -rs tests/ops/test_autotuning_candidates.py
python -m pytest -q -rs tests/benchmarks/test_autotuning_benchmark.py
```

Candidate qualification launches every production candidate directly and
compares output, auxiliary/workspace and gradient bytes with the fixed launch;
there is no selector or fallback in that path. All candidates matched under both
Triton versions. The [coverage manifest](qualified-configurations.json), extracted
from the JUnit `qualified_launches` properties, lists 24 CUDA-reachable kernels and
101 kernel/configuration pairs.

Skips: the 9 GEMM skips require ROCm gfx942; the 27 candidate and 2 regression
skips are dtypes the existing operators do not support (FP16 GEMM, FP32
attention), whose original errors are checked instead.

The GEMM test previously caught every `ImportError` as "Triton is unavailable";
it imported a reduce kernel under a name that no longer exists, so on `main` its
Triton cases skip. With the same environment and `_C`, `main` reports 22 passed
and 49 skipped for this file.

### Two-GPU tests

The five `tests/ops` cases that need two devices were run in environment B on a
two-GPU allocation, for this change and for `main`, with tuning off and on:
cross-device rejection in SwiGLU (Triton and CUDA), the deterministic TP
linear-logprob check, and device-context restoration in reproducible norm.
Every run, on both trees, reports 4 passed and 1 skipped; the skipped case is the
TP=4 parameterisation, which needs four GPUs. See
[native/two-gpu-tune0.xml](native/two-gpu-tune0.xml) and
[native/two-gpu-tune1.xml](native/two-gpu-tune1.xml). On a four-GPU allocation the
TP linear-logprob test passes for both TP=2 and TP=4, on both trees, with tuning
off and on ([tune0](native/four-gpu-tune0.xml), [tune1](native/four-gpu-tune1.xml)).

### Existing operator suite against `main`

`tests/ops` (excluding the two auto-tuning files) was run in environment B with
tuning off and on, for this change and for `main` at `16b81175` with the same
`_C` build; see [native/ops-regression-vs-main.json](native/ops-regression-vs-main.json).

| | This change | `main` |
| --- | --- | --- |
| Tuning off | 1577 passed, 377 skipped, 2 failed | 1527 passed, 417 skipped, 1 failed |
| Tuning on | 1578 passed, 377 skipped, 1 failed | 1526 passed, 417 skipped, 2 failed |

No test that passes on `main` fails with this change apart from the flaky case
below; 40 GEMM tests that skip on `main` pass. The remaining failures also occur
on `main`:

- `test_rms_norm::test_registry_dispatches_rms_norm` fails in every run.
- `test_swiglu::test_autograd_wrapper_contiguity_and_gradient_routing`, a CPU test
  of the Ascend SwiGLU wrapper with unseeded inputs, fails intermittently with a
  one-element difference of about 7e-9: here in one parameter case with tuning off,
  on `main` in another with tuning on. In 20 repeated runs it failed 6 times on
  `main` and 4 times with this change. It does not execute Triton code.

## Performance

Environment A, on a single idle B200:

```bash
python -m benchmarks.tuning.benchmark_autotuning \
  --operator all --dtype bfloat16 --preset all --phase all --trials 5 \
  --output issue50-performance.json
```

All 5 independent process groups completed for the 52 supported
operator/preset/phase cases (8 phase combinations are explicitly unsupported).
Raw per-group data are in [performance.json.gz](performance.json.gz) (`zcat` to
read), five-group medians and ranges in [performance.csv](performance.csv), and
the counts below in [performance-status.json](performance-status.json). A benefit
is confirmed only when the fixed and tuned ranges of process medians do not overlap.

| Metric | Confirmed faster | Tuned slower (ranges do not overlap) | Overlapping |
| --- | ---: | ---: | ---: |
| Eager wall time | 0 | 24 | 28 |
| CUDA graph device time | 19 | 5 | 28 |

- **Graph replay.** Gains are concentrated in small elementwise cases: SiLU and
  SwiGLU small forward/backward 20–31%; RoPE, embedding, ratio/KL and GRPO small
  cases 4–9%; GEMM and attention under 1%. Break-even is about 22k to 600k calls.
  Five cases are slightly slower (RMSNorm small backward and forward-backward,
  SwiGLU throughput forward-backward, ratio/KL small backward, GRPO throughput forward).
- **Eager execution.** No case is faster. Tuned forward calls are typically
  7–15 µs slower (attention forward about 30 µs); the median difference across
  cases is +20.5 µs. The selection and dispatch cost per call outweighs the
  kernel-level gain at these sizes.
- **Cache behaviour.** In all groups, cold and precompiled selections report
  `tuned`, restarted processes report `disk`, and hot selector calls total zero.

Auto-tuning is therefore useful for graph-captured execution and remains opt-in.
An earlier run of the same experiment on the pre-refactor layout reached the same
conclusion (0 eager benefits; 11 graph benefits, 15 graph slowdowns).

## Not covered

- ROCm hardware, including the ROCm fused GEMM leaf.
- Performance in environment B or with the native extension; the performance
  experiment compares tuned and fixed Triton launches only.
- Read-only cache directories and failures during atomic cache replacement on real
  hardware; physical multi-device isolation and device-visibility remapping.
