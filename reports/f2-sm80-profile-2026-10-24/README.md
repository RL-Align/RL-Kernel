# F2 A100 timing decomposition

Profiling completed for 24 combinations: T=1/128/2048, local/global, CUDA/Triton,
forward/backward. Uses the unchanged standard native extension from the preceding
integration experiment on A100 80GB PCIe, CUDA 13.0, PyTorch 2.14.1, Triton 3.8.0.
No operator implementation, numerical contract, bounds checks or fast-math settings
were changed in this experiment.

## Method

`f2_profile.py` imports `measure` from the preceding experiment's `probe.py`.
Run both scripts from the same directory, with the repository on PYTHONPATH:

```bash
PYTHONPATH=. python f2_profile.py --out profile-results-v2
```

The remote run used `/workspace/rlk-f2-integration-e3gNYR1a` and the existing
`/workspace/rlk-f2-sm80-DpNQapig/test-venv/bin/python` interpreter.

Unprofiled synchronized wall time uses seven samples of ten calls after warmup.
A separate PyTorch CPU/CUDA profiler pass records 20 calls. GPU activity durations
are read from Chrome traces, and each case asserts exactly 20 fused-kernel events.
CUDA names match `f2_forward`/`f2_backward`; Triton names match `_kernel` and
`_backward_kernel` exactly, excluding PyTorch validation kernels.

`profile-results-v2/` contains the raw 24 traces and results JSON; `profile-v2.log`
contains the execution log. The initial script used an overly broad Triton name
substring filter; its assertion caught validation-kernel contamination. The corrected
script completed all cases; only the corrected run is used below.

## Results

Representative local-layer values in microseconds:

| Tokens | Backend | Phase | Wall median | Fused GPU kernel mean under profiler | All GPU kernels mean under profiler |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | CUDA | Forward | 193.3 | 3.0 | 13.7 |
| 1 | Triton | Forward | 194.8 | 2.9 | 8.0 |
| 1 | CUDA | Backward | 74.9 | 3.1 | 8.4 |
| 1 | Triton | Backward | 49.3 | 2.9 | 2.9 |
| 2048 | CUDA | Forward | 225.8 | 117.3 | 135.8 |
| 2048 | Triton | Forward | 217.3 | 114.8 | 124.6 |
| 2048 | CUDA | Backward | 184.8 | 133.3 | 141.8 |
| 2048 | Triton | Backward | 122.7 | 122.2 | 122.2 |

Python local validation alone measured about 74–96 us across these shapes.
CUDA local forward validates positions twice: Python checks out-of-range flags,
then the native entry point checks min/max with host scalar reads. Native local
backward also checks min/max; internal Triton backward assumes validated inputs.
Thus the paths do not have identical validation overhead.

For T=1, the fused GPU work is short relative to complete call time; validation,
launch, allocation and host scheduling are the main optimization candidates. For
T=2048, fused GPU computation is substantial as well. The traces support reducing
duplicate synchronous checks first for small inputs, then tuning kernel geometry
for larger inputs while retaining the fixed reduction and rounding rules.

Wall times and GPU activity durations come from separate runs. They must not be
subtracted to claim an exact CPU cost or converted into precise overhead percentages.
Profiler instrumentation can perturb GPU timings; clocks were not pinned. This
is diagnostic profiling, not a final kernel-throughput benchmark or speedup claim.
Backward is measured with saved rstd, excluding forward/autograd-engine time.

## Next implementation work

Preserve fail-closed bounds checking for every public native/Python entry point,
but avoid checking the same positions twice in one CUDA forward. Any shared or
cached validation path must handle mutated positions/tables safely. After changing
the validation path, rerun invalid-input tests, the full 73-test targeted suite,
and fixed-data bitwise comparisons, then repeat the same timing protocol.
