# H3 sigma schedule: RTX 4090D evidence

These are historical operator measurements from October 16–17, 2026,
rechecked against candidate commit `374488b` on October 21. No new GPU run
was performed during the source-binding check.

## Correctness and source identity

The final NVIDIA closing run (`run-20261017-002253-9749`) produced
**34 passed, 1 skipped, 0 failures/errors**. The skipped device-guard test
requires two GPUs; this host had one. [CUDA JUnit](cuda-4090d.xml) preserves
all outcomes; its hostname attribute was removed and the XML reserialized.

[Source provenance](source-provenance.json) records SHA256 hashes for the
tested archive, selected source files, and included results. The reference,
CUDA backend, fixture, CPU/GPU/dispatch tests, and benchmark in the final
closing snapshot are byte-identical to those files in `374488b`.

The performance run (`run-20261016-235221-8055`) predates the final HIP
rejection guard. Its reference and benchmark are byte-identical to the
candidate; CUDA kernels and the scalar helper have identical ASTs. The
forward method differs only in the error text for a rejected non-CUDA
device. The final NVIDIA correctness run tested the updated CUDA file.

Hardware/software: one NVIDIA GeForce RTX 4090D, x86_64 Linux, driver
580.76.05, Python 3.12.3, torch 2.5.1+cu121, CUDA runtime 12.1, Triton 3.1.0.

## Comparable latency

[Raw benchmark JSON](benchmark-4090d.json) is copied without modification.
It contains 100 samples per case and path (2,400 samples total), after
five warmup calls. Path A builds on CPU and leaves outputs there; path B
builds on CPU and copies both outputs to CUDA; path C constructs on CUDA.
Only B/C compares equal output placement. Timing is per-call host wall
latency with synchronization before and after each GPU sample; correctness
checking and compilation happen before timed samples.

| Workload | B median (ms) | C median (ms) | B/C |
| --- | ---: | ---: | ---: |
| N=25, shift=3 | 0.04158 | 0.11552 | 0.360 |
| N=25, shift=12 | 0.04188 | 0.13293 | 0.315 |
| N=50, shift=3 | 0.04523 | 0.11281 | 0.401 |
| N=50, shift=12 | 0.04495 | 0.11351 | 0.396 |
| N=1000, shift=3 | 0.04609 | 0.11484 | 0.401 |
| N=1000, shift=12 | 0.04672 | 0.11597 | 0.403 |
| N=1000000, shift=3 | 7.91859 | 1.00316 | 7.894 |
| N=1000000, shift=12 | 48.37110 | 0.93057 | 51.980 |

C is approximately 2.5–3.2 times slower at the typical sizes. N=1000000
is a deduplication stress case: the shift=12 gain exceeds the benchmark's
dispersion check; shift=3 does not and is not claimed as a reliable gain.
These are single-operator results, with no whole-model speedup claim.

## Reproduction

From a checkout with the declared project dependencies installed, run:

```bash
python -m pytest tests/reference/minimax_h3 tests/runtime/test_h3_sigma_schedule_dispatch.py -q
python -m pytest tests/backends/cuda/test_h3_sigma_schedule_cuda.py -v
python benchmarks/models/benchmark_h3_sigma_schedule.py --samples 100 --warmup 5 --json /tmp/h3-sigma.json
```

The second command and benchmark paths B/C require NVIDIA CUDA hardware
and Triton. The CPU fixture needs neither Diffusers nor network access.

## Evidence limits

ROCm is unsupported by this device backend, which uses inline NVIDIA PTX.
Other GPUs, ARM-hosted NVIDIA systems, multi-GPU behavior, and full-model
execution are unvalidated. Invalid-operation NaN payloads follow the pinned
x86 CPU convention; an ARM live CPU oracle may differ for extreme shifts.

The historical broad dispatch suite also recorded two attention failures
(`test_registered_backend_resolves_with_provenance` and
`test_implementation_kind_policy_filters_without_marking_fallback`). The
recorded baseline/candidate A/B runs contained the same failures. The H3
CUDA result above is not an all-green historical pipeline claim.

The existing Actions workflow runs on `main` and `test`; a PR targeting
`test-h3` has no automatic coverage from it. CPU checks for this submission
are reported separately from the historical GPU evidence.
