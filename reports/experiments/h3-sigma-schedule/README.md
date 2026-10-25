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
CUDA backend, fixture and CPU reference test in the final closing snapshot are
byte-identical to those files in `374488b` and to the current branch.

The dispatch test, the CUDA test and the benchmark docstring were revised
after the 4090D runs for the review-driven default backend change (the legacy
`sigma_schedule_h3` candidate list is now reference-only on CUDA; the Triton
backend is opt-in through the semantic `requested_backend` or direct
construction). Those edits touch candidate order, test selection logic and
prose only: every kernel bit assertion is unchanged, and the CUDA suite gained
an assertion that the default reference path matches the Triton path bit for
bit. No GPU run backed the revised tests at the time; the SM90 (H800) run
below now does. `source-provenance.json` has been refreshed accordingly: its
`cuda_validation` block keeps the historical `374488b` hashes of the
4090D-tested snapshot, and the new `sm90_validation` block records the hashes
at the SM90-tested commit (`40070d3`), the post-run changes, and the SHA256
of every included SM90 artifact.

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
builds on CPU and copies both outputs to CUDA (the registry default path);
path C constructs on CUDA (the Triton opt-in backend).
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

## SM90 (H800) evidence

Collected on 2026-10-26 on a rented SM90 host (one **NVIDIA H800 PCIe**, per
`nvidia-smi`), with the same three pytest invocations and the same benchmark
command as above, at tested commit
`40070d37495d193e2d8764837321d3a68dad9483` (`40070d3`). The JUnit XML for each
run, the benchmark JSON, the environment snapshot and the full collection
report (which embeds the raw console logs verbatim) are in
[sm90-h800/](sm90-h800/). The standalone console `.log` files are excluded by
the repository-wide `*.log` ignore rule — as in the 4090D directory; their
hashes are recorded in `source-provenance.json` (`original_server_sha256`).
GPU UUID, container hostname and `/root/…` paths
were redacted: the JUnit `hostname` attribute was removed (as for the 4090D
XML, which keeps the historical `/root/…` skip-message paths — the SM90 files
additionally replace them with `REPO_ROOT` / `EVIDENCE_DIR` /
`MINICONDA_PYTHON` tokens). `benchmark-h800.json` and `env.txt` required no
redaction; a final newline was appended to the three XMLs and the JSON to
satisfy the repository's end-of-file hook. `source-provenance.json` records
both the committed (redacted) and the original server-side SHA256 of every
file.

Hardware/software: NVIDIA H800 PCIe (compute capability 9.0), driver
580.82.07, x86_64 Linux 5.15.0-78-generic, 2× Intel Xeon Platinum 8458P,
Python 3.12.3, torch 2.5.1+cu124, CUDA runtime 12.4, Triton 3.1.0 — the same
Triton as the 4090D run; torch differs only in the CUDA build (cu124 vs
cu121), the driver is 580.82.07 vs 580.76.05.

| Run | Scope (`-v -rs`) | Collected | Passed | Skipped | Failed |
| --- | --- | ---: | ---: | ---: | ---: |
| A | reference + dispatch + CUDA suites | 69 | 67 | 2 | 0 |
| B | CUDA + dispatch subset | 48 | 47 | 1 | 0 |
| C | the two registry tests introduced after the 4090D runs | 2 | 2 | 0 | 0 |

Skips, with the reasons printed by `-rs`: the optional Diffusers parity test
(`diffusers` not installed on the host; the pinned fixture test passed) and
the multi-GPU device-guard test (the host has one GPU) — the same test, for
the same reason, as on the 4090D. Run C was the first hardware execution of
`test_public_entry_defaults_to_reference_and_matches_triton_bits` and
`test_public_entry_opt_in_selects_triton_and_reports_provenance`; both
passed — on SM90 the legacy `get_op(device="cuda")` reference path and the
directly constructed Triton backend produce identical sigma/timestep bits.

Versus the 4090D record (34 passed / 1 skipped): one more CUDA test passed
because the default-backend change split the registry-identity test into
two; no test that passed on the 4090D failed here; the single skip is the
same test for the same reason.

Benchmark (100 samples, 5 warmups, [raw JSON](sm90-h800/benchmark-h800.json)):
the pre-timing bitwise gate (path C vs path A) passed for all 8 cases; Triton
compilation 1143.6 ms, reported once and excluded from timed samples.

| Workload | B median (ms) | C median (ms) | B/C |
| --- | ---: | ---: | ---: |
| N=25, shift=3 | 0.0551 | 0.0926 | 0.60x |
| N=25, shift=12 | 0.0554 | 0.0913 | 0.61x |
| N=50, shift=3 | 0.0556 | 0.0918 | 0.61x |
| N=50, shift=12 | 0.0554 | 0.0887 | 0.62x |
| N=1000, shift=3 | 0.0598 | 0.0907 | 0.66x |
| N=1000, shift=12 | 0.0609 | 0.0892 | 0.68x |
| N=1000000, shift=3 | 5.5054 | 1.3377 | 4.12x |
| N=1000000, shift=12 | 5.8109 | 1.3374 | 4.34x |

All eight ratios pass the benchmark's dispersion check. These latency numbers
are supplementary only and not comparable with the 4090D table: the H800 has
reduced FP64 throughput (the Triton kernel evaluates its intermediates in
fp64), and paths A/B run on a different host CPU (at N=1e6 the CPU path is
about 5.5 ms here versus about 48 ms on the 4090D host). On this host the
device build is 1.5–1.65x slower than CPU build plus copy at N=25/50/1000
and 4.1–4.3x faster at the N=1e6 dedup stress case — directionally the same
default-relevant conclusion as the 4090D, at a smaller margin.

Post-run changes: after the tested commit `40070d3`, commit `ace9c20`
("fix(h3): address CodeRabbit review on sigma schedule benchmark and docs")
modified only the benchmark's GPU-without-Triton and HIP/ROCm branches, its
docstring, and documentation. The H800 run executed the NVIDIA + Triton
branch, whose logic, timing order, sample counts and JSON keys that commit
leaves unchanged; the new branches were verified by CPU-only execution,
`_summarize` simulations and static review, and have not run on GPU or HIP
hardware.

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
