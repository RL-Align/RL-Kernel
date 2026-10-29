# PR #566 `h3_sigma_schedule` — SM90 CUDA bitwise evidence (NVIDIA H800 PCIe)

Collected 2026-10-26 01:27–01:34 CST (2026-10-25 17:27–17:34 UTC) on a rented
AutoDL container (`CONTAINER-NAME-REDACTED`). Evidence-only run:
no source, test or documentation file was modified; nothing was pushed or
fetched from GitHub. The GPU name below is copied from `nvidia-smi` as printed.

## Environment

| Item | Value |
| --- | --- |
| GPU | `NVIDIA H800 PCIe` (nvidia-smi; 1 device, 81559 MiB; UUID GPU-UUID-REDACTED) |
| Compute capability | `9.0` (`nvidia-smi --query-gpu=compute_cap`); `torch.cuda.get_device_capability()` = `(9, 0)` |
| NVIDIA driver | `580.82.07` (nvidia-smi header reports driver-side CUDA Version 13.0) |
| CUDA runtime seen by torch | `12.4` |
| torch | `2.5.1+cu124` (pre-installed in the image; not touched) |
| Triton | `3.1.0` (pre-installed; not touched; identical to the RTX 4090D run) |
| Python | `3.12.3` CPython, `MINICONDA_PYTHON` |
| pytest | `9.1.1` (the only package installed for this run) |
| Host | x86_64, Linux 5.15.0-78-generic; 2× Intel Xeon Platinum 8458P (176 logical CPUs); ~1 TB RAM |
| Repository HEAD | `40070d37495d193e2d8764837321d3a68dad9483` — "Merge remote-tracking branch 'fork/h3-sigma-schedule' into h3-sigma-schedule" |
| Working tree | clean (`git status --short` printed nothing) |
| Repo path on host | `REPO_ROOT` |

## Code transfer and version gates (step 2)

The code was transferred from the developer machine, not fetched from GitHub:
a `git clone --branch h3-sigma-schedule --single-branch` of the local
repository was tarred (`rlk-pr566-40070d3.tgz`, 66 MB, sha256
`3fef727cb277d412a1d2f31fbf56708c423ca5b2c0c4b632de14bbd20b8fd4c0`), copied
with scp, and the sha256 recomputed on the host matched. Extraction printed
harmless `LIBARCHIVE.xattr.com.apple.provenance` header warnings (macOS tar).
Because the tarball preserved the developer's uid, git refused with
"detected dubious ownership"; `chown -R root:root` on the extracted tree
cleared it (ownership only — no file content changed).

```
$ git log --oneline -8
40070d3 Merge remote-tracking branch 'fork/h3-sigma-schedule' into h3-sigma-schedule
eae0118 Merge branch 'test-h3' into h3-sigma-schedule
a66c999 refactor(h3): default sigma_schedule_h3 to the PyTorch reference on CUDA
d22d4b4 Merge pull request #576 from fusheng-ji/feat/h3-flash-attention-v1
ae8dff6 Merge branch 'test-h3' into feat/h3-flash-attention-v1
8542cf7 docs(h3): document the missing-Triton RuntimeError for h3-flash-attention-v1
4c2348f fix(h3): address CodeRabbit review on h3-flash-attention-v1
6f59021 Merge branch 'test-h3' into feat/h3-flash-attention-v1

$ git status --short
(empty)

$ git rev-parse HEAD
40070d37495d193e2d8764837321d3a68dad9483

$ git log --oneline | grep -c 'refactor(h3): default sigma_schedule_h3 to the PyTorch reference on CUDA'
1

$ grep -n sigma_schedule_h3 -A2 rl_engine/runtime/registry.py
431:            semantic_op="sigma_schedule_h3",           # semantic descriptor: reference
451:            semantic_op="sigma_schedule_h3",           # semantic descriptor: triton
739:                "sigma_schedule_h3": [OpBackend.PYTORCH_H3_SIGMA_SCHEDULE],   # cuda
763:                "sigma_schedule_h3": [OpBackend.PYTORCH_H3_SIGMA_SCHEDULE],   # rocm
807:                "sigma_schedule_h3": [OpBackend.PYTORCH_H3_SIGMA_SCHEDULE],   # musa
847:                "sigma_schedule_h3": [OpBackend.PYTORCH_H3_SIGMA_SCHEDULE],   # cpu
```

All gates satisfied: HEAD is `40070d3…`, the default-backend commit `a66c999`
is in the history, and every legacy platform list for `sigma_schedule_h3`
contains only `PYTORCH_H3_SIGMA_SCHEDULE`.

## Test files (step 3)

`git diff --stat origin/test-h3...HEAD -- tests` fails on this copy
(`fatal: bad revision` — the single-branch clone carries no `origin/test-h3`
ref). Fallbacks used:

```
$ git diff --stat 97a7ab9 HEAD -- tests          # 97a7ab9 = PR base before the test-h3 merge
 tests/backends/cuda/test_h3_flash_attention.py     | 129 +++++++   <- PR #576 via test-h3, NOT this PR
 tests/backends/cuda/test_h3_sigma_schedule_cuda.py | 392 +++++++++++++++++++++
 tests/reference/minimax_h3/__init__.py             |   2 +
 .../h3_sigma_schedule_upstream_fixture.json        | 201 +++++++++++
 .../reference/minimax_h3/make_upstream_fixture.py  | 154 ++++++++
 .../reference/minimax_h3/test_h3_sigma_schedule.py | 185 ++++++++++
 tests/runtime/test_h3_sigma_schedule_dispatch.py   | 190 ++++++++++

$ git ls-files | grep -i h3_sigma
benchmarks/models/benchmark_h3_sigma_schedule.py
tests/backends/cuda/test_h3_sigma_schedule_cuda.py
tests/reference/minimax_h3/h3_sigma_schedule_upstream_fixture.json
tests/reference/minimax_h3/test_h3_sigma_schedule.py
tests/runtime/test_h3_sigma_schedule_dispatch.py
```

Sigma-schedule test files run: `tests/reference/minimax_h3/test_h3_sigma_schedule.py`,
`tests/runtime/test_h3_sigma_schedule_dispatch.py`,
`tests/backends/cuda/test_h3_sigma_schedule_cuda.py`.
`test_h3_flash_attention.py` belongs to PR #576 and was not run.

All runs: `python -m pytest … -v -rs` from the repository root, same
interpreter, no `-x`. Logs and JUnit XML are in `EVIDENCE_DIR/`.

## Results

| Run | Command | Collected | Passed | Skipped | Failed | Errors | Wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A — all H3 sigma tests | `python -m pytest tests/reference/minimax_h3 tests/runtime/test_h3_sigma_schedule_dispatch.py tests/backends/cuda/test_h3_sigma_schedule_cuda.py -v -rs` | 69 | **67** | 2 | **0** | 0 | 4.86 s |
| B — CUDA/Triton/bitwise + dispatch subset | `python -m pytest tests/backends/cuda/test_h3_sigma_schedule_cuda.py tests/runtime/test_h3_sigma_schedule_dispatch.py -v -rs` | 48 | **47** | 1 | **0** | 0 | 2.89 s |
| C — the two new/renamed registry tests | `python -m pytest "tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_public_entry_defaults_to_reference_and_matches_triton_bits" "tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_public_entry_opt_in_selects_triton_and_reports_provenance" -v -rs` | 2 | **2** | 0 | **0** | 0 | 2.51 s |

Run A by file: `test_h3_sigma_schedule.py` 20 passed / 1 skipped;
`test_h3_sigma_schedule_dispatch.py` 12 passed; `test_h3_sigma_schedule_cuda.py`
35 passed / 1 skipped.

### Skips (both runs, same two)

| Test | Reason printed by `-rs` | Explanation |
| --- | --- | --- |
| `tests/reference/minimax_h3/test_h3_sigma_schedule.py:171` `test_matches_installed_diffusers_scheduler_when_available` | `could not import 'diffusers': No module named 'diffusers'` | Optional dependency not installed on this host (it was not installed to avoid touching the pinned torch). The pinned fixture test covers the upstream parity and passed. |
| `tests/backends/cuda/test_h3_sigma_schedule_cuda.py:378` `test_target_device_guard_with_mismatched_current_device` | `multi-GPU device-guard test needs 2 devices` | This host has exactly one GPU (`nvidia-smi` lists GPU 0 only; `--query-gpu=count` = 1). This is the same test skipped for the same reason on the RTX 4090D host. |

### Run C — singled out (first execution on any GPU)

Both tests were introduced/renamed in `a66c999` after the RTX 4090D runs, so
this is their first execution on hardware.

- `test_public_entry_defaults_to_reference_and_matches_triton_bits` — **PASSED**.
  `kernel_registry.get_op("sigma_schedule_h3", device="cuda")` returned
  `NativeH3SigmaScheduleOp`; its outputs built on CUDA (`N=25, shift=12.0`)
  compared bit-for-bit equal (`_int32_bits`) to `H3SigmaScheduleCudaOp()`'s
  `sigmas` and `timesteps`, and `num_model_evals` matched.
- `test_public_entry_opt_in_selects_triton_and_reports_provenance` — **PASSED**.
  Semantic `requested_backend="rlkernel.h3_sigma_schedule.triton"`
  instantiated `H3SigmaScheduleCudaOp` and provenance reported that backend.

No bitwise test failed, so step 5 (mismatch forensics) was not triggered.

## Benchmark (step 3d, optional)

```
python benchmarks/models/benchmark_h3_sigma_schedule.py --samples 100 --warmup 5 --json ~/pr566_evidence/benchmark-h800.json
```

Pre-timing correctness gate: path C (Triton, device build) bitwise equal to
path A (CPU reference) for all 8 cases — **passed**; the gate's total wall
time was 1143.6 ms (Triton kernel compilation included), reported once and
excluded from timed samples. Raw JSON:
`EVIDENCE_DIR/benchmark-h800.json` (52 KB, 100 samples per case and
path after 5 warmups).

| Workload | A median (ms) | B median (ms) | C median (ms) | B/C |
| --- | ---: | ---: | ---: | ---: |
| N=25, shift=3 | 0.0306 | 0.0551 | 0.0926 | 0.60x (separated) |
| N=25, shift=12 | 0.0305 | 0.0554 | 0.0913 | 0.61x (separated) |
| N=50, shift=3 | 0.0308 | 0.0556 | 0.0918 | 0.61x (separated) |
| N=50, shift=12 | 0.0310 | 0.0554 | 0.0887 | 0.62x (separated) |
| N=1000, shift=3 | 0.0347 | 0.0598 | 0.0907 | 0.66x (separated) |
| N=1000, shift=12 | 0.0346 | 0.0609 | 0.0892 | 0.68x (separated) |
| N=1000000, shift=3 | 5.4651 | 5.5054 | 1.3377 | 4.12x (separated) |
| N=1000000, shift=12 | 5.5773 | 5.8109 | 1.3374 | 4.34x (separated) |

A = reference, CPU build, CPU outputs; B = reference, CPU build, outputs
copied to CUDA (the registry default path); C = Triton opt-in, device build.
"separated" = the benchmark's own dispersion check passed.

Caveats (why these numbers are supplementary only): the H800 PCIe has reduced
FP64 throughput and the Triton kernel evaluates its intermediates in fp64;
paths A/B run on this host's Xeon 8458P, which is a different CPU from the
4090D host (at N=1e6 the CPU path here is ~5.5 ms versus ~48 ms there);
single-operator microbenchmark, one card. On this host the device build is
about 1.5–1.65x slower than CPU build plus copy at N=25/50/1000 and about
4.1–4.3x faster at the N=1e6 dedup stress case. The reviewer's question is
answered by the bitwise results above, not by this table.

## Observations (not failures)

- `WARNING … Core binary extension (_C) unavailable … Falling back to native code`
  was printed once during the benchmark. The repository was not `pip install -e .`-built
  (its declared `transformers==5.13.1`/`accelerate` dependencies were not installed to
  keep torch untouched); `h3_sigma_schedule` does not use `_C`. No test printed this
  warning.
- Only `pytest` was installed; torch 2.5.1+cu124 and Triton 3.1.0 are the image's.
- `git diff --stat origin/test-h3...HEAD` is unavailable on this copy (no such ref);
  `97a7ab9` and `git ls-files` fallbacks were used as instructed.

## Factual conclusion

On NVIDIA H800 PCIe (SM90, compute capability 9.0, driver 580.82.07, torch
2.5.1+cu124, CUDA 12.4, Triton 3.1.0) at HEAD `40070d37495d193e2d8764837321d3a68dad9483`:

- All 35 runnable tests in `tests/backends/cuda/test_h3_sigma_schedule_cuda.py`
  passed — pinned-fixture bitwise match, live-CPU-reference bitwise match on
  the three combos that test parametrizes (N=50/shift=3, N=50/shift=12 and
  N=1000/shift=12; its parametrization has no N=1000/shift=3 case),
  dense-grid dedup, grid invariants, subnormal
  (1e-45) / underflow (1e-46) / ±NaN / inf / 1e39 / 3.5e38 / 1e30 / 1e-10
  edge parity, float-N truncation, input rejections, repeated/interleaved
  stability, backend identity, default-reference-vs-Triton bit equality, and
  semantic Triton opt-in. 1 test skipped (needs 2 GPUs; host has 1). The
  three-combo list is that test's full parametrization — all-8-case coverage
  (N=25/50/1000/1e6 × shift 3/12) is provided separately by the benchmark's
  pre-timing correctness gate, not by this suite.
- All 12 dispatch tests passed. 20 reference tests passed, 1 skipped
  (`diffusers` not installed).
- 0 failures and 0 errors across runs A, B and C.
- Compared with the historical RTX 4090D CUDA run (34 passed / 1 skipped,
  torch 2.5.1+cu121, Triton 3.1.0, driver 580.76.05): one more test passed
  here because `a66c999` split the registry-identity test into two; the one
  skip is the same test for the same reason; no test that passed on the
  4090D failed here. Triton version is identical; torch differs only in the
  CUDA build (cu124 vs cu121); driver 580.82.07 vs 580.76.05.
- Benchmark correctness gate passed on all 8 cases; latency ratios differ
  from the 4090D (see caveats) and are reported as supplementary data only.

Evidence files: `EVIDENCE_DIR/{env.txt, run_a.log, run_a.xml, run_b.log,
run_b.xml, run_c.log, run_c.xml, bench.log, benchmark-h800.json}` on the host;
a copy was pulled to the developer machine.

## Raw outputs

<details><summary><code>env.txt</code></summary>

```text
Sun Oct 25 17:32:03 UTC 2026
NVIDIA H800 PCIe, 9.0, 580.82.07
Python 3.12.3
torch 2.5.1+cu124 cuda 12.4 triton 3.1.0 cc (9, 0) dev NVIDIA H800 PCIe
pytest 9.1.1
HEAD 40070d37495d193e2d8764837321d3a68dad9483
```

</details>

<details><summary><code>run_a.log</code></summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.7.0 -- MINICONDA_PYTHON
cachedir: .pytest_cache
rootdir: REPO_ROOT
configfile: pyproject.toml
plugins: anyio-4.6.2.post1
collecting ... collected 69 items

tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_matches_pinned_upstream_set_timesteps_bitwise PASSED [  1%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_grid_invariants[2-12.0] PASSED [  2%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_grid_invariants[2-3.0] PASSED [  4%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_grid_invariants[8-12.0] PASSED [  5%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_grid_invariants[8-3.0] PASSED [  7%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_grid_invariants[25-12.0] PASSED [  8%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_grid_invariants[25-3.0] PASSED [ 10%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_grid_invariants[50-12.0] PASSED [ 11%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_grid_invariants[50-3.0] PASSED [ 13%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_grid_invariants[1000-12.0] PASSED [ 14%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_grid_invariants[1000-3.0] PASSED [ 15%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_unique_consecutive_collapses_shift_created_collisions PASSED [ 17%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_video_and_audio_shifts_give_distinct_grids PASSED [ 18%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_rejects_num_inference_steps_below_two[1] PASSED [ 20%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_rejects_num_inference_steps_below_two[0] PASSED [ 21%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_rejects_num_inference_steps_below_two[-4] PASSED [ 23%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_rejects_non_positive_shift[0.0] PASSED [ 24%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_rejects_non_positive_shift[-1.5] PASSED [ 26%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_device_argument_preserves_bytes_and_dtype PASSED [ 27%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_builds_on_cpu_under_non_cpu_default_device PASSED [ 28%]
tests/reference/minimax_h3/test_h3_sigma_schedule.py::test_matches_installed_diffusers_scheduler_when_available SKIPPED [ 30%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_get_op_cpu_selects_reference_and_builds_pinned_grid PASSED [ 31%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_get_op_cuda_defaults_to_reference_regardless_of_triton PASSED [ 33%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_cuda_backend_construction_fails_loudly_without_triton PASSED [ 34%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_shift_scalars_pinned_to_cpu_under_meta_default PASSED [ 36%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_shift_scalars_keep_fp64_subtract_then_round_semantics PASSED [ 37%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_cuda_backend_construction_rejected_on_hip_build PASSED [ 39%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_get_op_on_hip_build_serves_reference PASSED [ 40%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_triton_backend_is_not_a_legacy_candidate_on_any_platform PASSED [ 42%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_semantic_descriptors_declared_truthfully PASSED [ 43%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_semantic_reference_path_resolves_instantiates_and_reports_provenance PASSED [ 44%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_semantic_triton_backend_fails_explicitly_when_unavailable PASSED [ 46%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_semantic_unknown_backend_is_rejected PASSED [ 47%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_matches_pinned_cpu_fixture_bitwise PASSED [ 49%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_matches_live_cpu_reference_bitwise[50-12.0] PASSED [ 50%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_matches_live_cpu_reference_bitwise[50-3.0] PASSED [ 52%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_matches_live_cpu_reference_bitwise[1000-12.0] PASSED [ 53%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_dense_grid_dedup_matches_cpu_reference PASSED [ 55%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_reference_with_cuda_device_preserves_fixture_bytes PASSED [ 56%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_grid_invariants_on_device[2-3.0] PASSED [ 57%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_grid_invariants_on_device[2-12.0] PASSED [ 59%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_grid_invariants_on_device[4-3.0] PASSED [ 60%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_grid_invariants_on_device[4-12.0] PASSED [ 62%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_grid_invariants_on_device[25-3.0] PASSED [ 63%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_grid_invariants_on_device[25-12.0] PASSED [ 65%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_num_inference_steps_below_two[1] PASSED [ 66%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_num_inference_steps_below_two[0] PASSED [ 68%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_num_inference_steps_below_two[-4] PASSED [ 69%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_subnormal_shift_1e_45_matches_reference_upstream_parity PASSED [ 71%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_underflow_shift_1e_46_matches_reference_upstream_parity PASSED [ 72%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_positive_nan_shift_propagates_positive_quiet_nan PASSED [ 73%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_negative_nan_shift_propagates_negative_quiet_nan PASSED [ 75%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_edge_shifts_match_reference_upstream_parity[nan] PASSED [ 76%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_edge_shifts_match_reference_upstream_parity[inf] PASSED [ 78%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_edge_shifts_match_reference_upstream_parity[1e-10] PASSED [ 79%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_edge_shifts_match_reference_upstream_parity[1e+39] PASSED [ 81%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_edge_shifts_match_reference_upstream_parity[3.5e+38] PASSED [ 82%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_edge_shifts_match_reference_upstream_parity[1e+30] PASSED [ 84%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_float_num_steps_truncate_like_reference[2.9-2] PASSED [ 85%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_float_num_steps_truncate_like_reference[3.1-3] PASSED [ 86%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_non_positive_shift[0.0] PASSED [ 88%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_non_positive_shift[-1.5] PASSED [ 89%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_non_positive_shift[-inf] PASSED [ 91%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_cpu_device_instead_of_silent_fallback PASSED [ 92%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_repeated_and_interleaved_calls_are_stable PASSED [ 94%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_backend_identity_is_device_path_not_cpu_fallback PASSED [ 95%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_public_entry_defaults_to_reference_and_matches_triton_bits PASSED [ 97%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_public_entry_opt_in_selects_triton_and_reports_provenance PASSED [ 98%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_target_device_guard_with_mismatched_current_device SKIPPED [100%]

-------------- generated xml file: EVIDENCE_DIR/run_a.xml --------------
=========================== short test summary info ============================
SKIPPED [1] tests/reference/minimax_h3/test_h3_sigma_schedule.py:171: could not import 'diffusers': No module named 'diffusers'
SKIPPED [1] tests/backends/cuda/test_h3_sigma_schedule_cuda.py:378: multi-GPU device-guard test needs 2 devices
======================== 67 passed, 2 skipped in 4.86s =========================
```

</details>

<details><summary><code>run_b.log</code></summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.7.0 -- MINICONDA_PYTHON
cachedir: .pytest_cache
rootdir: REPO_ROOT
configfile: pyproject.toml
plugins: anyio-4.6.2.post1
collecting ... collected 48 items

tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_matches_pinned_cpu_fixture_bitwise PASSED [  2%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_matches_live_cpu_reference_bitwise[50-12.0] PASSED [  4%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_matches_live_cpu_reference_bitwise[50-3.0] PASSED [  6%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_matches_live_cpu_reference_bitwise[1000-12.0] PASSED [  8%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_dense_grid_dedup_matches_cpu_reference PASSED [ 10%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_reference_with_cuda_device_preserves_fixture_bytes PASSED [ 12%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_grid_invariants_on_device[2-3.0] PASSED [ 14%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_grid_invariants_on_device[2-12.0] PASSED [ 16%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_grid_invariants_on_device[4-3.0] PASSED [ 18%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_grid_invariants_on_device[4-12.0] PASSED [ 20%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_grid_invariants_on_device[25-3.0] PASSED [ 22%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_grid_invariants_on_device[25-12.0] PASSED [ 25%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_num_inference_steps_below_two[1] PASSED [ 27%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_num_inference_steps_below_two[0] PASSED [ 29%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_num_inference_steps_below_two[-4] PASSED [ 31%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_subnormal_shift_1e_45_matches_reference_upstream_parity PASSED [ 33%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_underflow_shift_1e_46_matches_reference_upstream_parity PASSED [ 35%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_positive_nan_shift_propagates_positive_quiet_nan PASSED [ 37%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_negative_nan_shift_propagates_negative_quiet_nan PASSED [ 39%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_edge_shifts_match_reference_upstream_parity[nan] PASSED [ 41%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_edge_shifts_match_reference_upstream_parity[inf] PASSED [ 43%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_edge_shifts_match_reference_upstream_parity[1e-10] PASSED [ 45%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_edge_shifts_match_reference_upstream_parity[1e+39] PASSED [ 47%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_edge_shifts_match_reference_upstream_parity[3.5e+38] PASSED [ 50%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_edge_shifts_match_reference_upstream_parity[1e+30] PASSED [ 52%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_float_num_steps_truncate_like_reference[2.9-2] PASSED [ 54%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_float_num_steps_truncate_like_reference[3.1-3] PASSED [ 56%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_non_positive_shift[0.0] PASSED [ 58%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_non_positive_shift[-1.5] PASSED [ 60%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_non_positive_shift[-inf] PASSED [ 62%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_rejects_cpu_device_instead_of_silent_fallback PASSED [ 64%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_repeated_and_interleaved_calls_are_stable PASSED [ 66%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_backend_identity_is_device_path_not_cpu_fallback PASSED [ 68%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_public_entry_defaults_to_reference_and_matches_triton_bits PASSED [ 70%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_public_entry_opt_in_selects_triton_and_reports_provenance PASSED [ 72%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_target_device_guard_with_mismatched_current_device SKIPPED [ 75%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_get_op_cpu_selects_reference_and_builds_pinned_grid PASSED [ 77%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_get_op_cuda_defaults_to_reference_regardless_of_triton PASSED [ 79%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_cuda_backend_construction_fails_loudly_without_triton PASSED [ 81%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_shift_scalars_pinned_to_cpu_under_meta_default PASSED [ 83%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_shift_scalars_keep_fp64_subtract_then_round_semantics PASSED [ 85%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_cuda_backend_construction_rejected_on_hip_build PASSED [ 87%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_get_op_on_hip_build_serves_reference PASSED [ 89%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_triton_backend_is_not_a_legacy_candidate_on_any_platform PASSED [ 91%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_semantic_descriptors_declared_truthfully PASSED [ 93%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_semantic_reference_path_resolves_instantiates_and_reports_provenance PASSED [ 95%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_semantic_triton_backend_fails_explicitly_when_unavailable PASSED [ 97%]
tests/runtime/test_h3_sigma_schedule_dispatch.py::test_semantic_unknown_backend_is_rejected PASSED [100%]

-------------- generated xml file: EVIDENCE_DIR/run_b.xml --------------
=========================== short test summary info ============================
SKIPPED [1] tests/backends/cuda/test_h3_sigma_schedule_cuda.py:378: multi-GPU device-guard test needs 2 devices
======================== 47 passed, 1 skipped in 2.89s =========================
```

</details>

<details><summary><code>run_c.log</code></summary>

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.7.0 -- MINICONDA_PYTHON
cachedir: .pytest_cache
rootdir: REPO_ROOT
configfile: pyproject.toml
plugins: anyio-4.6.2.post1
collecting ... collected 2 items

tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_public_entry_defaults_to_reference_and_matches_triton_bits PASSED [ 50%]
tests/backends/cuda/test_h3_sigma_schedule_cuda.py::test_public_entry_opt_in_selects_triton_and_reports_provenance PASSED [100%]

-------------- generated xml file: EVIDENCE_DIR/run_c.xml --------------
============================== 2 passed in 2.51s ===============================
```

</details>

<details><summary><code>bench.log</code></summary>

```text
path A (reference, cpu build, cpu outputs): 100 calls x [25, 50, 1000, 1000000] x [3.0, 12.0]
verifying path C against path A bitwise before timing...
WARNING 10-26 01:33:03 [RL-Kernel]: Core binary extension (_C) unavailable: cannot import name '_C' from 'rl_engine' (REPO_ROOT/rl_engine/__init__.py). Falling back to native code.
correctness gate passed (includes Triton kernel compilation: 1143.6 ms wall, reported once, excluded from all timed samples)
path B (reference, cpu build, outputs moved to gpu): 100 calls
path C (triton, device build): 100 calls
                        path_a_reference_cpu         N=25,shift=3: 0.0306 ms/call (median)
                        path_a_reference_cpu        N=25,shift=12: 0.0305 ms/call (median)
                        path_a_reference_cpu         N=50,shift=3: 0.0308 ms/call (median)
                        path_a_reference_cpu        N=50,shift=12: 0.0310 ms/call (median)
                        path_a_reference_cpu       N=1000,shift=3: 0.0347 ms/call (median)
                        path_a_reference_cpu      N=1000,shift=12: 0.0346 ms/call (median)
                        path_a_reference_cpu    N=1000000,shift=3: 5.4651 ms/call (median)
                        path_a_reference_cpu   N=1000000,shift=12: 5.5773 ms/call (median)
 path_b_reference_cpu_build_then_copy_to_gpu         N=25,shift=3: 0.0551 ms/call (median)
 path_b_reference_cpu_build_then_copy_to_gpu        N=25,shift=12: 0.0554 ms/call (median)
 path_b_reference_cpu_build_then_copy_to_gpu         N=50,shift=3: 0.0556 ms/call (median)
 path_b_reference_cpu_build_then_copy_to_gpu        N=50,shift=12: 0.0554 ms/call (median)
 path_b_reference_cpu_build_then_copy_to_gpu       N=1000,shift=3: 0.0598 ms/call (median)
 path_b_reference_cpu_build_then_copy_to_gpu      N=1000,shift=12: 0.0609 ms/call (median)
 path_b_reference_cpu_build_then_copy_to_gpu    N=1000000,shift=3: 5.5054 ms/call (median)
 path_b_reference_cpu_build_then_copy_to_gpu   N=1000000,shift=12: 5.8109 ms/call (median)
                  path_c_triton_device_build         N=25,shift=3: 0.0926 ms/call (median)
                  path_c_triton_device_build        N=25,shift=12: 0.0913 ms/call (median)
                  path_c_triton_device_build         N=50,shift=3: 0.0918 ms/call (median)
                  path_c_triton_device_build        N=50,shift=12: 0.0887 ms/call (median)
                  path_c_triton_device_build       N=1000,shift=3: 0.0907 ms/call (median)
                  path_c_triton_device_build      N=1000,shift=12: 0.0892 ms/call (median)
                  path_c_triton_device_build    N=1000000,shift=3: 1.3377 ms/call (median)
                  path_c_triton_device_build   N=1000000,shift=12: 1.3374 ms/call (median)
speedup B/C         N=25,shift=3: 0.60x (separated from dispersion)
speedup B/C        N=25,shift=12: 0.61x (separated from dispersion)
speedup B/C         N=50,shift=3: 0.61x (separated from dispersion)
speedup B/C        N=50,shift=12: 0.62x (separated from dispersion)
speedup B/C       N=1000,shift=3: 0.66x (separated from dispersion)
speedup B/C      N=1000,shift=12: 0.68x (separated from dispersion)
speedup B/C    N=1000000,shift=3: 4.12x (separated from dispersion)
speedup B/C   N=1000000,shift=12: 4.34x (separated from dispersion)
wrote EVIDENCE_DIR/benchmark-h800.json
```

</details>
