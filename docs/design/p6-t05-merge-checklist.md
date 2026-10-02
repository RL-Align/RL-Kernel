# P6 T05 merge checklist

Audit date: 2026-10-03. Target is the proposed T01 contract at PR #448,
commit `05bc53aa0396ffaf2c9241559b388159563e6b36`. No upstream merge or owner
approval is implied by a local pass.

## Authoritative requirements

- [Official contributor guide](https://github.com/RL-Align/RL-Kernel/blob/main/docs/contributing/README.md): implementation/dispatch, focused correctness, dedicated operator page, navigation and strict docs build.
- [Official testing guide](https://github.com/RL-Align/RL-Kernel/blob/main/docs/contributing/testing.md) and [pre-commit hooks](https://github.com/RL-Align/RL-Kernel/blob/main/.pre-commit-config.yaml): correctness plus Black 24.4.2, isort 5.13.2, flake8 7.0.0, whitespace/YAML/large-file checks.
- [T01 PR #448](https://github.com/RL-Align/RL-Kernel/pull/448) and the member handoff T05 contract: byte-equal fusion, debug identity/RNG, recorded/mock schedule invariance and fail-closed unsupported cases. The profile/ABI is still proposed.

## Requirement-to-test mapping

| Requirement | Executable coverage in `tests/p6/test_fused_combine.py` |
| --- | --- |
| T02→T03→T04 equals fusion, every debug boundary | `test_gpu_frozen_bytes_debug_on_and_off`, all nine goldens; blocks 128/256/512 |
| Correct arithmetic, once-only branches/cast | Frozen cancellation, BF16 ties, signed zero, zero-routes; FP32/BF16 overflow and subnormal tests |
| Debug changes neither bytes nor RNG/logical identity | Frozen test asserts CPU/CUDA RNG, plan/order hash, raw stages and output |
| Same ExchangePlan under backend/topology/chunk/arrival/delay/overlap | `test_gpu_same_exchange_p4_mock_chunk_topology_arrival_and_overlap`, EP1/2/4/8, rotated placement, zero peers, reordered chunks, two CUDA streams |
| Batch/padding/physical-order invariance | `test_gpu_batch_partition_preserves_each_token`, permutation/NaN padding test |
| No default `[T,6,H]` materialization | `test_gpu_default_buffers_and_launch_do_not_materialize_canonical`, allocated shapes and warmed PyTorch allocator peak; not a whole-process/native VRAM certificate |
| Graph replay with changed values; status bad→good | `test_gpu_graph_replay_changes_output_and_rechecks_status` |
| Invalid mapping/context/input values/layout/backend fail closed | Host metadata matrix, four context identities, CPU backend; all three tensor roles; branch numeric cases |
| No input/write-buffer alias races | Host nonzero-offset alias matrix and GPU output/debug alias rejection |
| T01 registry replacement and actual byte evidence | Nine candidate/recorded comparisons, scoped provider gates and sealed readback roundtrip |
| sm80/sm90 compile, debug on/off | 12 opt-in offline compiler cases, not H100 runtime certification |

The recorded/mock EP test models completion schedules; it does not execute
network traffic, EP reductions or live Foundation. It preserves the exact plan
fingerprint instead of creating a second plan/schema.

## Local checks / CI

Run at the checkout root, using a dedicated environment or an already working
PyTorch/Triton pair. Missing development-only tools can be installed separately;
do not replace the user's training environment's torch.

```bash
python -m black --workers 1 --check --line-length=100 rl_engine/p6/combine.py rl_engine/p6/combine_provider.py rl_engine/kernels/ops/triton/moe tests/p6/test_fused_combine.py benchmarks/p6_combine.py
python -m isort --check-only --profile black --line-length=100 rl_engine/p6/combine.py rl_engine/p6/combine_provider.py rl_engine/kernels/ops/triton/moe tests/p6/test_fused_combine.py benchmarks/p6_combine.py
python -m flake8 --max-line-length=100 --extend-ignore=E203,E704 rl_engine/p6/combine.py rl_engine/p6/combine_provider.py rl_engine/kernels/ops/triton/moe tests/p6/test_fused_combine.py benchmarks/p6_combine.py
python -m ruff check rl_engine/p6 tests/p6 rl_engine/kernels/ops/triton/moe benchmarks/p6_combine.py
python -m ruff format --check rl_engine/p6 tests/p6 rl_engine/kernels/ops/triton/moe benchmarks/p6_combine.py
P6_COMPILE_T05=1 OMP_NUM_THREADS=1 python -m pytest tests/p6 tests/test_moe_merge.py -q
python -m mypy --ignore-missing-imports rl_engine/
mkdocs build --strict -f mkdocs.yaml
git diff --check
# After staging the exact T05 files, before committing:
pre-commit run
```

P6 CI covers Python 3.10/3.12 CPU conformance, changed kernel/provider/test/benchmark
paths, the official focused formatters, offline sm80/sm90 compile, strict docs,
JUnit and immutable reference artifacts. It does not mark skipped GPU tests as a
GPU pass. Actual GPU CI/performance orchestration is T09, not duplicated here.
The general upstream workflow's branch filters do not by themselves validate a
stacked PR targeting #448's feature branch; keep this focused P6 workflow.

Before commit, new Markdown files have no Git revision history and the docs date
plugin warns in strict mode. Validation uses a disposable Git snapshot with the
same document content; the real PR CI sees the contributor's committed history.
Do not disable strict checking or commit fake history in the development branch.

## Evidence and remaining gates

- User reported A100 run of the pre-audit patch: **30 passed, 12 skipped**,
  Python 3.12.13, torch 2.11.0+cu130, Triton 3.6.0. This covers the earlier
  11 CPU and 19 GPU cases, not the newly added cases/fix.
- The final-audit A100 suite passed **91 tests (43 CPU + 48 CUDA), 12 skipped**
  in 7.56s, reported by the user. Attempt-002 verification reported 9 cases,
  45 operator recordings and `ARTIFACT_INTEGRITY_AND_CPU_REPLAY_PASS`;
  `gpu_reexecuted=false`, `production_certified=false`. Remote JUnit and the
  complete immutable seal directory have not been copied into this repository.
- Six shared-A100 benchmark runs completed with BYTE_EQUAL and a source digest
  matching the local implementation. Ray was present, with GPU utilization 0%
  before and 97% after; these observations do not prove stable speedup.
- Run Compute Sanitizer if available, or explicitly report tooling unavailable.
  It is a recommended safety check, not an observed pass or an invented official gate.
- Measure H=4096 at T=32 and T=256 on an idle/allocated A100. Keep correctness
  checks before timing, both GPU-event/wall latency, source/input hashes and JSON.
  No numerical speedup floor was specified; do not hide checked-API overhead.
- H100 execution is still pending; offline sm90 compile is not equivalent.
  ROCm/Ascend, real EP and production Foundation remain out of this T05 backend.
- Request review by T02/T03/T04 owners as required by the handoff. T01 owner must
  approve provisional ABI, first-valid/invalid/signed-zero policy and integration.
  #448 remains open: submit as a stacked PR against its branch, not as if the base
  contract had merged or production acceptance were already approved.
- Whole-repository pytest currently has the inherited ROCm collection blocker
  (`_AITER_FWD_REQUIRED_KEYWORDS` missing). This audit does not silently fix an
  unrelated backend or report the full repository as passing.
- Whole-repository MyPy reports 49 errors in 15 unchanged files in this local
  environment (MyPy 2.4.0/Python 3.13). The clean base snapshot produces the
  identical 49 errors; the three added T05 implementation modules pass focused
  MyPy. Report the inherited gate issue to the base owner, not as a green full CI.

## Shared-A100 observations

Raw results and environment logs are in
`docs/validation/p6-t05-a100-2026-10-03`; the six JSON files are byte-for-byte
copies. The two nvidia-smi text logs normalize trailing whitespace only.
Three repetitions per shape use the same inputs, not three independent seeds.
All timing values below are per-run medians in microseconds.

| T / run | Eager reference Event | Prepared launch Event | Checked eager Event | Prepared wall |
| --- | ---: | ---: | ---: | ---: |
| 32 / 1 | 627.200 | 57.344 | 201.744 | 82.756 |
| 32 / 2 | 201.728 | 11.264 | 823.264 | 580.536 |
| 32 / 3 | 610.304 | 58.368 | 199.360 | 82.385 |
| 256 / 1 | 645.632 | 74.752 | 632.352 | 101.557 |
| 256 / 2 | 333.824 | 65.536 | 2461.136 | 89.394 |
| 256 / 3 | 331.776 | 29.696 | 2451.568 | 1203.447 |

The reference is eager fixed-order PyTorch arithmetic, not an optimized native
T02-T04 baseline. Timing phases run sequentially on a shared card with no
continuous telemetry or exclusive allocation. Event intervals are not pure
kernel timing; prepared launch excludes allocation/status readback, whereas
checked eager includes them. Process peak includes reference/debug buffers,
not kernel-only memory. Do not turn these ratios into a 5-18x production claim
or discard the slower checked results. Repeat performance on an idle/allocated
card before claiming stable acceleration.

Local observed results are recorded in `docs/validation/p6-t05-2026-10-02.json`.
Correctness evidence and a scoped performance experiment are available for draft
review; owner approval and controlled performance remain open. This is not an
unconditional production-complete/merge claim.
