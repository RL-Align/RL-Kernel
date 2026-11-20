# C4 follow-up: new MI300X environment, 2026-10-29

This is a corrective follow-up to task 19 / PR #618, not a new task PR.

## Corrections

- The first group's four missing predecessors now use the existing aiter
  state-phase negative-position mask. The old finite `-2**30` gate floor was
  not a valid universal mask over the admitted gate domain. No handwritten
  numerical kernel or approximate masking workaround is used.
- Validate the predecessor context as well as the emitted current-group rows.
  An `emit_group_ids` filter cannot hide non-finite predecessor data.
- The independent oracle now preserves FP64 throughout softmax, pooling,
  normalization and rotation. The previous helper converted its inputs to
  FP32 internally, so the earlier report's description of that comparison as
  fully FP64 was incorrect. New regression tests verify FP64 perturbations
  survive and gradients remain FP64.
- Backward passes host-known unique context indices instead of iterating GPU
  index tensors and synchronizing each index.

## Dependency profile

The new image's existing aiter 0.1.16.post3 / FlyDSL 0.2.0 provider is admitted
by its exact source hash, alongside the previously pinned provider ABI.
The older ABI has a fixed single-wave path without a batch-size dispatch.
No numerical fallback is selected on a library error.

- PyTorch: `2.11.0+gitd0c8b1f`
- HIP: `7.2.53211`
- Triton: `3.6.0`
- GPU architecture: `gfx942`
- aiter C4 source SHA256:
  `0c2019ca3a4a0d6dc7d5fdd20ebcce658ff5f504fdc880dffd61c9c398029451`
- mLite functional backward source SHA256:
  `6a1e78ca89b83f56391b5e5ac409e765338f1e3a625251d0d4c877c1e80b4a68`

## Tests

- C4 CPU/GPU tests: **85 passed**, one warning, 23.66 seconds.
- Full targeted contract + DeepSeek-V4 suite: **227 passed**, zero skips,
  one warning, 34.38 seconds with coverage.
- Includes four extreme first-group score biases, hidden non-finite context,
  real FP64 oracle checks, real aiter launches, mLite VJP, streams and fresh
  subprocess runs.
- Changed-file Ruff lint/format checks pass; the 30-repetition benchmark was
  re-run on the new environment.
- Python wrapper coverage remains approximately 90%. Native instructions and
  autograd-internal tracing gaps are not evidence of native coverage; earlier
  statements describing all uncovered lines as constructor failures were too
  broad.

Logs, JUnit XML, coverage and benchmark JSON are under the member's mounted
`/outputs/c4-new-host-20261029/`, not the shared Docker writable layer.
The task-19 limits remain: no RCCL/multi-GPU CP, Graph replay, live cache
commit, full-model or end-to-end framework acceptance.

## November 3 timing follow-up

The current 30-repetition MI300X rerun is in
`reports/kv-compressor-c4-mi300x-20261103.json`. Each HIP event and synchronized
wall interval comes from the same pass; event intervals can include host
launch gaps and are not isolated kernel time. Bind, checked operator, and
raw provider calls have different scopes and must not be subtracted to claim
dispatch overhead. The source hashes are generated from the measured files.
The ROCm image returns an empty `torch.cuda.get_device_name()` but reports
`gfx942` as its architecture. No full-engine speedup is claimed.

| Groups | Bind event/wall us | Checked event/wall us | Provider event/wall us |
|---:|---:|---:|---:|
| 1 | 375.18 / 376.91 | 407.47 / 408.28 | 62.67 / 63.38 |
| 8 | 567.14 / 568.03 | 509.70 / 510.45 | 80.10 / 80.76 |
| 64 | 1938.57 / 1939.59 | 627.15 / 627.98 | 205.74 / 207.22 |
| 128 | 3271.41 / 3272.53 | 685.30 / 686.20 | 329.24 / 330.66 |

The November 3 targeted rerun passed 228 tests with one expected empty-Graph
warning; the October 29 coverage evidence above is historical.
