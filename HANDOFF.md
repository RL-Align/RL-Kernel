# RFC #642 / Task 12 handoff

Updated 2026-11-19 UTC. Owner: Nankago. CUDA/Hopper forward projection only;
Task 13 gradients and MUSA qualification remain separate.

- Recovered `../task12-preflight.md`, the unrelated root handoff and three runtime
  repair reports. No prior Task 12 implementation, checkout or test was found.
- Isolated checkout: `RL-Kernel-task12`, branch
  `codex/task12-deterministic-gemm`, base upstream main
  `49941801f942428fd52ac5b66a1b0c070418724e`. Clean before this work; one worktree.
- Refreshed RFC and all 21 comments, PRs #679, #450, #697 and #707. Shared forward
  fixes/tests independently supported by main; no miles/model adapter changes.
  #679 scalar recursion/index fixes are not duplicated. #450 already identifies
  the stale Triton test import; only its import correction is needed here.
- No applicable AGENTS.md. Contributor guide requires DCO signoff, canonical
  source ownership, focused validation and separate accuracy/bitwise/performance.
- Default executor cannot spawn; `/bin/bash`, `login=false`, host execution works.
  Workspace create/read/rename/delete passed. Do not alter daemon/configuration.
- Eight H100 80GB HBM3, driver 580.126.20, idle at resume. Use only GPU 0 UUID
  `GPU-25dc68ae-c0e6-49b1-23b5-c4da9d63e8ef`, recheck before starting work.
- Environment `../.venv-task12` inherits system packages read-only: Python 3.12,
  torch 2.14.0a0+4fdf77b940.nv26.08, CUDA 13.4, Triton 3.8.0+git4c7e62b7.nv26.8.
- Baseline and repaired full SM90 extension builds completed successfully:
  `CUDA_VISIBLE_DEVICES=<GPU0 UUID> CUDA_HOME=/usr/local/cuda MAX_JOBS=4
  KERNEL_ALIGN_DET_GEMM_SM90=1 KERNEL_ALIGN_FORCE_SM90=1
  KERNEL_ALIGN_USE_FAST_MATH=0 RL_KERNEL_REQUIRE_EXT=1
  ../.venv-task12/bin/python setup.py build_ext --inplace`.
  Logs: `../task12-artifacts/build-baseline.log`, `build-after.log`.
- Original binary preserved outside the repository. CUDA implementation,
  validation and timing are complete. Code commit:
  `b3ddb9df22d7810116ad168182e689f6e699d843`; DCO signed as Nankago using the
  account-ID noreply address. The follow-up evidence commit contains this final
  handoff/report; use `git log -1` for its SHA.
- Fixed model config downloaded at HF revision
  `345d467b117d1fd5eda9b879a5988de9ae5f47a7`; official Transformers v5.13.1
  call-chain source in `../task12-artifacts/`. Do not commit downloaded sources.
- GitHub connector identity confirmed Nankago. Fork GET remains 404; no CLI gh
  or authenticated HTTPS push established. Never push upstream/default branch.
  No available connector fork-creation endpoint. Continue local delivery if
  publication is unavailable; preserve commit/bundle for later push.
  `origin=https://github.com/Nankago/RL-Kernel.git`; upstream push URL is DISABLED.
  Authorized task-branch push was attempted and failed with
  `fatal: could not read Username for 'https://github.com': terminal prompts disabled`;
  log `../task12-artifacts/push-code.log`. No upstream write or PR/comment occurred.
- Baseline build completed successfully. Preserved original binary at
  `../task12-artifacts/baseline-extension.so`. Cast regressions on unmodified
  main: 4 failed, 4 profile-inapplicable skips, 144 deselected, 3.97 s; log/XML
  `baseline-cast-tests.log`, `baseline-cast.xml`. Actual first blocker is missing
  CUDA pybind exports for `det_gemm_fwd_fp32` and `det_gemm_rowwise_fwd_fp32`.
  The first benchmark attempt failed on that missing API, without timing claims.
  Updated bounded benchmark records missing API explicitly and times BF16 plus
  output-equivalent BF16-tree-to-FP32 widening. Baseline timing running now.
- Incremental fix build succeeded. Changes: expose existing CUDA tree-FP32 and
  SM90 rowwise-FP32 entry points; select the already implemented float store;
  document BF16 tree versus full-FP32 profiles; restore stale Triton test imports
  with unexpected ImportError re-raised. No gradient body changed.
- New matrix completed: **158 passed**, zero skips/failures, 596.93 s (first
  Triton compilation included); `../task12-artifacts/projection.log` and XML.
  Twenty pytest xunit2 property-format warnings; use `-o junit_family=xunit1`
  for subsequent XML. FP64 target maximum absolute error 0.037068993886350654;
  all 20 backend/shape accuracy cases pass unchanged contract limits. All raw
  comparisons have zero mismatches; signed-zero negative controls pass.
- Independent CPU-FP64 dyadic K-tree goldens plus restored CUDA/Triton forward
  regressions: **38 passed, 65 deselected**, 83.12 s; `profiles-regression.log/xml`.
- Final reference-helper/handoff check: **6 passed, 152 deselected**, 6.80 s;
  `reference-handoff.log/xml`. MUSA reference uses CPU FP64, avoiding a device
  FP64 GEMM dependency; CUDA reference precision/device are unchanged.
- Profiler confirms `det_gemm_sm90_kernel<float,false>` for [129,2048,512]
  and `det_gemm_naive<float,false>` for [129,2048,32]; `kernel-provenance.json`.
- Baseline benchmark completed: `../task12-artifacts/benchmark-before.json`.
  Original extension SHA256
  `e8c207324aad3ca02feabad6735417c6e2001de4b0dc638922612e01232fd488`;
  empty production diff hash confirms original source. API forward_fp32 is marked
  binding_unavailable; BF16 and equivalent FP32 widen timing recorded separately.
- Repaired benchmark completed at code commit b3ddb9d:
  `../task12-artifacts/benchmark-after.json`. Same seed/device, 10 warmups and
  30 CUDA-event samples. BF16 median changes -0.45% to +1.63%; FP32 direct output
  comparable to original BF16-result widening. LM-head M=1 FP32 padded temporary
  adds 61.10 MiB peak allocation. Tables/p95/memory/hashes are committed in
  `reports/experiments/task12-projection-gemm.md` and `.json`.
- Preserved original and repaired binaries in separate processes: 13 K/N pairs
  (all targets plus ragged cases), M=7, seed 649, BF16 and FP32 widened/direct
  outputs: **26 comparisons, zero differing bytes**. Local scripts/results:
  `compare_binary.py`, `binary-{baseline,repaired}.{json,pt}` and
  `binary-comparison.json`, all under `../task12-artifacts/`.
- CPU layout **33 passed, 1 extension-smoke skip**; attention projection **8
  passed**. Fixed-version Black/isort/flake8, wrapper mypy and diff checks pass.
  Dev tools installed only in `../.venv-task12`; no global/project environments
  altered. All test processes above have exited; no duplicate GPU suite running.
- English PR draft: `reports/experiments/task12-pr-draft.md`; no PR created.
- Remaining blockers: fork visibility/CLI authentication (publication only) and
  MUSA hardware/collaborator qualification. The report gives single-device MUSA
  commands with a CPU FP64 oracle and excludes CUDA-specific tree probes.
  H200, cuBLASLt, non-finites, zero dimensions, large K, engine/full-model,
  distributed and gradient acceptance remain outside the measured scope.
- All Task 12 build/test/benchmark processes have exited; no live task to resume.
  Next: make fork accessible or create it through an authenticated Nankago
  connection, then `git push -u origin codex/task12-deterministic-gemm` without
  force. Existing authorization covers the task branch; no new approval needed.
  Do not rerun passing tests unless code/stack changes. Latest pushed commit: none.
  A portable bundle is saved after final commit at
  `../task12-artifacts/task12-deterministic-gemm.bundle`; restore/fetch its branch
  before pushing if this workspace is lost. Large logs, binaries, synthetic
  tensors, downloaded model sources and the venv are not committed.
  No upstream comments/PR/merge authorized. Latest pushed commit: none.
