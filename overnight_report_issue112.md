# Overnight Report - RL-Kernel Issue #112

## Current branch

`cse/issue-112-deterministic-nccl`

## Environment

- Host: `dedicated-developjob-wtl-t1wjo-7c6d5f4d56-qzkfm`
- GPU: 8x `NVIDIA L20X` reported by `nvidia-smi`
- CUDA: driver reports CUDA 12.8, driver version 570.172.08
- PyTorch: `torch 2.11.0+cu128` in `.codex-nightly/envs/issue112-py312`
- NCCL: `torch.cuda.nccl.version()` reports `(2, 28, 9)`
- Python: system `python3` is 3.12.3; isolated venv is `.codex-nightly/envs/issue112-py312`; `python` is not on PATH
- Commit base: `a302be4593bc1715688558a7eb3d3704bf625c4d`

## Work completed

- Re-checked issue #112 and roadmap issue #83.
- Created local working branch `cse/issue-112-deterministic-nccl`.
- Audited repository call sites for all-reduce, reduce-scatter, all-gather,
  DDP/FSDP, DeepSpeed, NCCL, and gradient synchronization keywords.
- Added a distributed audit document.
- Created an isolated workspace-local Python environment under `.codex-nightly/envs/issue112-py312`.
- Installed PyTorch CUDA 12.8 and RL-Kernel dev/docs dependencies into that isolated environment.

## Commits created

- `docs(distributed): audit all-reduce call sites for issue 112`

## Files changed

- `docs/.nav.yml`
- `docs/distributed/deterministic_allreduce_audit.md`
- `overnight_report_issue112.md`

## Tests run

- `git diff --check`
- `.codex-nightly/envs/issue112-py312/bin/python -c "import torch; ..."`
- `.codex-nightly/envs/issue112-py312/bin/mkdocs build --strict -f mkdocs.yaml`

## Test results

- `git diff --check` passed.
- PyTorch environment check passed: CUDA available, 8 devices visible, NCCL available.
- `mkdocs build --strict -f mkdocs.yaml` passed. It emitted expected git revision-date warnings for the new uncommitted audit doc.

## GPU validation results

No distributed GPU validation has been run yet. `nvidia-smi` is available and reports 8 GPUs. The isolated venv can import PyTorch with CUDA and NCCL support.

## PR split recommendation

- PR 1: audit and deterministic all-reduce contract documentation.
- PR 2: deterministic all-reduce helper with ordered rank fallback and smoke
  tests.
- PR 3: NCCL ring fast path and GPU validation report, after PyTorch/NCCL runtime
  is available.

## PR body files created

None yet.

## Blockers

- `python` is not available on PATH.
- System `python3 -m venv` cannot create venvs because `ensurepip` is unavailable; used workspace-local `virtualenv` bootstrap instead.
- Current GPU model reported by `nvidia-smi` is `NVIDIA L20X`, not the H200 model
  assumed by the original overnight manual.

## Unsafe operations skipped

- No push attempted.
- No system dependency installation attempted; all Python dependencies were installed under `.codex-nightly/`.
- No upstream branch or PR creation attempted.

## Remaining work

- Commit the audit phase.
- Add deterministic all-reduce helper API and focused distributed tests.
- Use `.codex-nightly/envs/issue112-py312` for GPU/NCCL validation.
- Prepare PR body drafts and patch artifacts.

## Suggested next Codex prompt

Continue issue #112 from `overnight_report_issue112.md`. Prioritize the next
reviewable phase: add `rl_engine.distributed.deterministic_allreduce` with an
ordered-rank fallback and focused smoke tests, then update this report.
