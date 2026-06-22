# Overnight Report - RL-Kernel Issue #112

## Current branch

`cse/issue-112-deterministic-nccl`

## Environment

- Host: `dedicated-developjob-wtl-t1wjo-7c6d5f4d56-qzkfm`
- GPU: physical machine confirmed by user as H200; `nvidia-smi` labels devices as `NVIDIA L20X`
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
- Added `docs/distributed/deterministic_allreduce_audit.md`.
- Created an isolated workspace-local Python environment under `.codex-nightly/envs/issue112-py312`.
- Installed PyTorch CUDA 12.8 and RL-Kernel dev/docs dependencies into that isolated environment.
- Added `rl_engine.distributed.deterministic_allreduce` with:
  - `configure_deterministic_nccl_env()`;
  - `DeterministicAllReduceConfig`;
  - `nccl_ring` fast path using `torch.distributed.all_reduce`;
  - `ordered_rank_fallback` using all-gather, rank-ordered accumulation on rank 0, and broadcast.
- Added user-facing docs for deterministic all-reduce modes and fallback behavior.
- Added a torchrun-compatible distributed all-reduce smoke test.
- Added a DP gradient fixed-step smoke test comparing a DP=1 full-batch baseline against DP=N local gradients reduced with the deterministic all-reduce helper.
- Created local PR body drafts `.pr_body_issue112_pr1.md`, `.pr_body_issue112_pr2.md`, and `.pr_body_issue112_pr3.md`.
- Generated patch artifacts under `.codex-nightly/artifacts`.

## Commits created

- `2bccb64 docs(distributed): audit all-reduce call sites for issue 112`
- `feat(distributed): add deterministic all-reduce helper`
- `test(distributed): compare DP gradients against single-rank baseline`

## Files changed

- `docs/.nav.yml`
- `docs/distributed/deterministic_allreduce_audit.md`
- `docs/distributed/deterministic_allreduce.md`
- `rl_engine/distributed/__init__.py`
- `rl_engine/distributed/deterministic_allreduce.py`
- `tests/distributed/test_deterministic_allreduce.py`
- `tests/distributed/test_dp_gradient_determinism.py`
- `overnight_report_issue112.md`

## Tests run

- `git diff --check`
- `.codex-nightly/envs/issue112-py312/bin/python -c "import torch; ..."`
- `.codex-nightly/envs/issue112-py312/bin/black --check --line-length 100 rl_engine/distributed tests/distributed/test_deterministic_allreduce.py`
- `.codex-nightly/envs/issue112-py312/bin/isort --check-only --profile black --line-length 100 rl_engine/distributed tests/distributed/test_deterministic_allreduce.py`
- `.codex-nightly/envs/issue112-py312/bin/flake8 --max-line-length=100 --extend-ignore=E203,E704 rl_engine/distributed tests/distributed/test_deterministic_allreduce.py`
- `.codex-nightly/envs/issue112-py312/bin/ruff check rl_engine/distributed tests/distributed/test_deterministic_allreduce.py`
- `.codex-nightly/envs/issue112-py312/bin/python -m mypy --ignore-missing-imports rl_engine/distributed`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .codex-nightly/envs/issue112-py312/bin/python -m pytest tests/distributed/test_deterministic_allreduce.py -q`
- `.codex-nightly/envs/issue112-py312/bin/torchrun --standalone --nproc_per_node=2 tests/distributed/test_deterministic_allreduce.py --backend gloo --mode ordered_rank_fallback --dtype fp32 --device cpu`
- `CUDA_VISIBLE_DEVICES=0,1 NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 .codex-nightly/envs/issue112-py312/bin/torchrun --standalone --nproc_per_node=2 tests/distributed/test_deterministic_allreduce.py --backend nccl --mode nccl_ring --dtype fp32 --device cuda`
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 .codex-nightly/envs/issue112-py312/bin/torchrun --standalone --nproc_per_node=8 tests/distributed/test_deterministic_allreduce.py --backend nccl --mode nccl_ring --dtype fp32 --device cuda`
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .codex-nightly/envs/issue112-py312/bin/torchrun --standalone --nproc_per_node=8 tests/distributed/test_deterministic_allreduce.py --backend nccl --mode ordered_rank_fallback --dtype fp32 --device cuda`
- `.codex-nightly/envs/issue112-py312/bin/mkdocs build --strict -f mkdocs.yaml`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .codex-nightly/envs/issue112-py312/bin/python -m pytest rl_engine/tests/test_dispatch.py -v`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .codex-nightly/envs/issue112-py312/bin/python -m pytest tests/distributed/test_dp_gradient_determinism.py -q`
- `.codex-nightly/envs/issue112-py312/bin/torchrun --standalone --nproc_per_node=2 tests/distributed/test_dp_gradient_determinism.py --backend gloo --mode ordered_rank_fallback --dtype fp32 --device cpu`
- `CUDA_VISIBLE_DEVICES=0,1 NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 .codex-nightly/envs/issue112-py312/bin/torchrun --standalone --nproc_per_node=2 tests/distributed/test_dp_gradient_determinism.py --backend nccl --mode nccl_ring --dtype fp32 --device cuda`
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 .codex-nightly/envs/issue112-py312/bin/torchrun --standalone --nproc_per_node=8 tests/distributed/test_dp_gradient_determinism.py --backend nccl --mode nccl_ring --dtype fp32 --device cuda`
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .codex-nightly/envs/issue112-py312/bin/torchrun --standalone --nproc_per_node=8 tests/distributed/test_dp_gradient_determinism.py --backend nccl --mode ordered_rank_fallback --dtype fp32 --device cuda`

## Test results

- `git diff --check` passed.
- PyTorch environment check passed: CUDA available, 8 devices visible, NCCL available.
- black, isort, flake8, ruff, and mypy passed for the new Python files.
- `tests/distributed/test_deterministic_allreduce.py`: 3 passed.
- 2-rank CPU/Gloo ordered fallback smoke passed with bitwise equality.
- 2-rank CUDA/NCCL `nccl_ring` smoke passed with bitwise equality against the ordered fallback oracle.
- 8-rank CUDA/NCCL `nccl_ring` smoke passed within tolerance against the ordered fallback oracle; it was not bitwise equal.
- 8-rank CUDA/NCCL ordered fallback smoke passed with bitwise equality.
- `mkdocs build --strict -f mkdocs.yaml` passed. It emitted expected git revision-date warnings for new uncommitted docs.
- `rl_engine/tests/test_dispatch.py`: 3 passed.
- DP gradient unit test: 1 passed.
- 2-rank CPU/Gloo DP gradient ordered fallback smoke passed against DP=1 baseline.
- 2-rank CUDA/NCCL DP gradient `nccl_ring` smoke passed against DP=1 baseline.
- 8-rank CUDA/NCCL DP gradient `nccl_ring` smoke passed against DP=1 baseline.
- 8-rank CUDA/NCCL DP gradient ordered fallback smoke passed against DP=1 baseline.

## GPU validation results

Physical machine is user-confirmed H200, while `nvidia-smi` labels devices as `NVIDIA L20X`. No NVLS- or H200-specific support claim is made from this label mismatch.

2-rank NCCL ring smoke result:

```json
{"backend":"nccl","bitwise_equal":true,"device":"cuda:0","dtype":"fp32","iterations":3,"max_abs_diff":0.0,"max_rel_diff":0.0,"mismatch_count":0,"mode":"nccl_ring","op":"sum","status":"pass","world_size":2}
```

8-rank NCCL ring smoke result:

```json
{"backend":"nccl","bitwise_equal":false,"device":"cuda:0","dtype":"fp32","iterations":3,"max_abs_diff":4.76837158203125e-07,"max_rel_diff":3.2424927098873013e-07,"mismatch_count":62,"mode":"nccl_ring","op":"sum","status":"pass","world_size":8}
```

8-rank ordered fallback smoke result:

```json
{"backend":"nccl","bitwise_equal":true,"device":"cuda:0","dtype":"fp32","iterations":3,"max_abs_diff":0.0,"max_rel_diff":0.0,"mismatch_count":0,"mode":"ordered_rank_fallback","op":"sum","status":"pass","world_size":8}
```

8-rank DP gradient NCCL ring smoke result:

```json
{"backend":"nccl","bitwise_equal":false,"device":"cuda:0","dtype":"fp32","global_batch_size":16,"max_abs_diff":5.960464477539063e-08,"max_rel_diff":6.11946063600044e-07,"mismatch_count":81,"mode":"nccl_ring","status":"pass","world_size":8}
```

8-rank DP gradient ordered fallback smoke result:

```json
{"backend":"nccl","bitwise_equal":false,"device":"cuda:0","dtype":"fp32","global_batch_size":16,"max_abs_diff":2.9802322387695312e-08,"max_rel_diff":3.059730261156801e-06,"mismatch_count":74,"mode":"ordered_rank_fallback","status":"pass","world_size":8}
```

## PR split recommendation

- PR 1: audit and deterministic all-reduce contract documentation.
- PR 2: deterministic all-reduce helper with ordered rank fallback and smoke tests.
- PR 3: split DP gradient fixed-step comparison if maintainers prefer it separate from the helper.
- PR 4: NVLS/NVLink-Sharp probe and documentation only if hardware and logs prove it.

## PR body files created

- `.pr_body_issue112_pr1.md`
- `.pr_body_issue112_pr2.md`
- `.pr_body_issue112_pr3.md`

## Blockers

- `python` is not available on PATH.
- System `python3 -m venv` cannot create venvs because `ensurepip` is unavailable; used workspace-local `virtualenv` bootstrap instead.
- Hardware label mismatch: physical machine is user-confirmed H200, while `nvidia-smi` labels devices as `NVIDIA L20X`.
- NVLS has not been probed or validated.
- DeepSpeed DP gradient synchronization order is not controlled by the new helper yet; the DP gradient smoke uses a tiny local model rather than DeepSpeed internals.

## Unsafe operations skipped

- No push attempted.
- No system dependency installation attempted; all Python dependencies were installed under `.codex-nightly/`.
- No upstream branch or PR creation attempted.
- No sudo used.

## Remaining work

- Probe NVLS only if the current hardware/software setup clearly supports it.

## Suggested next Codex prompt

Continue issue #112 from `overnight_report_issue112.md`. Prioritize PR body drafts, patch artifact generation, and optional NVLS probing only if the hardware/software logs clearly prove support.
