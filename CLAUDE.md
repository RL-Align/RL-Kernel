# CLAUDE.md — DSv4 CUDA work on `test-dsv4-cuda` (PR #635)

Handoff notes for continuing the DSv4 SM90 batch-invariant work. The plan of record is
`docs/architecture/dsv4-cuda-design.md` (decisions D1-D7, phases, todo); read it first and
keep its checkboxes current. Last updated 2026-11-09 at `ec257d43` (pushed).

## Working with the user

- Reply in Traditional Chinese (technical terms may stay English); the user writes a mix.
- Work step by step: explain why, how and in what order, and confirm before starting a large
  step. Commit only when asked; push only when asked.
- Commits: conventional style (`feat(dsv4): ...`, `fix(dsv4): ...`), `git commit -s` (DCO),
  end with the Co-Authored-By line. Split pure formatting from logic changes.
- Goal right now: every DSv4 operator runs, is batch-invariant (bitwise), and is within
  1.3x of the non-invariant path (naive PyTorch, or an efficient non-BI kernel, same shapes).

## Settled decisions (beyond the design doc's D1-D7)

- Integration target and formal #230 anchor: **verl** (rollout and actor recompute on the
  same kernel set). The vLLM fork (`ISEEKYAN/vllm@1a6542c`) is the semantic reference for
  rollout bytes; its golden is deferred.
- CUDA SM90 first; Triton paths deferred; ROCm out of scope.
- P3 and P7 are redone on this branch; the old P-dev branches are ignored. Merge main last.
- Dense FP8 GEMMs (q/kv/o/indexer projections and the shared expert) use **unmodified
  DeepGEMM** pinned to `vllm-project/DeepGEMM@e21c821f` (D3; probe passed, forward and
  backward). Routed experts stay on P5 `csrc/cuda/moe/sm90_fused_moe_mlp.cu` (DeepGEMM runs
  FP4 only on SM100).
- Shared expert moved to DeepGEMM FP8 (forward and dX). The det_gemm BF16 backward stays as
  a fallback/reference (`backward="bf16_det_gemm"`). Shared-expert weights are frozen: dX only.
- Keep `csrc/cuda/moe/mxfp8_mxfp4_grouped_gemm_wgmma.cu` although nothing references it.

## Done on this branch (newest last)

| Commit | What |
|---|---|
| `c2de2741`, `900f01d4`, `9df841a1` | Phase 0: pre-commit formatting; SM90 MoE build gating (`RL_KERNEL_ENABLE_MOE_SM90`, sm_90a only), fast-math error on ROCm, fp32_gemm_rms G8 N=31 fix, benchmark fixes; merge main |
| `d1bfa2b5`, `4d6e033e` | Phase 1: `scripts/probe_dsv4_deepgemm_bi.py` — DeepGEMM BI forward/dgrad, deterministic wgrad, perf, negative control |
| `06e68f7b` | `fp8_group_quant` (golden `rl_engine/kernels/ops/pytorch/quant/`, CUDA `csrc/cuda/quant/`) — byte-exact with vLLM `per_token_group_fp8_quant` (group 128, eps 1e-10, UE8M0); token-direction variant for wgrad |
| `5252ccb8` | `dsv4` extra pin + `rl_engine/kernels/ops/cuda/deepgemm.py` (`require_deepgemm`, fail-closed); `rl_engine/moe/backends/deepgemm_shared_expert.py` (profile `dsv4-deepgemm-fp8-v1`) |
| `ec257d43` | `csrc/cuda/activation/silu_and_mul_clamp.cu`: vLLM clamped SwiGLU (limit 10) fused with FP8 quant, plus backward; shared expert now 0.71-0.78x of eager BF16 |

## Next steps (agreed order)

1. Housekeeping: tick "Merge main" (Phase 0) and "DeepGEMM backward determinism" (Phase 4) in
   the design doc; fix det_gemm launching an empty grid for M=0; decide whether the P5
   oracle's shared-expert SwiGLU should clamp like the rollout.
2. A general DSv4 dense FP8 linear operator (fp8_group_quant + DeepGEMM + weight prep),
   extracted from the shared-expert provider, for the attention and indexer projections.
3. Phase 2 (critical path): T04 stable top-k first (its comparator is shared with P3's
   `stable_topk6`), then T03 indexer (needs a BI probe of DeepGEMM `fp8_mqa_logits`), T02 KV
   compressor, T05 FlashMLA sparse wrapper (new dependency: pin + probe like DeepGEMM).
4. P3 router redo. Check first: D4 wants `det_gemm_fwd_fp32`, which is not in the default
   build of `rl_engine._C`.

## Environment (TWCC-style cluster; paths are specific to this account)

- GPU nodes via Slurm (`normal2` = H200, SM90). Login nodes have no GPU. `/home` and `/work`
  are shared; `/tmp` (and the Claude scratchpad) is node-local.
- `gh` is not installed: use `curl https://api.github.com/repos/RL-Align/RL-Kernel/...`.
- `ls` is aliased to `eza` (missing): use `command ls`.
- Python: conda env `cosq` (torch 2.9.0+cu128). Do not pip-install into it. Work venv:
  `~/dsv4-work/venv` (`--system-site-packages` on cosq) with pytest, pre-commit, a real
  `ninja` binary (cosq's `ninja` module has none), and `deep_gemm 2.6.1+e21c821`.
- CUDA for building `rl_engine._C`: `module load cuda/12.4` (this sets `CUDA_HOME`).
- CUDA for DeepGEMM's JIT: must be 12.8+. Conda toolkit at `/work/u4478697/cuda128`, with a
  standard-layout symlink facade `/work/u4478697/cuda128-home` (conda puts headers/libs under
  `targets/x86_64-linux`). Run DeepGEMM code with
  `CUDA_HOME=/work/u4478697/cuda128-home PATH=/work/u4478697/cuda128-home/bin:$PATH`.
  `module load cuda/12.4` breaks DeepGEMM's JIT (`require_deepgemm` now says so).
- DeepGEMM source: `~/dsv4-work/deepgemm-src/DeepGEMM` (pinned commit). Reinstall from a clean
  tree (a stray `third-party/deep_jit/` makes the version `+local`, which the pin rejects):
  `CUDA_HOME=/work/u4478697/cuda128-home pip install --no-build-isolation --no-deps .`
- Do not build in the main checkout: it holds the user's own `build/` and a py3.11 `_C.so`.
  Build in a worktree (`git worktree add --detach ~/dsv4-work/<name> HEAD`), e.g.:

  ```bash
  module load cuda/12.4
  export PATH=~/dsv4-work/venv/bin:$PATH MAX_JOBS=16 RL_KERNEL_REQUIRE_EXT=1
  python setup.py build_ext --inplace
  export CUDA_HOME=/work/u4478697/cuda128-home PATH=/work/u4478697/cuda128-home/bin:$PATH
  python -m pytest -q tests/ops/quant tests/ops/activation tests/test_deepgemm_shared_expert.py \
      tests/ops/gemm/test_deepgemm_capability.py tests/build
  python scripts/probe_dsv4_deepgemm_bi.py --output report.json  # report is gitignored
  ```

  Optional builds: `KERNEL_ALIGN_MOE_SM90=1` (P5 routed SM90 kernel),
  `KERNEL_ALIGN_DET_GEMM_SM90=1` (fast det_gemm; the BF16 fallback is very slow without it).
- Existing worktrees under `~/dsv4-work/` (base, fix, merged, aq, act) are disposable.

## Known failures that predate this work

`tests/runtime/test_attention_dispatch.py` (2, from the main merge),
`tests/test_p5_lora_delta.py` Triton cases (4), `tests/test_p1_provider.py::test_golden_manifest_anchor`.

## Numerics lessons (each one bit us)

- CUDA PyTorch `tensor / python_scalar` multiplies by the reciprocal (not correctly rounded);
  use `tensor / tensor` in goldens. CPU is exact, so CPU-only tests miss it.
- On CUDA, `torch.exp` is libdevice `expf`, same as vLLM; CPU `exp` differs in the last bit,
  so rollout-parity goldens are defined on CUDA.
- In CUDA kernels, spell FP32 steps with `__fadd_rn`/`__fmul_rn`/`div.rn` to stop FMA
  contraction; avoid `.ftz` paths; refuse `--use_fast_math`.
- vLLM's FP32-scale UE8M0 uses `exp2f(ceilf(log2f(s)))`; it equals bit-math ceiling for every
  BF16/FP16 input but not for FP32, so the quant ops reject FP32 input.
- DeepGEMM's `per_token_cast_to_fp8` clamps amax to 1e-4 (vLLM uses eps 1e-10): not a golden.
- DeepGEMM SM90 wgrad: FP32 output with accumulate only, token count % 16 == 0 (pad zeros).
- det_gemm with M=0 leaves a pending CUDA error that a later, unrelated call reports.
- The P5 oracle's unweighted shared-expert SwiGLU has no clamp and one FP32 rounding; the
  rollout clamps at 10 and rounds twice from BF16.
- Timing: use CUDA-graph replay for small shapes; eager timing measures host overhead.
