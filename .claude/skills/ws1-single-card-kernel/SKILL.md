---
name: ws1-single-card-kernel
description: Use when writing a new WS1 single-card kernel operator in this repo (rl-kernel) - PyTorch golden first, then CUDA, then ROCm, then Ascend; gtest registration; PR with exact pytest/gtest commands and results; deterministic backward when the op needs one. The Ascend section is battle-tested; CUDA/ROCm sections are placeholders.
---

# WS1 Single-Card Kernel Workflow

Follow this workflow when adding a new operator (rmsnorm / embedding / lm_head / logp /
fused linear logp / rope / silu / swiglu / attention ...). The per-platform order is
fixed, as are the registration and PR deliverable requirements.

## Global Workflow (all platforms)

1. **Write the PyTorch golden first**: `rl_engine/kernels/ops/pytorch/<area>/<op>.py`,
   the WS1 ground-truth reference — a hand-written fixed-order fp32 reference (e.g.
   the hand-written softmax for attention, per-row `torch.mv` for lm_head),
   deliberately NOT `F.scaled_dot_product_attention` / `torch.matmul` shortcuts whose
   reduction order is unspecified. Expose both `forward` (dtype path) and
   `forward_fp32` (golden path).
2. **CUDA platform** (next section — placeholder for now).
3. **ROCm platform** (next section — placeholder for now).
4. **Ascend platform** (see the Ascend section) — **only after CUDA is done**: the
   Ascend kernel mirrors the CUDA deterministic kernel's reduction contract (e.g.
   contract v1 in `csrc/cuda/fused_linear_logp_sm90.cu`).
5. **Register in gtest**: add a platform entry (e.g. `"ascend"`) to the op's
   `candidate_paths` in `rl_engine/kernels/gtest/operator_specs.py`. Registration
   itself is the CI gate (`tests/test_ws1_gtest_gpu.py` checks every WS1 op is in the
   spec).
6. **PR must report exact commands and results**: give the actual pytest command
   line, the gtest command line (`scripts/check_operator.py` with full arguments),
   and the outputs (template below).
7. **Backward must also be deterministic**: whenever the op needs a backward, the
   backward must be a deterministic implementation (rules in the Ascend section).

## CUDA

(Placeholder — to be filled in.)

## ROCm

(Placeholder — to be filled in.)

## Ascend (battle-tested workflow)

### Branch and PR conventions

- Branch off `upstream/test` (NOT `main`): `git checkout -b feat/ascend-deterministic-<op> upstream/test`.
- PR base = `test`; title format: `[WS1][Ascend] [Qwen3-8b] <Op> ops` (e.g.
  `[WS1][Ascend] [Qwen3-8b] Fused logp ops`).
- Pushing: authenticate `gh` (`gh auth login --with-token`, then
  `gh auth setup-git`); if direct github.com connectivity is flaky, push through
  the ghfast proxy (token as the proxy host's userinfo). `gh pr edit --base` hits a
  GraphQL classic-projects deprecation error — use
  `gh api -X PATCH repos/RL-Align/RL-Kernel/pulls/<n> -f base=test` instead.
- PR description drafts: keep a scratch directory OUTSIDE the repo for
  `PR_DESCRIPTION_*.md` drafts (template below).

### Implementation checklist (file level)

The complete landing list for a new Ascend op:

1. `csrc/ascend/<op>_ascend.asc` — Ascend C kernel + torch host wrapper. No
   `PYBIND11_MODULE` (consolidated in npu_module.cpp).
2. `csrc/ascend/npu_module.cpp` — the single pybind entry declaring and binding all
   ops. Each `.asc` carrying its own `PYBIND11_MODULE` causes duplicate
   `PyInit__C_npu` link errors; for an existing `.asc` (e.g. batch_invariant_logp)
   just drop its `PYBIND11_MODULE` block.
3. `setup.py` — port the Ascend extension build (bisheng, `**/*.asc` glob,
   `_find_ascend_home()` exporting `ASCEND_HOME_PATH`/`ASCEND_TOOLKIT_HOME`).
   Fastest: `git checkout <recent-ascend-branch> -- setup.py scripts/check_operator.py`.
4. `rl_engine/_C_npu.pyi` — type stub (black: no blank line between two top-level
   defs).
5. `rl_engine/kernels/ops/ascend/<area>/<op>.py` — the op wrapper (mirror the CUDA
   wrapper's surface: `__call__`/`apply`/`forward`/`forward_fp32`, dtype gate,
   `_NPU_EXT_AVAILABLE` + `hasattr(_C_npu, ...)` check, native fallback path).
6. `rl_engine/kernels/gtest/operator_specs.py` — the `"ascend"` candidate.
7. `rl_engine/kernels/registry.py` — `ASCEND_<OP>` enum member + npu priority map
   override (`self._priority_map["npu"]["<op>"] = [ASCEND_..., PYTORCH_...]`).
8. `rl_engine/tests/test_dispatch.py` — npu priority assertion.
9. `tests/test_<op>_ascend.py` — pytest suite (PR 320 style, see below).
10. `docs/operators/<op>.md` — Ascend row in the Backends table, npu dispatch
    paragraph, Tests and Implementation Files updates (**keep existing entries**,
    add only).
11. `scripts/check_operator.py` — already supports `--device npu` (auto-detect).

Build and smoke test:

```bash
KERNEL_ALIGN_FORCE_ASCEND=1 pip install -e . --no-build-isolation
```

### Bitwise-consistency rules (mandatory; must be stated clearly in the PR)

Classify the op BEFORE writing the PR:

- **Copy/lookup ops (elementwise, e.g. embedding)**: the forward is a pure byte
  move, so it MUST be bitwise-identical to the PyTorch golden — assert with
  `torch.equal`.
- **Reduction ops (reduction / logprob, e.g. lm_head, logp, fused linear logp)**:
  **no independent kernel can be bitwise-identical to the golden** — the golden's
  reduction order is the private implementation of
  `torch.mv`/`torch.matmul`/`logsumexp`, fp32 addition is not associative, and two
  different reduction trees over D=4096 inevitably drift ~1e-4 (the logprob
  contract's fp32 atol=1e-5 is naturally unmeetable). Practice:
  - The bitwise guarantee goes to **batch invariance on the NPU**: the same row
    content across batch 1 vs {2,4,16,300}, different positions, strided blocks
    (>MAX_BLOCKS), multi-tile shapes, repeated runs — all asserted with
    `torch.equal`.
  - Compare against the golden at the existing contract tolerances; state
    prominently in a blockquote at the top of the PR body WHY bitwise parity is
    impossible (golden's private reduction order + measured drift numbers).
- **Never touch tolerances**: `rl_engine/kernels/gtest/tolerance_contract.json` is
  read-only; look up rows by op_class x dtype.

Known NPU-side golden gotchas (check before writing tests):
- NPU `torch.mv` **rejects bf16** → golden references must go through the
  `forward_fp32` paths.
- The gtest `linear_logp` forward comparison is unwinnable even for the CUDA
  candidate (the golden's `apply()` accumulates the matmul in the input dtype); CI
  never executes that candidate, it only checks registration. Do not try to adjust
  tolerances for it.
- `torch.argsort(int64, stable=True)` on NPU runs on the AiCpu — a performance
  warning only, results are correct.

### Ascend C kernel gotchas (each one hit on real hardware)

- **Cross-pipe race on shared UB buffers**: when one UB tile is written by MTE2 and
  read by MTE3, use the canonical two-queue GM→UB→GM pipeline; the fixed out-queue
  order is `AllocTensor → EnQue → DeQue → DataCopy → FreeTensor` (the queues
  provide the MTE2→V / V→MTE3 sync). Do NOT hand-roll `MTE2_MTE3`/`MTE3_MTE2`
  flags (random data corruption or hangs).
- **Vector ops need 32B-aligned counts**: UB→UB `DataCopy`, `Cast`, etc. report
  "VEC supports illegal configurations" for small counts → round the count up to a
  multiple of `32/sizeof(T)` (over-copy inside UB is harmless; the copy-out writes
  only the real byte count to GM).
- **GM scalar reads/writes are unreliable**: `GlobalTensor.GetValue/SetValue` has
  hardware issues — always read through a 32B `DataCopyPad` window (int64 window =
  4 per 32B, fp32 = 8 per 32B), sync with an `MTE2_S` flag before `GetValue`.
- **`SyncAll` deadlocks**: with more blocks launched than physical cores the
  cross-core barrier deadlocks — only per-pipe `SetFlag`/`WaitFlag` (V_S, S_V,
  S_MTE3, MTE3_S, ...).
- **Strided rows across blocks**: `MAX_BLOCKS=128`,
  `for (row = GetBlockIdx(); row < N; row += GetBlockNum())`, host side
  `blockNum = min(N, MAX_BLOCKS)` — each row is processed end-to-end by one block,
  so the instruction sequence depends only on the shape, never on batch layout or
  block assignment (the foundation of batch invariance).
- **Scalar math in the kernel**: the scalar unit has no exp/log → use a padded
  8-element vector `Exp`/`Log` (`SetValue → S_V flag → vector op → V_S wait →
  GetValue`).
- **Output staging**: `SetValue` into a UB scalar buffer, `S_MTE3` flag, then
  `DataCopyPad` out to GM; drain with `MTE3_S` after each row so the next row does
  not overwrite the staging area.
- **fp16/bf16 output cast**: `Cast(..., RoundMode::CAST_RINT, 32/sizeof(T))` —
  CAST_RINT is IEEE round-to-nearest, matching CUDA's `static_cast` semantics.
- **bisheng build**: needs `ASCEND_HOME_PATH` (setup.py exports it automatically);
  when pip swallows the real compiler error, compile the `.asc` manually with
  `bisheng` to see it.
- **const pointers**: kernel-launch GM_ADDR parameters take `uint8_t*` (non-const).

### Backward determinism

Per PR #299 (frank-2077, FFN deterministic backward): **the backward is assembled
from existing deterministic forward kernels — no new reductions, no fallback to
cuBLAS/torch.matmul**. Priority order:

1. **Reuse a pure-PyTorch deterministic formula**: when the CUDA op's backward is
   itself pure PyTorch (e.g. embedding's sorted-segment dweight: stable argsort +
   unique_consecutive + fixed-order accumulation), the Ascend op reuses the exact
   same function → bitwise-identical backward.
2. **Row-local fp32 VJP formulas** (logp / linear_logp / lm_head): compute the VJP
   in fp32 with torch ops, cast back to the input dtype at the end; no cross-row
   reduction → batch-layout independent.
3. **GEMM-shaped backward**: assemble with `det_gemm` forwards
   (`grad_hidden = det_gemm(grad, W)`, `grad_weight = det_gemm(grad^T, H)`); the
   wrapper must raise when the det_gemm symbols are missing instead of silently
   falling back.
4. **TP scenarios**: mind the shard semantics (PR #299 checklist: gate/up input
   grads each take one AllReduce, weight grads stay column-parallel shards, down is
   row-parallel, etc.).
5. Low-precision gradient comparisons: when both implementations compute the VJP in
   fp32 and quantize at the end, compare against the quantization-aligned
   reference (`ref_grad.to(dtype)`) — can be bitwise equal; tolerances only absorb
   the rare 1-ULP straddle.

### pytest suite conventions (PR 320 style)

`tests/test_<op>_ascend.py` structure:

- Module docstring stating the two orthogonal properties (correctness + batch
  invariance).
- `_npu_available()` / `_ascend_kernel_available()` helpers +
  `requires_ascend = pytest.mark.skipif(...)`.
- `TestAscend<Op>Correctness`: class-level
  `@pytest.mark.parametrize("dtype", [fp32, bf16, fp16])`; forward vs golden
  (bitwise for copy ops / contract tolerance for reductions), `forward_fp32`,
  out-of-range targets, backward, bias (if any).
- `TestAscend<Op>BatchInvariance`: bitwise (`torch.equal`).
- `TestAscendRegistryDispatch`: `kernel_registry.get_op("<op>", device="npu")`
  `type(op).__name__` assertion.

Test bugs already hit (check before writing new tests):
- Under class-level `parametrize`, every method must take the `dtype` argument —
  move tests that don't into their own class.
- Batch-comparison tests must **reuse the same weight** (regenerating with the same
  seed produces a different weight for different batch sizes).
- Position-invariance tests: pin the same row content
  (`logits[pos].copy_(base)` + `target[pos] = base_id`).
- Row-local VJP bitwise assertions: align the `grad_out` rows too
  (`grad_out[1] = grad_out[0]`).

### PR description template

Structure (mirror the wording of previous Ascend PR descriptions):

```markdown
## Latest Status [date]
Ready for review.

## Summary
- Bitwise-consistency status (prominent blockquote — mandatory for reduction ops)
- Forward kernel design (which CUDA kernel/contract it mirrors)
- Wrapper / Backward / Registration / Build

## Files (table, one row per file with Status)

## Test
# The exact commands that were run:
export KERNEL_ALIGN_FORCE_ASCEND=1
pip install -e . --no-build-isolation
python scripts/check_operator.py --op <op> --candidate ascend --device npu \
    --dtype {fp32,bf16,fp16} --batch 2 --seq 16 --vocab 257 --normalized-dim 4096 --check-grad
python -m pytest tests/test_<op>_ascend.py -v
python -m pytest tests/test_batch_invariant_logp.py -q    # regression
python -m pytest rl_engine/tests/test_dispatch.py -q      # regression

## Test results (environment line + results table + <details> folded raw output)

## Notes
```

Must include: the actual test environment (NPU model, CANN version, torch +
torch_npu versions), per-dtype gtest output and pytest results,
bitwise-invariance conclusions, regression results, pre-commit status.
