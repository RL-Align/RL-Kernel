# Qwen-Image multi-axis RoPE: SM120 validation

Initial runs recorded 2026-10-28; self-review updated 2026-11-02 for
[PR #623](https://github.com/RL-Align/RL-Kernel/pull/623).
The operator was introduced in `ac17ab87e8d93dabe93dfdc134e9853d47de7af1`.
The [operator documentation](../../docs/operators/multi-axis-rope.md) defines the
contract, upstream reuse, API and detailed reproduction commands.

## Environments

Both initial profiles use one RTX 5080 (SM120, 16 GB), Windows driver 617.42,
PyTorch 2.9.1+cu128 and CUDA Toolkit 12.8. They are separate recorded runs,
not a controlled Windows-versus-WSL performance comparison.

| Item | WSL2 Linux | Windows |
| --- | --- | --- |
| OS | Ubuntu 24.04.5, glibc 2.39; WSL 3.0.3 on Windows 11 | Windows |
| Kernel | 6.18.54.1-microsoft-standard-WSL2 | Native Windows |
| Python | 3.12.3 | 3.13.9 |
| Compiler | CUDA 12.8.93, GCC 13.3 | CUDA 12.8, MSVC 2022 x64 |
| Triton | 3.5.1 | triton-windows 3.5.1.post24 |
| Native build | Ordinary complete `rl_engine._C`, all 10 configured translation units | Explicit operator-only `rl_engine._rope_cuda` |

The WSL checkout, environments and caches are in the Linux filesystem.
The normal full wheel was built with `RL_KERNEL_ROPE_ONLY`, forced SM90 and
fast-math flags unset. Linking existing CUDA driver calls used
`LIBRARY_PATH=/usr/lib/wsl/lib`; the Windows driver supplies CUDA driver access.

## Initial qualification

| Check | WSL2 Linux | Windows |
| --- | --- | --- |
| Public operator suite | 81 passed, 1 two-GPU guard skip | 81 passed, 1 two-GPU guard skip |
| Related regressions | 195 passed, 7 Ascend skips | 195 passed, 7 Ascend skips |
| CUDA/Triton WS1 forward/gradient accuracy and invariance | Both passed with backend provenance | Both passed with backend provenance |
| Standard operator CLI forward/input gradient | Both backends passed | Both backends passed |
| Pinned actual Diffusers symbols | 30 cases passed; image/text tables exact | 30 cases passed; image/text tables exact |
| Compute Sanitizer | 34 selected cases passed, zero errors | 34 selected cases passed, zero errors |
| Fresh installed wheel outside checkout | Full wheel: both backends' eager/compiled forward and input-gradient bytes exact | Operator wheel: same checks passed |
| CPU-only PyTorch 2.9.1+cpu, Python 3.10.22 and 3.12.3 | Each: 57 passed, 52 expected GPU skips; installed reference forward/backward passed | Not recorded |
| Repository layout | 37 passed, zero skips/failures | 31 passed, 5 failures, 1 skip |
| Changed-file static checks and strict docs | Passed locally | Passed locally |

The operator suite covers FP32/FP16/BF16, independent accuracy, byte invariance,
packed-QKV/no-copy and table strides, empty inputs, higher input VJPs, stream
selection, graph replay, opcheck and fullgraph forward/backward. The installed
wheels contain the FlashAttention BSD license; the CPU packages contain no CUDA
binary. Local CPU checks are separate from hosted CI and GPU acceptance.

The five historical Windows layout failures also reproduce on its clean base
(27 passed, 5 failures, 1 skip): driver linking and Windows symlink privileges or
placeholders. The later Linux run preserved all 27 tracked symlinks and passed.
The earlier Windows logs remain unchanged in the archive.

## Automated-review follow-up

The standard CLI and WS1 adapters now share a centered image-grid prefix,
selected by full logical sample length and token identity. Eleven CPU regression
cases cover empty/square/nonsquare lengths and packed/permuted/chunked/padded
layouts. Before the change, ten of those cases failed; all eleven now pass.
Additional tests pin non-CUDA reference routing, library-load error propagation,
shared-table broadcasting, and head-dimension stride 2 or outermost storage.
Native shape checks and indexing share named dimension constants; the rotation
arithmetic is unchanged. Backward telemetry documents its graph-replay limits.

The following are new local WSL2 runs of the updated implementation and fixtures:

| Check | Result |
| --- | --- |
| Rebuilt ordinary complete CUDA wheel; fresh installed-package eager/fullgraph forward/input gradient | Passed for both GPU backends |
| Public operator suite | 105 passed, 1 two-GPU guard skip |
| CUDA/Triton WS1 accuracy/invariance and ordinary CLI checks | Both passed with explicit backend provenance |
| Related regressions, including the eleven new input cases | 206 passed, 7 Ascend skips |
| Repository layout with native extension enforcement | 37 passed |
| CPU-only PyTorch 2.9.1+cpu, Python 3.10.22 and 3.12.3 | Each: 80 passed, 64 expected GPU skips; installed reference forward/backward passed |
| Expanded Compute Sanitizer subset, including the new stride cases | 52 passed, zero errors |
| Changed-file checks and strict docs | Passed locally |

The full wheel contains the shared validation helper and the FlashAttention BSD
license. Wheel SHA256:
`9feb8200b948021c268c933f93efcd424b6b68d6c55bc24c3004d6336deb3a13`.
Non-CUDA routing tests exercise candidate selection with CPU tensors; they do
not certify NPU/ROCm/MUSA hardware. The benchmark below retains its initial-run
measurements; no new timing comparison was performed for this follow-up.

## 2026-11-02 self-review

The CUDA/Triton byte contract exposed a signed-zero bug in Triton conjugation:
in Triton 3.5.1, unary minus lowers to `0 - x`, so `+0` does not become `-0`.
Conjugation now flips the FP32 sign bit directly. The twelve new dtype/sine-sign
regressions went from three failures against the previous kernel to all passing.
Generic slice autograd can erase negative-zero signs when merging gradients;
the benchmark now uses the explicit transpose as its byte oracle. CUDA/Triton
retain strict byte checks; upstream baselines use numerical accuracy checks with
raw byte differences recorded. NaN payload equality is not claimed.

The next automated review exposed two native-provider bugs. Loading the full
and companion libraries in either order aborted with duplicate registration.
Registration now checks process-wide dispatcher state and reuses the first
same-revision provider. Fresh subprocesses test both raw load orders, concurrent
loading and both Python import orders. Each performs 18 byte comparisons across
FP32/FP16/BF16, forward/transpose, subnormal values and signed zeros.

A forced `--use_fast_math` build also reproduced FTZ differences: 96 output
bytes for FP32 and 96 for BF16 in the subnormal probe (FP16 had zero).
Explicit scalar PTX uses separate `.rn.f32` instructions without `.ftz`,
and transpose negation flips the sine sign bit. The same probe now has zero
differences in all three dtypes. This qualifies this translation unit under
that flag, not unrelated native operators. Dispatch instantiates only the three
supported dtypes. Twelve permanent GPU cases cover subnormal inputs/results
and input gradients. The NPU diagnostic routing and backend-specific backward
identity remain covered by the existing regression and WS1 checks.

The following self-review closed the raw native entry's autograd boundary.
Previously, a direct grad-enabled call could warn during backward and leave
the input gradient absent. It now raises with the public API entry point;
the Python custom op still supplies recursive VJPs, while explicit no-grad
and inference calls remain valid. Twelve dtype/transpose/empty-input cases
cover this boundary. Together with manual-workflow tests, the pre-fix run had
15 failures; the expanded targeted post-fix run passed all 24 cases.

The CPU/static workflow now targets `main`, `test` and `test-qwenimage`.
Manual static checks use an explicit `base_sha` or all tracked files, rather
than only the final commit. Tests execute the actual workflow script against
a multi-commit temporary Git repository, including spaces and leading-dash
filenames. Additional loader tests cover missing full `_C`, a single successful
load and a library becoming available later; root-cause load errors still propagate.
Fresh standalone-wheel installation also verifies both backends without `_C`.

This run uses the same WSL/Python/PyTorch/Triton profile with driver **621.01**.
The earlier driver-617.42 records above remain historical evidence.

| Check | Result |
| --- | --- |
| Ordinary complete CUDA wheel and fresh installed eager/fullgraph forward/input gradient | Passed; BSD license present |
| Operator suite and CUDA/Triton WS1 gates | 144 passed, 1 two-GPU skip; both gates passed |
| Related regressions / repository layout | 210 passed, 7 Ascend skips / 41 passed |
| CPU-only Python 3.10/3.12 | Each: 87 passed, 100 expected GPU skips; installed reference forward/backward passed |
| Expanded Compute Sanitizer subset | 88 passed, zero errors |
| Fresh standalone wheel without full `_C`, and installed raw/public autograd boundary in both wheel profiles | Passed; CUDA/Triton eager/fullgraph forward/input gradients exact |
| Full/companion coexistence, normal and forced-fast-math companion builds | Five subprocess modes per build, 180 byte comparisons passed; library fingerprints recorded |
| Additional adversarial checks | 22 groups passed: extreme/subnormal/non-finite values, signed zeros, logical negative views and dynamic fullgraph forward/backward |
| Standalone benchmark | 40 correctness comparisons passed; 80 measurements |
| Changed-file checks and strict docs | Passed |

Dynamic compilation checks isolate independent compiled functions' frame caches,
then vary five shapes within each function, including empty batch/sequence.
The full wheel SHA256 is
`7bd95ff6462f8b80f88cc10332aa6e8eddd997962dd696fd96651c349c587c2e`.

Latest BF16 [1,4096,24,128] packed-QKV graph medians, forward / forward+backward
ms/call: CUDA 0.06928 / 0.18138; Triton 0.06056 / 0.16448;
FlashAttention no-FMA 0.05498 / 0.16717; complex 0.34606 / 0.70062.
These are shared-GPU, unpinned-clock observations, not a controlled comparison
with the older driver or evidence of an advantage over FlashAttention.

## Initial standalone benchmark

WSL2 BF16, batch 1, 24 heads, D=128. CUDA Graph medians in ms/call;
each cell is **forward / forward+backward**.

| Grid / layout | Native CUDA | Adapted Triton | FlashAttention, no FMA | Diffusers complex |
| --- | --- | --- | --- | --- |
| 64x64 contiguous | 0.05780 / 0.12967 | 0.04929 / 0.12414 | 0.04898 / 0.12407 | 0.24446 / 0.51281 |
| 64x64 packed QKV | 0.05868 / 0.13079 | 0.04731 / 0.11841 | 0.04722 / 0.11847 | 0.24754 / 0.51583 |
| 83x83 contiguous | 0.11198 / 0.22392 | 0.10720 / 0.22538 | 0.10739 / 0.21420 | 0.52579 / 1.06721 |
| 83x83 packed QKV | 0.11233 / 0.22486 | 0.10745 / 0.21478 | 0.10744 / 0.21469 | 0.52820 / 1.06947 |
| 104x58 contiguous | 0.09797 / 0.19708 | 0.09366 / 0.18497 | 0.09362 / 0.18458 | 0.44992 / 0.90110 |
| 104x58 packed QKV | 0.09610 / 0.19612 | 0.09310 / 0.18828 | 0.09323 / 0.18817 | 0.44979 / 0.90276 |

All 40 pre-timing comparisons passed; CUDA, adapted Triton and the no-FMA
baseline had zero differing output/input-gradient bytes against the FP32 gold.
The archive includes 80 measurements, the short case, default FlashAttention,
eager wall timings and peak allocation. At 4096 tokens, either GPU path's forward
extra peak is 24 MiB, the returned output size, for both layouts.

Each graph contains 20 calls, with five warm calls/replays and 30 measured
replays. Eager wall timings use three synchronized trials of 30 calls.
Tables, compilation and input/backing allocation are excluded; extra peak
includes the returned output or gradient. The upstream source/autograd wrapper
is pinned to `98c1cee0ed59307833841cdbf567fe336ceec13a`; the no-FMA variant changes
only contraction. These single-process runs use a shared GPU without clock
pinning and establish no advantage over FlashAttention or model speedup.

## Reproduction

Use a matching CUDA 12.8/PyTorch environment in a Linux checkout:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 RL_KERNEL_REQUIRE_EXT=1
unset RL_KERNEL_ROPE_ONLY KERNEL_ALIGN_FORCE_SM90 KERNEL_ALIGN_USE_FAST_MATH
python -m pip wheel --no-build-isolation --no-deps . -w /tmp/rope-full-wheel
python -m pip install --force-reinstall --no-deps /tmp/rope-full-wheel/rl_kernel-*.whl
python setup.py build_ext --inplace
ROPE_PYTHON=python ROPE_WS1_OUTPUT=/tmp/rope-ws1 bash ci/scripts/run_qwenimage_rope_ws1.sh
python ci/run.py layout -q -rs
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest \
  tests/ops/rope/test_multi_axis_rope.py -q \
  -k "forward_backward_accuracy or noncontiguous_input_tables_and_gradient or arbitrary_head_dimension_stride_and_gradient or backward_preserves_signed_zero or subnormal_values_are_not_flushed or native_entrypoint_requires_autograd_wrapper or empty_forward_backward or packed_qkv_and_independent_table_strides or native_entrypoint_checks"
```

The [GPU gate](../../ci/scripts/run_qwenimage_rope_ws1.sh) requires the full
native extension and Triton, records loaded binary/backend identity, and permits
only the two-GPU guard to skip. The [CPU workflow](../../.github/workflows/qwenimage-rope.yml)
contains the Python 3.10/3.12 contract commands. The operator documentation covers
installed-wheel probes, pinned upstream audits and benchmark reproduction.
Hosted CI status is tracked on the PR.

## Archived evidence and qualification limits

Detailed per-case JSON, JUnit, build/test logs, dependency versions, and
source/binary SHA256 manifests remain available at immutable commit
`84751473006401c19c1c8c17844e8ef8b1c44375`:

- [WSL2 complete-extension evidence](https://github.com/dandan20031031/RL-Kernel/tree/84751473006401c19c1c8c17844e8ef8b1c44375/reports/experiments/qwenimage_multi_axis_rope_wsl2_sm120)
- [Historical Windows operator-profile evidence](https://github.com/dandan20031031/RL-Kernel/tree/84751473006401c19c1c8c17844e8ef8b1c44375/reports/experiments/qwenimage_multi_axis_rope_v2_sm120)

Those archives identify the initial runs; the follow-up table records subsequent
reruns. Bare-metal Linux, A100/H100, two-GPU guards,
QK-norm fusion, model/block assembly and trajectories remain unqualified.
Building the full library does not certify every native operator or Dense Qwen3.
No distributed or cross-version/device byte-equality claim is made.
