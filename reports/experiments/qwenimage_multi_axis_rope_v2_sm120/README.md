# Qwen-Image multi-axis RoPE validation — 2026-10-28

Evidence for the CUDA `multi_axis_rope` row of
[RFC #386](https://github.com/RL-Align/RL-Kernel/issues/386), based on
`test-qwenimage@16b81175874ecf11d5ff05c00de3b69e9457591b`.
The source fingerprints in `manifest.json` identify the final implementation.
This is single-GPU operator evidence; it does not qualify a model or trajectory.

## Implementation

- Adjacent pairs, axes `[16, 56, 56]`, canonical centered image positions and
  diagonal text offsets, pinned to Diffusers `40f63690da440e8c382ab53e67508835deb1ed2f`.
- Explicit CPU table factories remain CPU factories inside CUDA/meta default-device contexts.
- Native CUDA reads all input/table strides directly, including packed QKV views.
  The shared Triton implementation adapts FlashAttention's existing rotary kernel
  at `98c1cee0ed59307833841cdbf567fe336ceec13a`, retaining attribution and BSD-3 terms.
- Both GPU paths have dispatcher, fake and input-autograd registration. The
  arithmetic is separate FP32 multiply/add followed by one output cast.
- OP_SPECS, the ordinary input builder and a gradient adapter expose the operator
  to existing validation APIs. The operator gate runs all four forward/gradient
  accuracy/invariance judgments with explicit backend/kernel provenance.
- Default CUDA builds include the native source in `_C`. The explicit
  `RL_KERNEL_ROPE_ONLY=1` build profile also produces an installable companion
  library, without compiling unrelated native operators.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080, SM120, 16 GB |
| Driver | 617.42 |
| OS / Python | Windows / 3.13.9 |
| PyTorch / CUDA toolkit | 2.9.1+cu128 / 12.8 |
| Triton | triton-windows 3.5.1.post24; runtime 3.5.1 |
| Native compiler | MSVC 2022 x64, C++17; CUDA target 12.0 |
| Sanitizer | NVIDIA Compute Sanitizer 13.3, memcheck |
| Benchmark dtype | BF16, FP32 position tables |

The installed wheel is CPython 3.13/Windows-specific. Toolkit/driver versions
have different roles; the native code was compiled with CUDA 12.8, not 13.3.

## Validation

| Check | Result | Evidence |
| --- | --- | --- |
| Public operator suite using the built companion library | 81 passed, 1 two-GPU skip | `tests.txt` |
| Operator/build/registry/validation regressions | 195 passed, 7 Ascend skips | `regression.txt` |
| CUDA WS1 four judgments | Passed | `ws1-cuda.json` |
| Triton WS1 four judgments | Passed | `ws1-triton.json` |
| Standard CLI, BF16 forward and input gradient, CUDA/Triton | Both passed | `operator-{cuda,triton}.json` |
| Pinned actual Diffusers symbols | 30 cases passed; image/text tables exact | `upstream.json` |
| Memory safety on selected accuracy/strided/empty/stream cases | 34 passed, zero memcheck errors | `memcheck.txt` |
| Source distribution + operator wheel build | Passed; native sources and BSD license present | `package-build.txt`, `package.json` |
| Fresh interpreter outside checkout after wheel installation | CUDA/Triton eager and fullgraph forward/input-gradient bytes exact | `installed-wheel.json` |
| Changed Python formatting/static checks; strict docs build | Passed | `quality.txt` |
| Repository layout check | 5 failures, 31 passed, 1 skipped | `layout.txt` |
| Same layout check on clean base | Same 5 failures, 27 passed, 1 skipped | `baseline-layout.txt` |

Operator tests cover FP32/FP16/BF16, shared/per-sample tables, odd and rectangular
grids, batch/chunk/padding invariance, higher input VJPs, strided gradients,
independent table strides, empty tensors, invalid-input rejection, non-default
streams, graph replay with changed static buffers, `torch.library.opcheck`, and
`torch.compile(..., fullgraph=True)` through backward. A CPU profiler assertion
rejects `contiguous`, `clone` or `copy_` inside either GPU call for a packed view.

The layout failures reproduce on the clean base: the Windows driver-link flag
assertion, symlink creation without Windows privilege, and three checks affected
by symlink-disabled JSON/source placeholders. The added build tests pass. This
report does not describe the entire repository test suite as green.

## Mature baseline and performance

`benchmark-bfloat16.json` contains 40 pre-timing correctness comparisons and 80
measurements: four geometries, two layouts, five implementations, two phases.
The CUDA, adapted Triton and contraction-disabled upstream FlashAttention paths
all have zero differing output/gradient bytes against the unfused FP32 gold.
Original FlashAttention and Diffusers complex arithmetic use declared tolerances.

The optional FlashAttention comparison loads the checksum-pinned original
`rotary_kernel`, `apply_rotary` and `ApplyRotaryEmb` class without installing
flash-attn. `flash_no_fma` adds only `enable_fp_fusion=False` to the original
launch; the default baseline retains its fused arithmetic.

CUDA Graph medians in ms/call; each cell is **forward / forward+backward**.
Batch 1, 24 heads, D=128:

| Grid / layout | Native CUDA | Adapted Triton | FlashAttention, no FMA | Diffusers complex |
| --- | --- | --- | --- | --- |
| 64×64 contiguous | 0.05771 / 0.12953 | 0.05191 / 0.12463 | 0.05202 / 0.12443 | 0.24277 / 0.50118 |
| 64×64 packed QKV | 0.05866 / 0.13089 | 0.04835 / 0.12314 | 0.04896 / 0.12374 | 0.24599 / 0.50083 |
| 83×83 contiguous | 0.11203 / 0.22478 | 0.10685 / 0.21339 | 0.10693 / 0.21351 | 0.51009 / 1.05054 |
| 83×83 packed QKV | 0.11280 / 0.22514 | 0.10758 / 0.21415 | 0.10746 / 0.21431 | 0.51117 / 1.05461 |
| 104×58 contiguous | 0.09809 / 0.19646 | 0.09356 / 0.18708 | 0.09362 / 0.18709 | 0.43411 / 0.89356 |
| 104×58 packed QKV | 0.09596 / 0.19664 | 0.09341 / 0.18776 | 0.09344 / 0.18786 | 0.43668 / 0.89561 |

The adapted Triton path is close to its mature upstream baseline. The native
CUDA path avoids requiring Triton but is slower in this graph comparison.
No speed advantage over FlashAttention is established. At 4096 tokens, either
GPU path's forward extra peak is 24 MiB, exactly the returned BF16 output size,
for both layouts. There is no extra packed-input materialization.

Five warm calls/replays, 20 calls per captured graph, median of 30 replay
measurements. Synchronized eager wall time is also recorded as three trials of
30 calls. It includes Python/dispatcher/autograd overhead and can differ
substantially from graph time. Tables, compilation, input allocation and packed
backing storage are outside timing. Extra peak allocation includes the returned
output or gradient. These are one-machine, one-process microbenchmarks, without
clock pinning; close differences do not establish a stable winner.

## Reproduction

Set a toolkit matching PyTorch, the target architecture and a matching host
compiler. On the recorded Windows host, run from an x64 MSVC environment with
`CUDA_HOME`/`CUDA_PATH` selecting CUDA 12.8 and `TORCH_CUDA_ARCH_LIST=12.0`.
Use `$env:RL_KERNEL_ROPE_ONLY='1'` for the operator package profile.

```bash
python setup.py sdist bdist_wheel
python setup.py build_ext --inplace
python -m pytest tests/ops/rope/test_multi_axis_rope.py -q -rs
python tools/validation/operators/check_operator.py --op multi_axis_rope \
  --candidate cuda --device cuda --dtype bf16 --batch 2 --seq 7 --n-heads 3 --check-grad --json
python tools/validation/operators/validate_multi_axis_rope.py --ws1 cuda --output ws1-cuda.json
python tools/validation/operators/validate_multi_axis_rope.py --ws1 triton --output ws1-triton.json
python tools/validation/operators/validate_multi_axis_rope.py --benchmark \
  --flash-source /path/to/flash_attn/ops/triton/rotary.py \
  --flash-layer-source /path/to/flash_attn/layers/rotary.py --output benchmark.json
python tools/validation/operators/check_qwenimage_rope_upstream.py \
  --source /path/to/transformer_qwenimage.py --output upstream.json
compute-sanitizer --tool memcheck --error-exitcode 1 python \
  tools/validation/operators/validate_multi_axis_rope.py -q \
  -k "forward_backward_accuracy or noncontiguous_input_tables_and_gradient or empty_forward_backward or packed_qkv_and_independent_table_strides or native_entrypoint_checks"
mkdocs build --strict -f mkdocs.yaml
```

Install the built wheel, change to a directory outside the checkout, then run
`check_multi_axis_rope_install.py --package-root /path/to/site-packages --output installed-wheel.json`.
This probe neither changes sys.path nor invokes the focused JIT loader.
See `quality.txt` for the regression command and static-check scope.

## Related work and remaining qualification

The [operator page](../../../docs/operators/multi-axis-rope.md) documents
FlashAttention reuse and the related Ascend #411, SGLang and vLLM-Omni paths.
The live repository search on 2026-10-28 found #411 for the exact row name;
its head remains `285c4dccab7e9048956584caf431946d82405c09`. Its half-split and
uncentered reference differs from the pinned adjacent-pair/centered contract.

Linux, A100/H100, multi-GPU guards, the complete monolithic `_C` build, QK-norm
fusion, model/block assembly and trajectories remain unqualified. There is no
Dense Qwen3 or distributed certification implied by the operator WS1 adapter.
Cross-version/device byte equality and fast-math builds are not claimed.
