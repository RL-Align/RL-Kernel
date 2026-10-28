# Qwen-Image RoPE: full native package qualification on WSL2

Recorded 2026-10-28 for [PR #623](https://github.com/RL-Align/RL-Kernel/pull/623).
Numerical/native sources are based on `ac17ab87e8d93dabe93dfdc134e9853d47de7af1`;
this follow-up adds the reproducible gate, CPU CI routing and package provenance.
Source fingerprints are in `manifest.json`. Historical Windows measurements stay
in the separate `qwenimage_multi_axis_rope_v2_sm120` directory.

## Environment and build

| Item | Recorded value |
| --- | --- |
| Linux distribution | Ubuntu 24.04.5 LTS, x86_64, glibc 2.39 |
| Virtualization | WSL 3.0.3 on Windows 11 |
| Linux kernel | 6.18.54.1-microsoft-standard-WSL2 |
| GPU / driver | RTX 5080, SM120, 16 GB / Windows driver 617.42 |
| Python / PyTorch / Triton | 3.12.3 / 2.9.1+cu128 / 3.5.1 |
| CUDA compiler / host compiler | CUDA 12.8.93 / GCC 13.3.0 |
| Compute Sanitizer | 2025.1.0.0, CUDA 12.8 |
| Build profile | Ordinary complete `rl_engine._C`; 10 source translation units |
| CUDA architecture / parallelism | `TORCH_CUDA_ARCH_LIST=12.0`, `MAX_JOBS=2` |

CUDA compiler, development libraries and command-line tools were installed from
NVIDIA's WSL-Ubuntu repository; the existing Windows GPU driver provides CUDA
driver access. The WSL driver directory `/usr/lib/wsl/lib` was added to
`LIBRARY_PATH` for linking the package's existing CUDA driver API calls.
`RL_KERNEL_ROPE_ONLY`, forced SM90 and fast-math flags were unset.

The normal PEP 517 wheel build compiled all ten configured translation units,
including the existing logprob/GEMM/norm/activation/attention/collective code and
the new RoPE source. The complete `_C` loaded and registered the native RoPE
dispatcher. `full-extension.json` records sources, flags, binary path and hash;
`package.json` records the built wheel and installed library fingerprints.

The checkout, virtual environments, build outputs and caches are inside the
Linux filesystem, backed by the D: virtual disk. A fresh Git clone preserved all
27 tracked symlinks. This is a WSL2 Linux userspace/compiler qualification on
SM120; bare-metal Linux and A100/H100 remain separate hardware profiles.

## Results

| Check | Result | Evidence |
| --- | --- | --- |
| Ordinary complete `_C` wheel and in-place build/import | Passed | `full-extension.json`, `full-wheel-build.txt`, `package.json` |
| Operator accuracy/invariance/autograd/graphs/opcheck/fullgraph | 81 passed, 1 two-GPU guard skip | `tests-and-ws1.txt`, `operator-tests.xml` |
| CUDA and Triton WS1 four judgments | Both passed with explicit provenance | `ws1-cuda.json`, `ws1-triton.json` |
| Standard operator CLI forward/input gradient | Both passed | `operator-cuda.json`, `operator-triton.json` |
| Related regressions | 195 passed, 7 Ascend skips | `regression.txt` |
| Repository layout with extension enforcement enabled | 37 passed, zero skips/failures | `layout.txt` |
| Fresh installed full wheel, outside source checkout | Eager/compiled forward and input-gradient bytes exact for both backends | `installed-wheel.json` |
| Actual CPU-only PyTorch wheel/reference installation | 57 passed, 52 expected GPU skips; reference forward/backward passed | `cpu-contracts.txt`, `cpu-environment.json` |
| Compute Sanitizer accuracy/stride/empty/stream subset | 34 passed; zero errors | `memcheck.txt` |
| Pinned actual Diffusers symbols | 30 forward/backward cases passed; image/text tables exact | `upstream.json` |
| Reference-grid benchmark correctness | 40 comparisons passed | `benchmark-bfloat16.json` |
| Shell/workflow syntax, changed-file checks, strict docs | Passed locally | `quality.txt` |

The full-extension GPU gate rejects missing native/Triton execution. Its only
allowed operator-test skip is the two-device guard on a one-GPU runner.
The CPU-only result is separate: it uses `torch==2.9.1+cpu`, and its installed
package contains no native CUDA binary. CPU skips are not GPU acceptance evidence.
The 37 passing Linux layout tests contrast with the five Windows baseline
failures recorded earlier; those historical logs were not rewritten.

The installed-wheel probe starts in a fresh process outside the checkout,
loads `_C` from site-packages and reports that exact path. It tests packed-QKV
forward/input-gradient equality in eager and `torch.compile(fullgraph=True)`
for both CUDA and Triton. The FlashAttention BSD license is present in the wheel.

## Standalone benchmark

BF16, batch 1, 24 heads, head dimension 128. CUDA Graph medians below are
**forward / forward+backward milliseconds per call**:

| Grid / layout | Native CUDA | Adapted Triton | FlashAttention, no FMA | Diffusers complex |
| --- | --- | --- | --- | --- |
| 64x64 contiguous | 0.05780 / 0.12967 | 0.04929 / 0.12414 | 0.04898 / 0.12407 | 0.24446 / 0.51281 |
| 64x64 packed QKV | 0.05868 / 0.13079 | 0.04731 / 0.11841 | 0.04722 / 0.11847 | 0.24754 / 0.51583 |
| 83x83 contiguous | 0.11198 / 0.22392 | 0.10720 / 0.22538 | 0.10739 / 0.21420 | 0.52579 / 1.06721 |
| 83x83 packed QKV | 0.11233 / 0.22486 | 0.10745 / 0.21478 | 0.10744 / 0.21469 | 0.52820 / 1.06947 |
| 104x58 contiguous | 0.09797 / 0.19708 | 0.09366 / 0.18497 | 0.09362 / 0.18458 | 0.44992 / 0.90110 |
| 104x58 packed QKV | 0.09610 / 0.19612 | 0.09310 / 0.18828 | 0.09323 / 0.18817 | 0.44979 / 0.90276 |

These are 20 calls per graph, five warm calls/replays and 30 replay measurements.
Three synchronized eager-wall trials of 30 calls are also in the JSON. Tables,
compilation and input/backing allocations are outside timing. Extra peak includes
the returned output or gradient. The short synthetic case is included in raw data.
At 4096 tokens, both GPU paths' forward extra peak is 24 MiB for either layout,
the returned BF16 output size. Packed inputs are read without materialization.

Original FlashAttention and its autograd wrapper are checksum-pinned to
`98c1cee0ed59307833841cdbf567fe336ceec13a`. The no-FMA baseline changes only the
launch contraction option. CUDA, adapted Triton and that baseline have zero
differing forward/input-gradient bytes against the unfused FP32 gold in every
benchmark case. Default FlashAttention and complex arithmetic use declared tolerances.
The adapted Triton path is close to the mature baseline; no performance advantage
over FlashAttention is established. Timings are one-process measurements on a
shared Windows/WSL GPU without clock pinning, not a model-throughput result or
a controlled comparison against the earlier Windows process.

## Reproduction and CI

Use the matching CUDA 12.8 toolkit and Linux PyTorch 2.9.1+cu128 in a fresh
Linux virtual environment. Other Python package versions are recorded in
`python-packages.json`. From the Linux checkout:

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
  -k "forward_backward_accuracy or noncontiguous_input_tables_and_gradient or empty_forward_backward or packed_qkv_and_independent_table_strides or native_entrypoint_checks"
```

The new [operator gate](../../../ci/scripts/run_qwenimage_rope_ws1.sh) requires
the complete `_C` registration, records actual library/backend identity and
enforces the GPU skip policy. The [CPU workflow](../../../.github/workflows/qwenimage-rope.yml)
targets `test-qwenimage`, runs Python 3.10/3.12 CPU contracts, checks changed files
and builds documentation. This report's actual CPU-only run used Python 3.12;
hosted CI status is tracked on the PR. GPU runners use the explicit gate.

Bare-metal Linux, A100/H100, two-GPU guards, QK-norm fusion, model/block assembly
and trajectories remain unqualified. Building the entire library does not
certify every native operator. No Dense Qwen3/distributed certification or
cross-device/version byte-equality claim is made.
