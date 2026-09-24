# AdaLN modulation (Qwen-Image WS1, #386)

`kernel_registry.get_adaln_modulation_op(device, hidden=H)` returns `(op, trace)`. Call
`modulated, gate = op(x, modulation, eps=1e-6)` with `x: [B,S,H]`
and shared `modulation: [B,3H]`. Both inputs must share device and dtype (FP32
or BF16). The chunks are **shift, scale, gate**. Each row uses LayerNorm over
H with no affine parameters, then `norm(x) * (1 + scale) + shift`. The gate is
`[B,1,H]` and remains connected to `modulation` for downstream gradients.
The Qwen-Image image/text streams and attention/MLP calls use separate inputs.
The `SiLU + Linear` projection and `adaln_gate_residual` are separate operators.

The FP32 hidden reductions use a lower-plus-upper pairwise tree padded to the
next power of two. Each sample's `dshift` and `dscale` contributions are folded
in ascending logical token order. The CUDA shared reduction reads its result
into each thread before a block barrier permits buffer reuse. No atomic partial sums, Split-K, Stream-K,
TF32, or fast math are used by these kernels. BF16 outputs are cast once after
FP32 arithmetic. The CPU reference computes square root in FP64 and rounds it
to FP32 because CPU `torch.sqrt(float32)` can differ by one ULP from CUDA's
correctly rounded square root. Hidden and token accumulators remain FP32.
Inputs with S=0 or H=0 and mixed dtype/device are rejected.
Strided model chunks are copied to contiguous storage inside
GPU wrappers. The CUDA kernel supports H<=4096; shape-aware dispatch selects
Triton for larger H. CUDA symbols are omitted from builds with
`KERNEL_ALIGN_USE_FAST_MATH=1`, making fallback explicit.

For Qwen-Image editing, pass `modulate_index` to both resolution and execution:

```python
op, trace = kernel_registry.get_adaln_modulation_op(
    x.device, hidden=x.shape[-1], modulate_index=modulate_index
)
modulated, gate = op(x, modulation, modulate_index=modulate_index)
```

This path accepts modulation `[2B,3H]` and an index `[B,S]` or `[1,S]` whose
values are 0 or 1. It uses the explicit PyTorch select01 reference and returns
gate `[B,S,H]`. The trace sets `fallback=true` and
`fallback_reason=modulate_index_requires_select01`; the indexed path does not
claim the fixed-order CUDA/Triton backward contract.

The resolution matrix `{1024², 1328², 1664×928}` implies image sequence
lengths `{4096, 6889, 6032}` for VAE scale 8 and 2x2 latent packing. H=3072.
These are derived model shapes, not tensor shapes stated by the issue.

## Validation

Run `python -m pytest tests/test_adaln_modulation.py -q` for focused tests.
Set `RLK_ADALN_REAL_SHAPES=1` to include the three full image shapes.
Run `python benchmarks/benchmark_adaln_modulation.py --real` for the three
image shapes, with `--dtype fp32` for the second dtype. The benchmark prints
median forward and forward+backward latency plus peak allocated memory;
`--warmup` and `--repeat` control the sample count. It explicitly skips the
CUDA extension if its symbols are absent. The registry trace records selected
backend, fallback, reduction order, accumulator dtype, disabled algorithm
flags, and a semantic kernel version. The semantic version is not a hash of the
compiled binary. Direct backend constructors accept validated launch settings:
CUDA `threads={128,256,512}` (default 256); Triton `num_warps={4,8}`
(default 4), `reduction_tile={64,128,256}` (default 128). Registry dispatch
uses those defaults. These settings change scheduling, not the reduction order.

Local CUDA validation environment (2026-09-23): RTX 3050 Ti (SM86, 4 GiB),
driver 610.60, CUDA toolkit/runtime 12.6, PyTorch 2.12.1+cu126, Triton 3.7.1,
Python 3.12.12. The extension was built with fast math disabled for SM86.
The pre-review focused suite plus all three real shapes passed 18 tests with no skips.
These historical results are not acceptance evidence for the synchronization fix.
FP32 small-shape and BF16 real-shape CUDA forward/backward outputs were byte
equal to the CPU reference. CUDA and Triton were also byte equal directly at
all three real shapes, and both passed FP32/BF16 batch-size and token-padding
byte-invariance checks.

Real-shape medians below use one warmup and three samples, so they are smoke
measurements rather than stable performance claims:

| dtype | S | Triton fwd / fwd+bwd (ms) | CUDA fwd / fwd+bwd (ms) |
| --- | ---: | ---: | ---: |
| BF16 | 4096 | 3.069 / 20.709 | 2.998 / 23.262 |
| BF16 | 6889 | 5.125 / 34.551 | 5.150 / 39.963 |
| BF16 | 6032 | 4.689 / 30.302 | 4.357 / 34.718 |
| FP32 | 4096 | 5.944 / 29.269 | 7.229 / 38.693 |
| FP32 | 6889 | 9.840 / 48.579 | 11.948 / 65.271 |
| FP32 | 6032 | 8.685 / 81.903 | 10.639 / 57.270 |

Forward and forward+backward peak allocation matched between CUDA and Triton
at each shape. The CUDA backward path was generally slower in this short SM86
run; rerun with the default warmup/repeat counts on target hardware before
making a performance claim.

## Review regression checks (2026-09-24)

The pre-fix current-HEAD suite passed 20 cases (including extension smoke),
but compute-sanitizer found one forward and three backward shared-memory hazards.
The missing synchronization was after reading the reduction result, before
reusing its buffer. The minimal fixed-seed H=64 regression returned exit code 7
even though its numerical assertion passed.

The non-native review selection passed 30 cases with 26 deselected using
`-k 'not cuda and not real_image and not gpu_padding'`. This does not validate
the rebuilt native extension.

Post-fix validation of the rebuilt extension passed **57 tests in 43.73s** with
no skips, including native extension smoke and the complete focused matrix.
The native registry selected `CUDA_ADALN_MODULATION` with `fallback=False`.
The expanded sanitizer regression passed **8 tests** with **0 hazards,
0 errors, and 0 warnings** (exit 0). The original seed-386 FP32
`[1,2,3072]` forward/backward reproducer also reported zero hazards after
the fix. Environment remained PyTorch 2.12.1+cu126, Triton 3.7.1, Python
3.12.12, RTX 3050 Ti SM86.

Observed commands (the sanitizer ran separately before the full pytest matrix):

```bash
bash scripts/adaln_cuda_acceptance/06_racecheck_adaln_cuda.sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
RLK_ADALN_REAL_SHAPES=1 RL_KERNEL_REQUIRE_EXT=1 \
.venv/bin/python -m pytest tests/test_extension_smoke.py \
  tests/test_adaln_modulation.py -q -rs -p no:cacheprovider
```

Local logs are `/tmp/rlk-adaln-race-fix-build.log`,
`/tmp/rlk-adaln-race-fix-racecheck.log`, and
`/tmp/rlk-adaln-race-fix-tests.log`. They describe the uncommitted review
fixes on top of `d14e4ed`; that commit alone predates the fix.

Run the combined fail-closed checks using the cu126 repository environment:

```bash
bash scripts/adaln_cuda_acceptance/04_test_adaln_cuda.sh
# Or only the sanitizer regression:
bash scripts/adaln_cuda_acceptance/06_racecheck_adaln_cuda.sh
```

Script 04 runs the full focused suite with real shapes and required native CUDA,
then script 06. Missing extension symbols or sanitizer failures must not count
as successful native acceptance. Do not use bare `uv run` or `uv sync` to
run these checks: invoke `.venv/bin/python` directly to preserve cu126.

The statuses below apply to the measured inputs and SM86 configurations, not
all shapes or hardware.

| Requirement | Implementation | Test | Observed evidence | Remaining gap | Status |
| --- | --- | --- | --- | --- | --- |
| Shared forward/backward math at H=7/3072 | Fixed FP32 normalization and VJP | Independent LayerNorm/autograd, both dtypes, random dy/dgate | All 12 backend/dtype/H cases passed | Independent full-image autograd not covered | PROVEN |
| Race-free CUDA shared reduction | Read-result/barrier/reuse | Native harness and geometry under racecheck | 8 tests, zero hazards; original reproducer also green | Other architectures not exercised | PROVEN |
| Launch geometry / tiling | CUDA threads 128/256/512; Triton 4/8 warps and 64/128/256 tiles | Public constructors, CPU-reference byte checks | Both dtypes passed all 3 CUDA and 6 Triton configurations | Coverage is H=3072, B=2, S=5 | PROVEN |
| Strict byte comparator | Contiguous uint8 views | Signed-zero regression and backend comparisons | Comparator rejects +0 versus -0; full matrix passed | Exceptional floating-point payloads not covered | PROVEN |
| Three real image sizes | S=4096/6889/6032, H=3072 | Both dtypes, random dy/dgate, CPU/native/Triton byte checks | All 6 real-shape cases passed | Full-size oracle is the fixed CPU implementation, not independent LayerNorm/autograd | PARTIAL |
| Indexed select01 | Explicit PyTorch fallback with [2B,3H] modulation | B=2, both index shapes/devices/dtypes, independent row-index and autograd gold; invalid routes | 12 indexed cases passed | No fixed-order indexed backward claim | PROVEN |
| FP32 accumulation / final BF16 cast | FP32 reductions, CUDA RN operations, Triton fusion disabled | Cast, byte-reference and independent accuracy tests | All focused checks passed | Exhaustive rounding boundaries and fast-math-on build exclusion not exercised | PARTIAL |
| Batch / padding invariance | Row-local hidden reduction, ascending token fold | Batch position, B=1/3, S=3/5, zero-gradient padding; raw bytes | FP32/BF16 CUDA and Triton cases passed | Tested padding matrix uses H=32 | PROVEN |
| Registry / trace | Explicit backend selection, H limit, indexed fallback reason | Registry tests and native preflight | Native selected without fallback; indexed fallback and H limit passed | Semantic fingerprint is not a binary hash; unavailable-backend combinations are not exhaustive | PARTIAL |
| Native SM86 build and binding | Conditional sources/symbols, defaulted threads argument | Symbol verification, native/reference tests and geometry launches | User build completed; new four-argument binding and full suite passed | Other build configurations untested | PROVEN |
| Indexed optimized kernels | Agreed explicit PyTorch fallback | Fallback and trace checks | Within agreed scope | No optimized indexed kernel required | OUT OF SCOPE |
| Full #386 WS1 exit criteria | All 19 operators and assembled model | MMDiT/LoRA, 60-layer curve, reproducible rollout logp | Not exercised by this operator suite | Separate system-level work | UNPROVEN |


The independent autograd checks are accuracy checks, not a replacement tolerance
profile for the fixed-reference byte contract. Shared checks use FP32
`atol=rtol=2e-5`, BF16 `atol=0.02, rtol=0.016`; indexed checks use
`atol=rtol=2e-6`. The full-size byte checks still use the fixed CPU reference
and do not independently validate LayerNorm/autograd at every image size.

The semantic fingerprint does not identify a compiled binary. This suite does
not claim coverage across other GPU architectures, every floating-point input,
or fast-math-enabled builds. Benchmark numbers above predate the synchronization
fix; they are historical smoke measurements, not performance claims for the fix.
