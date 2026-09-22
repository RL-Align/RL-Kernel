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
in ascending logical token order. No atomic partial sums, Split-K, Stream-K,
TF32, or fast math are used by these kernels. BF16 outputs are cast once after
FP32 arithmetic. The CPU reference computes square root in FP64 and rounds it
to FP32 because CPU `torch.sqrt(float32)` can differ by one ULP from CUDA's
correctly rounded square root. Hidden and token accumulators remain FP32.
Inputs with S=0 or H=0 and mixed dtype/device are rejected.
Strided model chunks are copied to contiguous storage inside
GPU wrappers. The CUDA kernel supports H<=4096; shape-aware dispatch selects
Triton for larger H. CUDA symbols are omitted from builds with
`KERNEL_ALIGN_USE_FAST_MATH=1`, making fallback explicit. Indexed
`modulate_index` is outside this shared-parameter
API pending the maintainer's answer on #386.

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
compiled binary.

Local smoke environment (2026-09-22): RTX 3050 Ti (SM86, 4 GiB), driver
610.60, PyTorch 2.12.1+cu126, Triton 3.7.1, Python 3.12.12. The focused suite
passed 12 tests with 4 skips (three opt-in real-shape cases and the unavailable
CUDA extension). The three real-shape Triton BF16 forward/backward tests passed
with byte equality against the CPU reference when enabled. A smoke benchmark
at B=1, S=64, H=3072, BF16, 2 warmups and 3
repeats measured medians (ms): PyTorch 6.723 forward / 26.161 forward+backward;
Triton 0.364 / 2.631. The CUDA extension was explicitly skipped because no
matching `nvcc` is installed. These numbers are a script smoke check, not the
required full resolution benchmark.

**Unconfirmed acceptance:** #386 requires CPU-to-CUDA byte equality and CUDA
as the bit-level comparison backend. Existing repository tests for other ops
use a tolerance for CPU-to-GPU accuracy, but no such exception has been granted
for this operator. The test file checks FP32 accuracy against the independent
PyTorch LayerNorm expression, CPU-to-Triton byte equality for FP32/BF16 and the
three real image shapes, and exact batch-position invariance.
Do not claim cross-backend byte equality or full #386 acceptance until the CUDA
extension is built and compared on the required shapes and dtypes.
