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
compiled binary.

Local CUDA validation environment (2026-09-23): RTX 3050 Ti (SM86, 4 GiB),
driver 610.60, CUDA toolkit/runtime 12.6, PyTorch 2.12.1+cu126, Triton 3.7.1,
Python 3.12.12. The extension was built with fast math disabled for SM86.
The focused suite plus all three real shapes passed 18 tests with no skips.
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

**Remaining acceptance boundaries:** the maintainer has not answered whether
the indexed `modulate_index` path must be optimized beyond its explicit
PyTorch fallback. The trace fingerprint is a semantic kernel version rather
than a compiled-binary hash. The exactness
evidence above covers FP32 at the focused shape and BF16 at the three real image
shapes; a broader CUDA shape matrix remains optional evidence, not a result
claimed here.
