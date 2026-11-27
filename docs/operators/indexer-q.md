# indexer_q_rope_fp8 (DSv4 indexer query)

> Part of the DSv4 SM90 batch-invariant kernel set. Design, module map, how to run the
> tests and benchmarks: [DSv4 design](../architecture/dsv4-design.md).

RoPE plus per-(token, head) FP8 quantization of the DSv4 lightning-indexer
query, with the FP8 scale folded into the per-head weights (see the
[DSv4 design](../architecture/dsv4-design.md)). Its outputs feed DeepGEMM's
`fp8_mqa_logits` / `fp8_paged_mqa_logits`, whose scores pick the KV entries
each query attends to, so the bytes must match the rollout. The CUDA kernel is
byte-equal to the PyTorch golden, and both are byte-equal to the kernel the
vLLM CUDA rollout actually runs.

## Which vLLM kernel

vLLM (`ISEEKYAN/vllm@1a6542c`) has two implementations of
`fused_indexer_q_rope_quant` that round differently:

| | CuTe DSL `IndexerQFp8Kernel` (used) | Triton fallback (not reproduced) |
|---|---|---|
| when | `nvidia-cutlass-dsl` importable, which the vLLM CUDA build requires | otherwise |
| RoPE | `fma(q0, cos, -(q1 sin))`, `fma(q0, sin, q1 cos)` | `xe*cos - xo*sin`, `xo*cos + xe*sin` (Triton contraction) |
| scale | `max(amax, 1e-4) * fp32(1/448)`, bit-level ceil to a power of two | `div_rn(max(amax, 1e-4), 448)`, `exp2(ceil(log2))` |
| weights | `(w * fp32(softmax * head)) * scale` | `w * scale * softmax * head`, rounded each step |

The FMA contraction was established by running the compiled CuTe kernel
against all nine contraction variants; exactly one matches.

## Semantics

For each (token, head) row of `head_dim = 128` BF16 values:

```text
values [0, 128 - rope_dim)      unchanged
values [128 - rope_dim, 128)    GPT-J pairs (q0, q1), cos/sin = cos_sin_cache[pos] (cos first)
    rot0 = fma(q0, cos, -(q1 * sin)),  rot1 = fma(q0, sin, q1 * cos),  rounded to BF16
amax  = max |value| (NaN ignored, infinity counts)
s     = max(amax, 1e-4) * fp32(1/448)
e     = ((bits(s) + 0x7FFFFF) >> 23) & 0xFF,    scale = 2^(e-127)
q_fp8 = e4m3 satfinite(value * 2^-(e-127))
w_out = (fp32(w) * fp32(softmax_scale * head_scale)) * scale
```

- `positions` int32 or int64; `cos_sin_cache` FP32 or BF16 `[max_position, rope_dim]`
  (`rope_dim` a multiple of 16); `weights` BF16 `[tokens, heads]` (the
  `weights_proj` output); outputs FP8 E4M3 `[tokens, heads, 128]` and FP32
  `[tokens, heads]`.
- An infinite amax gives `e = 255`, so the scale is `+inf` and the inverse bit
  pattern is `-inf`, exactly as in vLLM (values flip sign and saturate, zeros
  become NaN, the weight becomes infinite).
- Each row depends only on itself, so the operator is batch-invariant.

## Backends

| Backend | Entry point | Notes |
|---|---|---|
| PyTorch golden | `rl_engine.kernels.ops.pytorch.indexer.indexer_q_rope_fp8` | CPU and CUDA; exact FP32 FMA emulated in FP64 with a TwoSum midpoint fix |
| CUDA | `rl_engine.kernels.ops.cuda.indexer.indexer_q_rope_fp8_cuda` (`IndexerQRopeFp8CudaOp`) | `csrc/cuda/indexer/indexer_q_rope_fp8.cu`; head dim 128; q rows contiguous and 16-byte aligned (head-strided views are fine) |

The kernel gives eight threads to a row (16 values each, kept as packed BF16
pairs) and two heads of a token to each thread group, so the token's rotary
values load once. Every FP32 step is an explicit `__fmaf_rn` / `__fmul_rn`,
the amax is a SIMD integer max over BF16 bit patterns, and the file refuses
fast-math.

## Performance

H200, 64 heads, FP32 rotary cache, CUDA-graph replay, median.

| Tokens | CUDA | vLLM CuTe | eager golden |
|---|---|---|---|
| 1 | 8.6 us | 8.4 us | 607 us |
| 512 | 9.5 us | 9.2 us | 639 us |
| 4096 | 39.6 us (2.6 TB/s) | 34.5 us | 3312 us |
| 16384 | 128.4 us (3.2 TB/s) | 113.1 us | 11764 us |

## Tests

```bash
python -m pytest tests/ops/indexer -q
python scripts/check_indexer_q_vllm_parity.py   # needs vLLM's CuTe kernel importable
python benchmarks/benchmark_indexer_q.py [--vllm]
```

`tests/ops/indexer/test_indexer_q.py` pins the golden (including an FMA case
where naive double rounding is wrong, and the floor / NaN / infinity paths);
`tests/ops/indexer/test_indexer_q_cuda.py` checks CUDA bytes against the golden
over magnitudes from 1e-6 to 3e4, both cache dtypes, both position dtypes,
rope widths 16-128, paired and unpaired head grouping, head-strided views and
batch invariance. The parity script matched vLLM's CuTe kernel on 50/50 cases.
