# DSv4 KV compressor (indexer entries)

The DSv4 compressor turns every `ratio` tokens of a request into one KV entry
(P2 T02; design doc `docs/architecture/dsv4-cuda-design.md`). This page covers
the indexer compressor (head 128, C4 with overlap), whose entries fill the FP8
indexer K cache that DeepGEMM's MQA-logits kernels read. The model math is
vLLM's; the arithmetic order is our own fixed contract (design decision (a)),
so the CUDA kernel is byte-equal to the golden, while vLLM's Triton kernel
agrees on about 99.999% of the FP8 bytes.

## Semantics

State save (every token; an exact copy, run as PyTorch indexing):

```text
state[slot] = [kv (2*128) | score + ape[pos % 4] (2*128)]          FP32, paged
```

Compress (each boundary token, `(pos + 1) % 4 == 0`), per channel, over the
window `pos-7 .. pos` (negative positions skipped; the older 4 tokens read
channel half 0, the newer 4 half 1):

```text
m = max_j s_j;  e_j = expf(s_j - m);  S = ((e_0 + e_1) + ...)       ascending j
x = (((kv_0 * e_0) + kv_1 * e_1) + ...) / S                         no FMA
v = sum x^2: 4 channels per thread in order, 32-lane butterfly, warps ascending
r = 1 / sqrt(v / 128 + eps);   n = (x * r) * w
GPT-J RoPE on the last 64 channels at position (pos // 4) * 4 (products rounded)
BF16 round; FP8 entry: amax (NaN ignored), scale = 2^ceil(log2(max(amax,1e-4)/448))
from the bits, e4m3 satfinite bytes + FP32 scale
```

`expf` is libdevice's, so the golden is defined on CUDA (where `torch.exp` is
the same function). Every entry reads only its own request's state through
its block table: batch-invariant, and chunked prefill (state saved over
several chunks) gives the same entries as one pass.

## Backends

| Backend | Entry point | Notes |
|---|---|---|
| PyTorch golden | `rl_engine.kernels.ops.pytorch.compressor` (`save_states`, `compress_indexer`) | CUDA only |
| CUDA | `rl_engine.kernels.ops.cuda.compressor.compress_indexer_cuda` | `csrc/cuda/compressor/compress_fp8.cu`; writes the paged K cache (`k_block_size` FP8 rows of 128, then their FP32 scales: DeepGEMM's layout) at `k_slots` |

One 32-thread block per token (non-boundary tokens exit at once); each thread
owns 4 consecutive channels and keeps the 8-token window in registers.

## Performance

H200, CUDA-graph replay, median; vLLM's Triton kernel on the same caches.

| Case | CUDA | vLLM Triton | FP8 bytes equal to vLLM |
|---|---|---|---|
| decode, 256 requests | 9.9 us | 9.7 us | 100% |
| prefill 1 x 4096 tokens | 12.9 us | 11.4 us | 99.9985% |
| prefill 4 x 4096 tokens | 21.9 us | 19.3 us | 99.9998% |
| prefill 1 x 32768 tokens | 38.6 us | 36.9 us | 99.9995% |

## Tests

```bash
python -m pytest tests/ops/compressor -q
python benchmarks/benchmark_compressor.py [--vllm-src .../fused_compress_quant_cache.py]
```
