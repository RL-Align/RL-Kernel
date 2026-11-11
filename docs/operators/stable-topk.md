# stable_topk_per_row (DSv4 indexer)

Per-row top-k selection for the DSv4 sparse-attention indexer (P2 T04; design
doc `docs/architecture/dsv4-cuda-design.md`, rule 2). It picks the KV entries
each query attends to, so its output must not depend on the batch, and it must
match the rollout: the CUDA kernel is byte-equal to the PyTorch golden, and
both are byte-equal to vLLM's batch-invariant DSv4 selector
(`torch.ops.ds4_bi.top_k_per_row_prefill`, `ISEEKYAN/vllm@1a6542c`), which the
rollout uses for prefill and decode under `VLLM_BATCH_INVARIANT=1`.

## Semantics

```text
row r considers logits[r, row_starts[r] : row_ends[r]]
candidate id = column - row_starts[r]                 (request-local index)
key          = (ordered_bits(score) << 32) | id       unique; ranked descending
ordered_bits(x) = x < 0 (sign bit set) ? ~bits(x) : bits(x) ^ 0x80000000
output       = int32 [rows, top_k]: ids in key order, then -1
```

- Equal scores prefer the **larger** local index (vLLM's order).
- Every FP32 bit pattern has a fixed place: `-0.0` ranks just below `+0.0`;
  NaNs with the sign bit clear rank above `+inf`, those with it set below
  `-inf`.
- `0 < top_k <= 2048`. Rows shorter than `top_k` are padded with `-1`; an empty
  or reversed range gives a row of `-1`.
- The selection reads nothing outside its own row range, so it is
  batch-invariant by construction. It is exact (no floating-point arithmetic),
  so any correct implementation gives the same bytes.

## Backends

| Backend | Entry point | Notes |
|---|---|---|
| PyTorch golden | `rl_engine.kernels.ops.pytorch.topk.stable_topk_per_row` | CPU and CUDA; rejects ranges outside `[0, width]` |
| CUDA | `rl_engine.kernels.ops.cuda.topk.stable_topk_per_row_cuda` (`StableTopkCudaOp`) | `csrc/cuda/topk/stable_topk.cu`; fills an `out=` view (unit column stride) for persistent index buffers; clamps out-of-range rows instead of checking them on the host |

The CUDA kernel uses one block per row. Pass 1 histograms the top 11 score
bits and locates the bucket holding the k-th key. Pass 2 keeps everything
above that bucket and copies the bucket's keys (as full 64-bit keys) into
shared memory. A radix select on those candidates finishes the job without
touching global memory again, and ties resolve by index for free because the
key is unique. A bucket too large for the candidate list (huge tie groups) falls
back to more global radix passes and a tie-cutoff scan from the row's end. The
selected keys are sorted with a CUB block radix sort over `32 + log2(length)`
bits. Shared-memory atomics only count or claim slots, and every claimed set is
sorted before it is written. The block size (256, or 1024 for fewer rows than
SMs with rows of 32K or more) changes speed only.

## Performance

H200, CUDA-graph replay, median, `top_k = 512`; prefill rows are causal (row
`r` of `R` sees `(r + 1) * W / R` candidates). The baseline is `torch.topk` on
the same rows (not batch-invariant on ties, and it ignores row ranges, so it
does less work). `vllm_bi` is the rollout's selector.

| Shape (rows x width) | relu data | randn data | torch.topk | vllm_bi |
|---|---|---|---|---|
| decode 64 x 4096 | 23.9 us (0.51x) | 26.0 us (0.53x) | 47 us | 65 us |
| decode 256 x 16384 | 50.5 us (0.43x) | 67.5 us (0.57x) | 118 us | 96 us |
| decode 64 x 65536 | 66.1 us (0.59x) | 105.7 us (0.96x) | 111 us | 146 us |
| prefill 4096 x 4096 | 314 us (0.82x) | 279 us (0.73x) | 383 us | 1147 us |
| prefill 4096 x 16384 | 393 us (0.42x) | 438 us (0.47x) | 940 us | 1205 us |
| prefill 8192 x 32768 | 1008 us (0.31x) | 1260 us (0.39x) | 3284 us | 2775 us |

`relu` (scores `relu(randn - 1)`, mostly exact zeros) mimics the indexer's
ReLU-weighted scores and their large tie groups. With `top_k = 2048` the ratios
stay at 0.33-0.65x.

## Tests

```bash
python -m pytest tests/ops/topk -q
python scripts/check_stable_topk_vllm_parity.py --vllm-ref /path/to/ds4_bi_topk_extension.cu
python benchmarks/benchmark_stable_topk.py [--vllm-ref ...] [--data randn] [--threads N]
```

`tests/ops/topk/test_stable_topk.py` pins the golden's semantics, including
vLLM's own test cases. `tests/ops/topk/test_stable_topk_cuda.py` checks the CUDA
bytes against the golden at every block size: random, tie-heavy, ReLU-like,
constant and special-value rows, rows at and around `top_k`, rows up to 262144
wide, a tie cutoff spanning many chunks, strided logits and output views, and
batch invariance. The parity script compares both against vLLM's kernel
(80/80 cases byte-equal, including random bit patterns).
