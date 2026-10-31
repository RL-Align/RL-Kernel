# Item 18 C128 KV compressor — MI300X validation

Branch: `feat/dsv4-kv-c128-bi`
Target: `dsv4-rocm`
Hardware: AMD Instinct MI300X VF (`gfx942`)
Environment: ROCm/HIP 7.14.60850, PyTorch 2.12.0+rocm7.14.0a20260608,
Triton 3.7.0, pinned aiter commit `326cffa38c9eda16bef19375ba55334e4d3acd59`,
pinned mLite runtime commit `327a238239448de6771beea185dc6902a8de3eaa`.

## Test evidence

- CPU-safe tests: **67 passed** in 5.65s.
- Real MI300X tests: **30 passed**, 0 skipped, 1 existing warning, 28.58s.
- Full targeted contract + DeepSeek-V4 suite: **239 passed**, 0 skipped, 1
  existing warning, 28.50s without coverage and 36.18s with coverage.
- Scoped Python wrapper coverage: **299/322 statements (93%)** and **65/66
  branches**. Coverage does not measure FlyDSL/native GPU instructions; the
  unexecuted wrapper lines are provider-constructor failure paths.
- Ruff format and lint: passed for all ten Python files.
- Real MI300X coverage includes aiter C128 pooling, aiter norm/RoPE,
  pinned mLite functional VJP, fresh-process/JIT isolation, streams and the
  uneven two-rank Gloo ownership fixture.
- The forward oracle is independent FP64 softmax/pool/RMS/rotation arithmetic.
  The backward oracle uses independent autograd of that formula and finite
  differences for parameter entries.

The tests deliberately do not claim RCCL/multi-GPU CP, Graph replay, live cache
state, prefill/decode runtime or full-model acceptance. Those are item 21/27
and communication tasks.

The benchmark is kept separate from correctness tests and records bind,
pool-only and full private-output timing. It includes Python dispatch and
explicit synchronization, so it is not a kernel-only or throughput claim.

## Synchronized MI300X benchmark (30 repetitions)

| Complete groups | Tokens | Bind (us) | Pool only (us) | Private forward (us) |
|---:|---:|---:|---:|---:|
| 1 | 128 | 378.45 | 58.47 | 436.13 |
| 8 | 1,024 | 5,378.20 | 64.67 | 464.99 |
| 32 | 4,096 | 4,029.29 | 64.68 | 481.90 |
| 128 | 16,384 | 22,281.68 | 56.38 | 453.14 |

These are synchronized average wall times with warmup and Python dispatch
included. They are not kernel-only latency or a speedup claim. The raw
benchmark also records provider source fingerprints.
