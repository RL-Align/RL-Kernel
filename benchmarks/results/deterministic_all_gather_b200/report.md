# Deterministic all-gather on B200: before and after

![all_gather before/after](all_gather.png)

Every value is the median over 100 calls, taken on the slowest rank. The data is BF16, and the
size given is the input per rank. "Before" is `main` at `43f150f` and "after" is this change,
both measured with `benchmarks/benchmark_deterministic_collectives.py`. Software: torch 2.13.0,
CUDA 13.0.

- 2 x B200: both builds ran on the same machine.
- 8 x B200: both builds ran on nodes of the same type, but on different nodes.

| per-rank input | 2 GPUs before | 2 GPUs after | 8 GPUs before | 8 GPUs after | NCCL, 8 GPUs |
| --- | --- | --- | --- | --- | --- |
| 1 KiB | 26.5 µs | 22.3 µs | 93.2 µs | 55.8 µs | 30.8 µs |
| 8 KiB | 84.5 µs | 24.7 µs | 464.6 µs | 69.5 µs | 32.0 µs |
| 32 KiB | 284.8 µs | 37.3 µs | 1688.0 µs | 73.1 µs | 33.5 µs |
| 128 KiB | 1083.8 µs | 38.0 µs | 77.7 µs | 73.3 µs | 35.5 µs |
| 256 KiB | 46.1 µs | 45.8 µs | 87.0 µs | 81.1 µs | 37.1 µs |
| 1 MiB | 51.0 µs | 48.2 µs | 133.8 µs | 84.0 µs | 60.3 µs |
| 4 MiB | 84.2 µs | 49.1 µs | 348.1 µs | 144.2 µs | 130.7 µs |

Before this change, every gather whose output was at most 256 KiB ran on a single block that
copied one byte per thread. Its cost grew at about 4 µs per KiB of output on 2 GPUs and about
6.5 µs per KiB on 8 GPUs. After the change:

- the copy moves 16-byte vectors and resolves each peer once;
- only outputs of at most 64 KiB stay on the single-block path.

`all_reduce`, `reduce_scatter` and `all_gather_many` are not changed. In the JSON files their
before/after rows agree within run-to-run noise: a repeated 2-GPU `all_reduce` measured 22–24 µs
before and 23–29 µs after at 1 KiB, and about 79 µs for both at 32 KiB.
