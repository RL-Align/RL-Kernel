# Indexer stable Top-512 (item 17)

```text
scores[M,N] FP32, lengths[M] int -> ids[M,512] int32
```

* Higher score first; ties pick the smaller column; signed zeros tie.
* `-inf` and columns `>= lengths[m]` are never chosen; short rows pad with `-1`.
* Output is column descending, `-1` last: the compressed-row read order.
  `JointMetadata.bind` (item 24) and `IndexerAuxMetadata.bind` (item 26) read
  compressed rows in this order, then the window ascending.

Kernel: one Triton program per row. Keys pack aiter `fpval_to_key(score)` and
the inverted column into a uint64 (item 31 scheme); `tl.topk` merges 512-column
blocks. gfx942 only, no Graph capture.

```python
ids = IndexerTopK()(record.scores, lengths)   # record from IndexerScore
```

```bash
python -m pytest tests/models/deepseek_v4/test_indexer_topk.py \
  tests/models/deepseek_v4/test_indexer_topk_rocm.py -q
python benchmarks/benchmark_indexer_topk.py --output benchmark.json
```

Results: `reports/indexer-topk-mi308x-20261115.md`.

Not covered: gfx950, CUDA Graph, DCP, vLLM/Megatron wiring (they still select
their own Top-K and keep the old read order).
