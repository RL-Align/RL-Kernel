# Item 17 MI308X report, 2026-11-15

Base `dsv4-rocm` `b184c3fb`. Same environment as #678/#702.

|Component|Version|
|---|---|
|Image|`vllm/vllm-openai-rocm:v0.26.0` (`sha256:5709fafe…a97a18`)|
|Torch / HIP / Triton|`2.11.0+gitd0c8b1f` / `7.2.53211` / `3.6.0`|
|aiter / vLLM|`0.1.16.post3` / `0.26.0+rocm723`|
|GPU|MI308X `gfx942`, one card|

Tests: item 17 28 passed (7 CPU, 21 GPU), byte-equal to a CPU reference;
item 24/26 283 passed. `tests/contracts/operators tests/models/deepseek_v4`:
1599 passed, 14 skipped, 15 failed (`test_indexer_score_rocm.py`, fixed by #702).

`benchmark_indexer_topk.py --repeats 30`, K=512, median µs:

|M|N|Op|Kernel|torch.topk|
|---:|---:|---:|---:|---:|
|1|4096|213.6|89.1|73.8|
|32|4096|226.5|89.6|78.8|
|128|16384|640.4|477.0|151.8|
|512|16384|1889.3|1600.6|418.4|
|16|131072|2788.8|2615.4|168.3|

`torch.topk` has no tie rule (speed reference only). One program per row makes
long rows slow (15x at N=131072).
