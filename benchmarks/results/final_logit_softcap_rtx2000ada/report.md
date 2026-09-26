# Final logit softcap benchmark

This report preserves the Markdown section of the contributor-supplied CUDA
benchmark console output received on 2026-09-25. It was measured on the server,
not on the macOS development machine. All 12 shape/dtype combinations passed
output and gradient accuracy checks before the 36 timing rows were produced.

This is one run on one GPU. It includes public-wrapper/autograd dispatch gaps,
so it is not isolated kernel execution time. Small FP32 cases and the smallest
BF16 backward case were slower than eager PyTorch; no across-the-board speedup
is claimed. The server's original [JSON report](results.json), including timing
standard deviations, accuracy errors and additional environment metadata, is
archived byte-for-byte alongside this report. All 36 rows match the rounded
console values, and the tolerance contract fingerprint matches the measured
checkout. Source JSON SHA256:
`eacf105f459bcf4d8fe6edf66189e8310b598229157d192dfd22774b4feae588`.

The JSON identifies Python 3.12.3, NVIDIA driver 580.159.04, compute capability
8.9, 22 multiprocessors and 15.57 GiB of device memory. The measured seed was 415.

Reproduction (at the measured commit below):

```bash
python benchmarks/benchmark_final_logit_softcap.py \
  --output-dir reports/final-logit-softcap
```

GPU: NVIDIA RTX 2000 Ada Generation; PyTorch: 2.13.0+cu130; Triton: 3.7.1; CUDA runtime: 13.0.
Commit: `ec0b1dd6b70984c5b6c21425490ab0ae7592feb8`; tracked changes: `none`.
Warmup: 10; measured repetitions: 50; BLOCK_SIZE: 1024.

Contiguous inputs; median CUDA event latency for public wrappers, including allocation and autograd dispatch. Input generation, correctness checks and JIT compilation are outside timing. Small cases can be dominated by host dispatch gaps.
Backward reuses a prebuilt graph; forward+backward builds a fresh graph per call. Forward runs under no_grad. Extra peak allocation excludes inputs and prebuilt graphs.
Speedup = eager PyTorch / Triton; values below 1 mean Triton was slower. All reported cases passed the output and gradient tolerance checks first.

| Input | Shape | Mode | Native ms | Triton ms | Speedup | Native extra MiB | Triton extra MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | 1x1x1025 | forward | 0.077824 | 0.071664 | 1.09x | 0.01 | 0.00 |
| fp16 | 1x1x1025 | backward | 0.229712 | 0.213008 | 1.08x | 0.01 | 0.00 |
| fp16 | 1x1x1025 | forward_backward | 0.414960 | 0.393808 | 1.05x | 0.02 | 0.01 |
| fp16 | 1x1x262144 | forward | 0.115968 | 0.107104 | 1.08x | 3.00 | 1.00 |
| fp16 | 1x1x262144 | backward | 0.259120 | 0.250496 | 1.03x | 2.00 | 0.50 |
| fp16 | 1x1x262144 | forward_backward | 0.416832 | 0.385664 | 1.08x | 4.00 | 1.50 |
| fp16 | 1x16x262144 | forward | 0.463984 | 0.101296 | 4.58x | 48.00 | 16.00 |
| fp16 | 1x16x262144 | backward | 0.616992 | 0.301360 | 2.05x | 32.00 | 8.00 |
| fp16 | 1x16x262144 | forward_backward | 1.006288 | 0.404144 | 2.49x | 64.00 | 24.00 |
| fp16 | 1x64x262144 | forward | 2.473984 | 0.532256 | 4.65x | 192.00 | 64.00 |
| fp16 | 1x64x262144 | backward | 2.826288 | 0.788064 | 3.59x | 128.00 | 32.00 |
| fp16 | 1x64x262144 | forward_backward | 5.232096 | 1.200816 | 4.36x | 256.00 | 96.00 |
| bf16 | 1x1x1025 | forward | 0.077936 | 0.070832 | 1.10x | 0.01 | 0.00 |
| bf16 | 1x1x1025 | backward | 0.227584 | 0.248000 | 0.92x | 0.01 | 0.00 |
| bf16 | 1x1x1025 | forward_backward | 0.411760 | 0.390592 | 1.05x | 0.02 | 0.01 |
| bf16 | 1x1x262144 | forward | 0.116336 | 0.111696 | 1.04x | 3.00 | 1.00 |
| bf16 | 1x1x262144 | backward | 0.251872 | 0.248208 | 1.01x | 2.00 | 0.50 |
| bf16 | 1x1x262144 | forward_backward | 0.421392 | 0.388688 | 1.08x | 4.00 | 1.50 |
| bf16 | 1x16x262144 | forward | 0.457024 | 0.106384 | 4.30x | 48.00 | 16.00 |
| bf16 | 1x16x262144 | backward | 0.602016 | 0.299776 | 2.01x | 32.00 | 8.00 |
| bf16 | 1x16x262144 | forward_backward | 0.981376 | 0.452352 | 2.17x | 64.00 | 24.00 |
| bf16 | 1x64x262144 | forward | 2.475392 | 0.532640 | 4.65x | 192.00 | 64.00 |
| bf16 | 1x64x262144 | backward | 2.831904 | 0.782224 | 3.62x | 128.00 | 32.00 |
| bf16 | 1x64x262144 | forward_backward | 5.230912 | 1.202800 | 4.35x | 256.00 | 96.00 |
| fp32 | 1x1x1025 | forward | 0.066704 | 0.074224 | 0.90x | 0.01 | 0.00 |
| fp32 | 1x1x1025 | backward | 0.190960 | 0.245408 | 0.78x | 0.01 | 0.00 |
| fp32 | 1x1x1025 | forward_backward | 0.348624 | 0.400656 | 0.87x | 0.02 | 0.01 |
| fp32 | 1x1x262144 | forward | 0.096176 | 0.122272 | 0.79x | 2.00 | 1.00 |
| fp32 | 1x1x262144 | backward | 0.224768 | 0.277344 | 0.81x | 2.00 | 1.00 |
| fp32 | 1x1x262144 | forward_backward | 0.370448 | 0.385264 | 0.96x | 4.00 | 2.00 |
| fp32 | 1x16x262144 | forward | 0.393664 | 0.194672 | 2.02x | 32.00 | 16.00 |
| fp32 | 1x16x262144 | backward | 0.562000 | 0.376576 | 1.49x | 32.00 | 16.00 |
| fp32 | 1x16x262144 | forward_backward | 0.892576 | 0.522736 | 1.71x | 64.00 | 32.00 |
| fp32 | 1x64x262144 | forward | 1.971664 | 0.692240 | 2.85x | 128.00 | 64.00 |
| fp32 | 1x64x262144 | backward | 2.338992 | 1.104080 | 2.12x | 128.00 | 64.00 |
| fp32 | 1x64x262144 | forward_backward | 4.245152 | 1.681168 | 2.53x | 256.00 | 128.00 |

## Timing variability

Selected rows illustrate the dispersion recorded in the original JSON.
All values below are milliseconds. Standard deviation is across the 50
individual timed invocations; it is not a confidence interval for the median.

| Input | Shape | Mode | Native median | Native std | Triton median | Triton std |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| fp16 | 1x1x262144 | forward | 0.115968 | 0.084987 | 0.107104 | 0.045300 |
| fp32 | 1x1x1025 | backward | 0.190960 | 0.007279 | 0.245408 | 0.024885 |
| bf16 | 1x16x262144 | forward_backward | 0.981376 | 0.873993 | 0.452352 | 0.044340 |
| fp32 | 1x64x262144 | forward | 1.971664 | 0.030552 | 0.692240 | 0.382200 |
| bf16 | 1x64x262144 | forward_backward | 5.230912 | 0.153233 | 1.202800 | 0.155220 |

Some near-parity comparisons fluctuate substantially. Small FP32 backward
also has a visible median regression that should not simply be dismissed as
noise. Raw per-repetition samples were not saved by the profiler, and these
summaries cannot determine statistical significance or whether host scheduling,
clock changes, contention, or kernel execution caused the variation.

Keep this as an initial public-wrapper benchmark. Before selecting a new
block size or introducing dispatch thresholds, repeat measurements on an idle
GPU and measure kernel execution separately from host dispatch. No kernel
configuration was changed on the basis of these timings.
