# DSv4 MoE Router

Batch-invariant router for DeepSeek-V4 MoE layers (P3 redo, SM90): gate GEMM, top-6 or hash
routing, routing weights, and the dX backward with a frozen gate.

| Piece | Where |
|---|---|
| Golden (contract) | `rl_engine/kernels/ops/pytorch/router/dsv4_router.py` |
| CUDA kernels | `csrc/cuda/router/dsv4_router.cu` (`dsv4_router_topk`, `dsv4_router_backward_dx`) |
| Python wrapper | `rl_engine/kernels/ops/cuda/router/dsv4_router.py` |
| Layer (gate GEMM + autograd) | `rl_engine/models/deepseek_v4/router.py` (`Dsv4Router`, `gate_logits`) |
| Tests | `tests/ops/router/test_dsv4_router.py`, `tests/models/test_dsv4_router_layer.py` |
| Benchmark | `benchmarks/benchmark_dsv4_router.py` |

## Contract

Gate: `logits = x @ W.T`, BF16 `x [T, 4096]` and `W [256, 4096]`, FP32 output on pinned
DeepGEMM `bf16_gemm_nt` (design doc D4). Then, per token (the golden is defined on CUDA:
libdevice `expf` / `log1pf`; every other step is correctly rounded, no FMA):

```text
s_e   = sqrt(max(x_e, 0) + log1pf(expf(-|x_e|)))
key_e = s_e + bias_e                 NaN -> -1e30
ids   = the 6 largest keys, ties to the smallest expert id
        (hash layers, layer < num_hash_layers: ids = tid2eid[input_ids])
w_k   = s_{ids_k}                    NaN -> 0
S     = (((((w_0 + w_1) + w_2) + w_3) + w_4) + w_5)
out_k = w_k * (1.5 / (S if S > 0 else 1))
```

Outputs: `weights` FP32 `[T, 6]`, `ids` int32 `[T, 6]` in slot order.

### Differences from vLLM

vLLM (`ISEEKYAN/vllm@1a6542c`) uses two kernels with different softplus formulas: the Triton
`dsv4_topk` for scored layers (`where(x > 20, x, log(1 + exp(x)))`, Triton approximate
math) and the CUDA `dsv4HashTopkSoftplusSqrt` for hash layers (`__expf` / `__logf`). Its BI
gate is cuBLAS `torch.mm`. Matching those bits would mean emulating approximate PTX, so the
arithmetic here is our own and the rollout patch runs this kernel too. The routing rules
(tie order, NaN handling, unbiased weights, renormalisation, the 1.5 scale) are vLLM's.

## Backward

The gate weight and the correction bias are frozen; autograd returns only `dL/dx`. With
`d = dL/dout`:

```text
g     = sum_k d_k * out_k                     slot order
dw_k  = (1.5 * d_k - g) / S                   (1.5 * d_k if S <= 0)
dx_k  = dw_k * sigmoid(x_k) / (2 * s_k)       0 where s_k is 0, inf or NaN
dX[t] = sum_k dx_k * W[ids_k]                 FP32, slot order (BF16 or FP32 output)
```

Only six logits per token carry a gradient, so the gate dgrad is a six-row gather: exact in
that order and batch-invariant.

## Results (H200, torch 2.13 / CUDA 13)

- CUDA is byte-equal to the golden: random, quantised (many ties) and extreme logits; E = 256
  and 384; T = 0, 1, 7, 1000; hash tables int32 / int64; dX in FP32 and BF16.
- The layer (gate + routing) is batch-invariant on scored and hash layers; autograd dX equals
  the golden.
- Graph-replay time in microseconds. The baseline is cuBLAS plus vLLM's Triton top-k for the
  forward, and the dense `scatter + torch.mm` dgrad for dX:

| T | forward | baseline | ratio | dX | dense | ratio |
|---|---|---|---|---|---|---|
| 1 | 14.8 | 12.3 | 1.20 | 8.0 | 12.3 | 0.66 |
| 64 | 15.6 | 12.8 | 1.22 | 8.0 | 13.9 | 0.58 |
| 256 | 16.1 | 13.4 | 1.20 | 8.4 | 17.6 | 0.48 |
| 1024 | 19.6 | 16.7 | 1.17 | 12.4 | 21.1 | 0.59 |
| 4096 | 31.2 | 30.3 | 1.03 | 29.5 | 42.7 | 0.69 |
| 16384 | 88.6 | 78.0 | 1.14 | 100.0 | 97.2 | 1.03 |
