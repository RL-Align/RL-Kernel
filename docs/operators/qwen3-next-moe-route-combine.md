# Qwen3-Next MoE route / combine contract

## Summary

Qwen3-Next's sparse MoE block routes every token to 10 of 512 experts (expert
width 512, TP4-local width 128) and adds a gated shared expert. RFC #428 row
`moe_route_combine_contract` asks for a routed MoE whose output for a token is a
pure function of that token: the same bits whatever else is in the batch, in
training replay and in rollout. This page states the contract and how
`rl_engine.models.qwen3_next.qwen3_next_forward.shared_moe` meets it.

Upstream references: `transformers` `Qwen3NextSparseMoeBlock` (5.17.0) and vLLM
0.30.0 `fused_topk` + `fused_experts`.

## Entry Point

```python
from rl_engine.models.qwen3_next.qwen3_next_forward import shared_moe, stable_top10_routes
from rl_engine.models.qwen3_next.qwen3_next_tp_blocks import TP4MoE

routed, routes = shared_moe(x, router_weight, gate_up, down)  # TP-local, not reduced
block = TP4MoE(group=tp_group, device="cuda")                  # routed + shared, reduced
block.load_hf_weights(hf_layer_mlp_weights)
y = block(hidden)
```

`TRITON_F32_DEFAULT=ieee` must be set before the process starts: the FP32 router
GEMM refuses to run on TF32 Triton dots.

## Contract

| Stage | Order and dtype | Atomics |
| --- | --- | --- |
| Router | BF16 `x`, `W` upcast to FP32; one pinned vLLM batch-invariant GEMM (IEEE FP32) | none |
| Softmax | max subtraction, `exp`, row sum by a fixed pairwise tree (`fixed_order_row_sum`) | none |
| Selection | `argsort(logits, descending, stable)[:, :10]`: score descending, equal scores to the lower expert id | none |
| Weights | the ten selected FP32 probabilities divided by their fixed-tree sum | none |
| Experts | per (token, route) row: `gate_up` GEMM (BF16 in, FP32 accumulate, BF16 out), SwiGLU in FP32, BF16 cast, `down` GEMM | none: one writer per row |
| Combine | `out = e0*w0`, then `out += e_k*w_k` for k = 1..9 in route order, FP32; one BF16 cast | none |
| Shared expert (TP4 block) | `routed + shared * sigmoid(gate)` in BF16, then one TP all-reduce | collective as configured |

Payload dtype between stages: BF16 expert outputs, FP32 route weights, BF16
block output. The expert GEMMs are vLLM 0.30.0's `fused_moe` Triton kernel with
the tile vLLM itself uses under `VLLM_BATCH_INVARIANT=1`
(`BLOCK_SIZE_M/N/K = 64/64/32`, `GROUP_SIZE_M = 8`, `SPLIT_K = 1`), passed
explicitly so no environment variable, tuned-config file or token count can
change it. The routed weight is *not* multiplied inside the kernel and the
kernel does not sum over routes; that stays in the combine above.

Fail-closed checks: CUDA tensors only (no CPU fallback, no ROCm), BF16 weights of
the exact `[512, 2*I, H]` / `[512, H, I]` shapes, finite FP32 router logits,
vLLM exactly 0.30.0, `VLLM_TRITON_USE_TD` off (the tensor-descriptor path is a
different instruction stream), and IEEE Triton FP32.

### Backward

`_RoutedExperts.backward` recomputes the `[gate, up]` projection with the same
grouped kernel, then visits experts in ascending id. Each expert's `dgate_up`
and `ddown` slice is one pinned GEMM written once into a dense buffer; `dx` is
accumulated per token in ascending expert order (a token's ten experts are
distinct, so each launch has unique rows). Router-weight gradients flow through
the selected probabilities; the indices are discrete.

## Reuse decision

| Implementation | Routes BI | Output BI | Backward | Why not reused as is |
| --- | --- | --- | --- | --- |
| vLLM `fused_topk` + `fused_experts`, `VLLM_BATCH_INVARIANT=0` | no | no | none | tuned configs and the BF16 router GEMM depend on the token count |
| vLLM, `VLLM_BATCH_INVARIANT=1` | yes | yes | none | no VJP for training; BF16 router logits (Qwen3-Next and VIME route in FP32); routed weight applied in the expert GEMM epilogue on the BF16 output, then routes summed by `moe_sum` |
| HF `Qwen3NextExperts` (eager) | yes | yes | autograd | `dx`/`dW` change with batch size (cuBLAS shape heuristics); BF16 router; `index_add_` combine in BF16 |
| FlashInfer `cutlass_fused_moe` | (torch routing) | no | none | output changes with batch size; no VJP |
| Megatron-core 0.16 `MoELayer` + TE 2.16 grouped GEMM, as VIME configures Qwen3-Next | no (256+) | no (1024) | autograd, not BI | FP32 router routes like FP64, but the routing, output, dx and dW change with batch size |
| SGLang 0.5.21 Triton `fused_moe`, default | no (256+) | no (256+) | none | tuned configs and the BF16 router GEMM depend on the token count |
| SGLang, deterministic inference | yes | yes | none | inference only; BF16 router logits; its deterministic tile (64/64/32) is the one vLLM uses in batch-invariant mode |

What RL-Kernel reuses: vLLM's batch-invariant GEMM for the router and the
backward, and vLLM's `fused_moe` Triton kernel for the forward expert GEMMs,
which is batch-invariant per route at the fixed tile. What it implements: the
FP32 fixed-order routing, the stable tie rule, the atomic-free FP32 combine, the
deterministic backward and the TP4 boundaries. The grouped kernel's per-route
output is bitwise equal to the per-expert pinned GEMM loop it replaced
(`tests/models/qwen3_next/check_qwen3_next_forward.py::test_grouped_*`), so this is a speed change
with no change in bits.

## Results

Measured on B200 with `tools/validation/models/qwen3_next_moe_prior_art.py` and the TP4 gate
`tools/validation/models/qwen3_next_tp_moe_check.py`; the reports, the figure and the exact
commands are in
[`docs/usage/evidence/qwen3-next-moe-route-b200/`](../usage/evidence/qwen3-next-moe-route-b200/README.md).

![Qwen3-Next routed MoE prior art](../usage/evidence/qwen3-next-moe-route-b200/figure.png)

TP1 shape (H=2048, 512 experts, top-10, width 512), random weights, BF16:

| | routes / output BI (16-1024 tokens) | dx / dW BI | rel. L2 vs FP64 | tokens routed unlike FP64 (of 256) | forward, 1 / 64 / 1024 / 4096 tokens (ms) | fwd + bwd, 64 / 1024 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| **RL-Kernel `shared_moe`** | **yes / yes** | **yes / yes** | **3.9e-3** | **0** | 1.39 / 1.69 / 2.04 / 3.05 | 114 / 169 |
| Megatron-core + TE (VIME config) | no / no | no / no (1024) | 4.6e-3 | 0 | 3.8 / 9.7 / 11.7 / 12.1 | 45 / 51 |
| vLLM BI=1 | yes / yes | no backward | 7.0e-2 | 11 | 0.39 / 0.66 / 0.87 / 1.05 | - |
| SGLang deterministic | yes / yes | no backward | 7.0e-2 | 11 | 0.35 / 0.66 / 0.84 / 1.37 | - |
| vLLM BI=0 | no / no | no backward | 7.0e-2 | 11 | 0.38 / 0.66 / 0.87 / 1.06 | - |
| SGLang default | no / no | no backward | 7.0e-2 | 11 | 0.34 / 0.61 / 0.83 / 1.35 | - |
| FlashInfer CUTLASS | no / no | no backward | 7.0e-2 | 11 | 0.37 / 0.72 / 0.90 / 1.12 | - |
| HF eager | yes / yes | no / no (1024) | 7.0e-2 | 11 | 2.1 / 62 / 89 / 87 | 877 / 1284 |

* The accuracy gap is the router: every candidate with BF16 router logits sends
  11 of 256 tokens to a different expert set than FP64 routing does. RL-Kernel
  and Megatron-core (VIME's training path) route in FP32 and select the same
  experts as FP64.
* The only batch-invariant candidates are RL-Kernel and the inference-only
  modes of vLLM and SGLang; of these, only RL-Kernel has a backward, so only it
  can run the same forward on the training and the rollout side.
* RL-Kernel's forward is 2-4x slower than vLLM's. About 0.5 ms of it, at any
  token count, is the FP32 IEEE router GEMM, which is part of the contract.
  The expert GEMMs themselves run in vLLM's kernel. The per-expert GEMM loop
  they replace (same bits) took 47 / 66 / 71 ms at 64 / 1024 / 4096 tokens in
  a local B200 run of the same runner; that run is not part of this evidence.
* The backward is still a per-expert loop of pinned GEMMs (512 experts x 4
  GEMMs): 169 ms at 1024 tokens, against 51 ms for Megatron-core + TE, which is
  not batch-invariant. It dominates a training step's MoE time.

TP4 gate on the real layer-0 weights (`tp4-moe/rank-*.json`): all twelve cases
pass on all four ranks, including the round trip of all 512 experts, identical
routes on every rank, full == chunked and reordered batches at 8-1024 tokens,
and identical replicated router gradients. The TP4 block forward (routed +
shared expert + all-reduce) takes 1.9-2.4 ms at 8-1024 tokens.

## Tests

| File | Device | Covers |
| --- | --- | --- |
| `tests/models/qwen3_next/test_qwen3_next_forward_contract.py` | CPU | CPU rejection, vLLM pin, fixed-order sum/softmax row independence |
| `tests/models/qwen3_next/test_qwen3_next_tp_blocks.py` | CPU | HF shard/assemble round trips, replica drift rejection, TP4 parameter ownership |
| `tests/validation/common/test_tensor_identity.py` | CPU | raw-bit identity (signed zero, NaN/Inf, dtype) |
| `tests/models/qwen3_next/check_qwen3_next_forward.py` | CUDA + vLLM | GEMM batch/chunk/reorder, route ties, combine order, MoE VJP vs FP64, route and output batch invariance, grouped == per-expert bitwise, fail-closed TD path |
| `tools/validation/models/qwen3_next_tp_moe_check.py` | 4 x CUDA | real layer-0 weights: HF round trip, replicated routes, chunk/reorder, training forward, gradients |

The `check_` file imports vLLM, so it runs in the `Qwen3-Next-provider-GPU`
workflow rather than the default collection.

## Scope

TP4 with EP1 (all 512 experts on every rank) on CUDA B200. ROCm, EP > 1 and
other TP sizes are separate claims. Model-level parity uses this block but is
the `full_model_chain` row.
