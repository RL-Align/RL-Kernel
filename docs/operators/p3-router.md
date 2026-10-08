# P3 Router Development Interfaces

The P3 T01 start kit provides deterministic Router references, a real SM90/H100
Top-K provider and versioned fixtures for independent operator development.
See the [start-kit guide](../design/p3-start-kit.md) for setup, artifacts, versions
and owner handoff.

## Interfaces

Every entry accepts `P3OpCtxHost` and returns `P3OpResult`. Inputs use FP32 except
INT64 input token IDs and INT32 expert tables/IDs. Shapes are T tokens, 256 experts
and six ordered slots.

| Interface | Payload | Production owner |
| --- | --- | --- |
| `router_sqrt_softplus_fwd` | scores and raw saved score | T02 |
| `router_sqrt_softplus_bwd` | logits gradient from sealed score | T02 |
| `hash_route_fwd` | table-ordered IDs, weights and raw saved route | T03 |
| `stable_topk6_fwd` | IDs ordered by descending score, ascending expert ID | T01 |
| `learned_route_fwd` | post-bias selection, pre-bias weights and saved route | T04 |
| `hash_route_bwd` / `learned_route_bwd` | score gradients from sealed route | T06 |

The CPU implementations are exported from `rl_engine.p3.reference`. The actual
CUDA Top-K implementation is `rl_engine.p3.stable_topk6.CudaTopKProvider`; it
requires a persistent `InvocationAllocator`, full contiguous FP32[T,256] scores
and H100. It synchronizes status/echo readback before returning.

This package does not register a production router in the global dispatcher.
Unsupported backends return a capability verdict rather than a CPU fallback.
All non-PASS payloads are absent. Zero-active calls launch nothing; backward
requires typed, checksum-validated sealed state.

## Verification

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests/p3 -q
P3_RUN_GPU=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests/p3 -q -ra
python scripts/generate_p3_fixtures.py --check
```

Coverage includes repeat/launch/padding invariance, exact and near ties, bias
precision, fixed reductions, duplicate gradients, typed failures, durable IDs and
artifact verification. CPU tests skip explicit GPU slices; H100 opt-in requires
those slices to execute. Numerical golden data is stored as compressed JSON and
retains raw FP32 bytes.

## Limits

The supported profile is CUDA SM90/H100 with dropless routing. Other production
operators, finite capacity, multi-GPU WS2, real Megatron/Miles L3b, approved
Foundation bindings and performance certification remain separate deliveries.
