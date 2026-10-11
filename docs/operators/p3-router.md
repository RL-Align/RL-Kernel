# P3 Router Development Interfaces

The P3 T01 start kit provides deterministic Router references, a real SM90/H100
Top-K provider and versioned fixtures for independent operator development.
See the [start-kit guide](../archive/design/p3-start-kit.md) for setup, artifacts, versions
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

## T05 assembly and saved state

`rl_engine.p3.assembler.assemble(case, score_result, route_result, ...)` remains
the sole `assembler_abi.v1` entry point. It consumes completed operator results;
it does not rerun score transforms or Top-K. Validation binds saved logits to the
case/round policy, saved scores and selected pre-bias values to the forward
outputs, and weights to the saved fixed-tree normalization. Hash table slots,
including duplicate experts, retain their original order.

Core/Envelope construction and field checks use the ordered schemas and field
types in `contract.py`. FP32 values retain their raw bytes. Padding is canonical
in the plan, excluded from semantic fingerprints, and included in artifact
fingerprints. Raw saved padding is preserved and covered by its checksum.
Runtime stream handles are archived but excluded from artifact identity.

After assembly, call
`seal_forward_saved(case, bundle, score_result, route_result, score_ctx, route_ctx)`.
Both contexts must identify the assembled run/engine/rank/attempt and artifact,
retain the forward mask, and contain distinct completed forward invocation IDs.
The returned `score` and `route` values are typed sealed objects. The recorded
development path uses the same entry point. Zero-active recordings have no plan
or saved state.

`CombinePlanSeed` is deterministically rebuilt from Core in token order, retaining
slots 0 through 5. Serialized bundles can be read in a separate process with the
existing `serialization.decode` and `boundary.consume` entry points.

### Additive WS2 development delta

The default remains `RoutePlanEnvelope.v1`; frozen v23 publications and numerical
goldens are unchanged. An explicit `envelope_version=ENVELOPE_SCHEMA_V2` selects
`RoutePlanEnvelope.v2`, which appends `token_ownership` and
`expected_token_set_fingerprint` before the artifact fingerprint. The full rank
token set is stored once in the plan rather than repeated in every slot.
Core v1, operator ABI v4 and assembler ABI v1
remain unchanged. `contract.envelope_fields(version)` supplies the ordered fields.

Supply `runtime` with `group`, `topology`, `token_ownership` and
`expected_token_set`. Ownership declares TP replica/sequence partition, SP
sequence partition, CP/DP partition, PP layer replica and EP as a non-token
dimension. T07/T08 provide the declaration and rank token set; the assembler
validates them without implementing a partition planner or collective.

`boundary.consume` accepts `expected_runtime` and `expected_placement_version`
for checking the consumer's independent configuration. Missing, unauthorized or
duplicate tokens, stale placements, slot drift and schema/type mismatches fail
closed. Changing placement, packing, padding or execution identity preserves
per-token Core bytes and semantic fingerprints while the artifact records the
configuration changes.

This additive delta is local development support for T07/T08 and P4/P6/P7. T01
review is required before formal WS2 publication; it does not certify Foundation
bindings, real engine traces or multi-GPU execution.

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
