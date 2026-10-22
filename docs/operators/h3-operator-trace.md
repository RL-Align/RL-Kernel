# MiniMax-H3 Operator Trace

## Summary

`h3_operator_trace` (RFC #420, foundation) records, for every node of the fingerprinted H3
graph, what actually ran: owning RFC row, binding status, requested and actual backend,
implementation identity, dtype, accumulator, split policy, topology and fallback state.
A strict run is only evidence when this ledger is complete and clean; it is the gate the
later rows (`ws1_full_h3_transformer`, `ws2_*`, `final_*`) report through.

It is pure Python (no torch import in the core) and builds on `h3_arch_fingerprint`
(`fingerprint.build_graph()` is the vocabulary: 2 refiner blocks x 14 nodes, 50 blocks x 17
nodes, stem and tail) and on the semantic operator registry
(`rl_engine/runtime/semantic_registry.py`).

## Entry Points

```python
from rl_engine.models.minimax_h3 import operator_trace as ot

trace = ot.OperatorTrace(mode="candidate", forwards=1)
trace.record_node("transformer_blocks.0.q_proj", status="row", backend="...", dtype="bfloat16")
call = ot.trace_call_from_resolution(path, resolution=resolution, provenance=provenance)
trace.record(call)                 # registry resolution + instance provenance -> one call

trace.assert_strict()              # raises ValueError listing every violation
trace.violations(paths=..., allowed_statuses=("row",))
trace.digest; trace.dumps(); ot.OperatorTrace.loads(text)
ot.first_difference(a, b)          # first call where two traces disagree
```

`block_observer(trace, block_index=0)` (`stage="token_refiner"` for the 14-node refiner block) plugs into `h3_block.run_graph` through the existing
`ctx["node_observer"]` hook. It records each node with the binding `h3_block.NODES` declares,
so the one-block replay and the full-model replay emit a trace without any change in arithmetic.

## What `violations` Reports

| Check | Reported when |
| --- | --- |
| Binding | status is not in `allowed_statuses` (default only `row`: `interim` stand-ins and `reference` provider replays are not strict evidence) |
| Fallback | any fallback path was taken (`OperatorResolutionTrace.fallback_attempts` non-empty) |
| Dispatch | `requested_backend` is set and differs from the backend that ran |
| Split policy | anything other than `none` (split-K, stream-K, atomics) |
| Accumulator | a reduction row (GEMMs, RMSNorm, attention, AdaLN projection, projections) without an FP32 accumulator; element-wise rows must declare `none` |
| Backend | no backend recorded |
| Coverage | a node ran a different number of times than `forwards`, or a node outside the expected set ran |

`record` also fails immediately for paths outside the graph, a row that disagrees with the
graph's owner, or an unknown status. A trace serialises with a content digest
(`trace_sha256`) and the graph digest; loading rejects tampered content or a trace recorded
against a different graph.

## Today's One-Block Result

On block 0 (tiny layout, H100) the trace reports 17 calls and 9 non-`row` nodes: 7
`interim` (O, gate-up and down GEMMs, QK norm, attention, SwiGLU, ...) and 2 `reference`
(RoPE of Q and K). That list is exactly the work the unclaimed and claimed rows still owe;
`h3_operator_trace.py verify --block 0` exits non-zero until it is empty, and
`--allow row interim reference` records the current honest state.

```bash
python tools/validation/models/h3_operator_trace.py block --layout tiny --out trace.json
python tools/validation/models/h3_operator_trace.py verify trace.json --block 0
python tools/validation/models/h3_operator_trace.py diff a.json b.json
```

## Tests

`tests/models/minimax_h3/test_h3_operator_trace.py`: strict full-graph trace; each violation
class individually; coverage (missing, repeated, foreign, subset, multiple forwards);
digest stability and tamper detection; `first_difference`; the registry bridge with real
`OperatorResolution`/`OperatorInstanceProvenance`; and (CUDA) a block-0 replay whose recorded
statuses and backends equal `h3_block`'s declared bindings and are repeatable.

## Out Of Scope

The trace does not measure numerics and does not change dispatch. Accumulator and split
policy are the contract values declared per row, not hardware counters; kernels that can
change them must record the real value through `record`/`trace_call_from_resolution`.
