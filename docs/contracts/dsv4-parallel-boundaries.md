# DSV4 CP ownership and PP2 boundaries (opt-in)

`Dsv4CpPlan` fixes contiguous global token ranges per CP rank. It assigns
each C128 compressed row to the owner of its **global end position**, and
window rows to the owner of their global position. The caller must first
produce complete compressed rows; this adapter never combines partial C128
groups or reconstructs the Indexer. `query_keys(rank)` emits the global
request/position coordinates consumed by the shared joint-attention metadata
contract. This contiguous layout is **not** the packed zigzag CP layout used
by some Megatron/VIME configurations; callers must select and verify the
actual framework layout before enabling this path.

`Dsv4CpKvExchange` uses the existing deterministic ROCm collective for
rank-ordered padded AllGather of int64 KV keys and BF16/FP16/FP32 values.
It validates the key owner, removes padding and returns keys/rows in canonical
logical order. The matching dKV return uses the same fixed balanced-rank
AllReduce then selects local rows in their original order. There are no
atomic gradient writes. Group, model version, phase, dtype and row layout
are checked across ranks before the data collective. Empty CP ranks and
globally empty row sets participate. No Graph capture is supported here.

`Dsv4Pp2Plan` fixes global layer ranges and stage-boundary activation shape,
dtype and model version. `Dsv4Pp2Boundary` performs cross-stage
run/microbatch/version checks, then delegates forward and backward transfers
to Megatron Core's `P2PCommunicator` supplied by the training engine. Its
CPU test transport is explicit and not a production replacement. PP4,
virtual PP and overlapping microbatches need separate ordering contracts.

## Validation and remaining integration

- Two-rank Gloo CP: global compressed/window ownership, canonical KV order,
  fixed dKV reduction/return, empty rank/all-empty, invalid owner and version.
- One-rank MI300X RCCL: C128 row and dKV data path.
- Two-rank Gloo PP2 protocol with a test P2P transport: layer boundaries,
  activation/gradient directions, stale version and wrong microbatch.
- **Not item-41 signoff:** actual verl/VIME model adapters, training CP vs
  rollout DCP, cross-TP/CP gradient paths, Megatron ROCm PP2 and two-node PP4
  remain untested/unwired. The `ljj` container exposes only one GPU.
