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
Construction validates the local CP plan and group first, then exchanges
validation status and the plan fingerprint across ranks before exposing a
usable adapter. A rank-local configuration error therefore fails peers too,
instead of leaving them blocked in the first data collective. The transport
constructor is supplied by the shared collective contract (#632).

`Dsv4Pp2Plan` fixes global layer ranges and stage-boundary activation shape,
dtype and model version. `Dsv4Pp2Boundary` performs cross-stage
run/microbatch/version checks, then delegates forward and backward transfers
to Megatron Core's `P2PCommunicator` supplied by the training engine. Its
CPU test transport is explicit and not a production replacement. PP4,
virtual PP and overlapping microbatches need separate ordering contracts.
Construction checks the local plan, communicator configuration and supplied
transport, then performs a rank-wide validation/fingerprint exchange. All
ranks reject a rank-local configuration error before any activation transfer.

## Validation and remaining integration

- Two-rank Gloo CP: global compressed/window ownership, canonical KV order,
  fixed dKV reduction/return, empty rank/all-empty, invalid owner and version.
- One-rank MI300X RCCL: C128 row and dKV data path.
- Two-rank MI300X RCCL: canonical KV order, BF16 dKV reduction, empty CP
  rank and globally empty rows. Override the container's one-card default
  with `ROCR_VISIBLE_DEVICES=0,1` for the isolated pytest process.
- Two-rank Gloo PP2 protocol with a test P2P transport: layer boundaries,
  activation/gradient directions, stale version, wrong microbatch and
  rank-local constructor errors without peer hangs. CP has the matching
  rank-local constructor rejection test.
- Two-rank MI300X RCCL PP2 with the installed Megatron Core
  `P2PCommunicator`: real activation and gradient transfers, plus a
  cross-rank stale-version and rank-local communicator configuration rejection
  before transfer.
- Four-rank MI300X RCCL CP2 x PP2: two CP groups and two PP lanes use the
  existing fixed dKV collective and Megatron Core stage transfer together.
  Both stages reproduce canonical KV bytes; PP backward gradients return to
  each CP source in fixed order. This tests group composition, not a model step.
- **Not item-41 signoff:** actual verl/VIME model adapters, training CP vs
  rollout DCP, cross-TP/CP gradient paths, end-to-end PP2 training and two-node PP4
  remain untested/unwired. The container has access to eight MI300X cards when
  the process visibility mask is overridden.
