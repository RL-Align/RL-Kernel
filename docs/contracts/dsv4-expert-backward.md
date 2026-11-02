# DSV4 expert backward from a saved forward route (opt-in)

`SavedForwardRoute` records an immutable run/model version and the exact
`(token, topk_slot, expert, destination_rank)` ordering produced by forward
expert dispatch. It rejects duplicate slots and invalid rank/token placement.
`from_p3_forward` consumes P3 T05's validated RoutePlan and sealed forward
route state, plus an explicit physical-expert-to-EP-rank placement. It checks
the saved invocation, artifact fingerprint, Envelope v2 token ownership and
placement version before constructing the receipt. This entry point requires
P3 #565, which is not in `main` yet. It never reruns the router during
backward. Recompute may supply a second receipt, which must have the same
fingerprint; changed route weights/artifact identity are rejected.
`from_p4_forward` can also freeze the **returned** P4 #643 EP send plan
(now present on `dsv4-rocm`),
but only when `EpExchange(strict=True)` verified the forward echo. It checks
the source/rank mapping, run tag and expert placement without rerunning Top-K.
An optional P3 sealed receipt must match every route slot and version; if
supplied, its stronger forward artifact fingerprint is retained.

`ExpertBackwardRoundTrip` coordinates rank-local receipt validation before
exchanging receipt metadata over the EP process group. A mismatched local group
or saved route rejects every rank before dy transport, preserving the original
error on the failing rank. It then sends `dy` along the saved slots via PyTorch's existing
`all_to_all_single` transport. The returned `inbound_slots` identify each
expert input. The caller computes expert dX in that order and passes it to
`return_dx`; reverse AllToAll restores the local slot ordering. Repeated
token gradients are gathered by unique `(token, topk_slot)` indices and
accumulated in ascending Top-K slot order in FP32, then cast to the input
gradient dtype. There are no atomic additions. `route_weight_gradient` returns
FP32 `dp_s` by dotting local `dy` with each **unweighted** expert output in
saved slot order. Discrete Top-K selection has no gradient.

The adapter uses a one-shot lifecycle. Every rank must enter the same phase;
control-plane handshakes reject stale run, changed forward fingerprint or
placement version,
recompute drift, dtype/layout mismatch, and replay before data transport.
After failure, reconstruct from a fresh forward receipt. Empty-token ranks
participate in all collectives. ROCm requires a NCCL/RCCL process group;
CPU/Gloo is an explicit test reference.

## Scope and validation

- Two-rank Gloo: duplicate token slots, empty source rank, expert destinations,
  reverse dX mapping, FP32 route-weight gradient, stale run/fingerprint,
  recomputation drift, replay rejection, and rank-local constructor failure.
- One-rank MI300X RCCL: BF16 dy/dX data path and FP32 `dp_s`.
- Two-rank MI300X RCCL: strict P4 forward return through saved dy/dX,
  including an empty source rank. P4 exchange and fused expert forward also
  passed on eight MI300X ranks against a no-exchange reference.
- Four-rank MI300X RCCL EP4: strict returned P4 route through saved dy/dX,
  exact source/expert ownership, fixed-slot dX and FP32 route-weight gradient;
  one rank has no source tokens and another receives no expert tokens.
- A real DSV4 expert autograd caller remains unwired; this is still a boundary
  interface. EP4 end-to-end training, wider backward, two-node execution,
  Graph and overlap remain untested. The container can use eight MI300X cards
  with a per-process visibility override.
  This is not item-39 full signoff.
