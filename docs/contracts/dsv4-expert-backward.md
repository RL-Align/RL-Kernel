# DSV4 expert backward from a saved forward route (opt-in)

`SavedForwardRoute` records an immutable run/model version and the exact
`(token, topk_slot, expert, destination_rank)` ordering produced by forward
expert dispatch. It rejects duplicate slots and invalid rank/token placement.
The forward owner must construct this receipt **from the route actually used**,
not by rerunning the router during backward. Recompute may supply a second
receipt, which must have the same fingerprint.

`ExpertBackwardRoundTrip` exchanges receipt metadata over the EP process group,
then sends `dy` along the saved slots via PyTorch's existing
`all_to_all_single` transport. The returned `inbound_slots` identify each
expert input. The caller computes expert dX in that order and passes it to
`return_dx`; reverse AllToAll restores the local slot ordering. Repeated
token gradients are gathered by unique `(token, topk_slot)` indices and
accumulated in ascending Top-K slot order in FP32, then cast to the input
gradient dtype. There are no atomic additions. `route_weight_gradient` returns
FP32 `dp_s` by dotting local `dy` with each **unweighted** expert output in
saved slot order. Discrete Top-K selection has no gradient.

The adapter uses a one-shot lifecycle. Every rank must enter the same phase;
control-plane handshakes reject stale run, changed forward fingerprint,
recompute drift, dtype/layout mismatch, and replay before data transport.
After failure, reconstruct from a fresh forward receipt. Empty-token ranks
participate in all collectives. ROCm requires a NCCL/RCCL process group;
CPU/Gloo is an explicit test reference.

## Scope and validation

- Two-rank Gloo: duplicate token slots, empty source rank, expert destinations,
  reverse dX mapping, FP32 route-weight gradient, stale run/fingerprint,
  recomputation drift, and replay rejection.
- One-rank MI300X RCCL: BF16 dy/dX data path and FP32 `dp_s`.
- The forward dispatch owner (KLR #81 items 35-38) and a real DSV4 expert
  autograd caller have **not** been wired. The route receipt is currently a
  boundary interface. Multi-GPU ROCm and two-node execution remain untested
  because the `ljj` container exposes a single GPU. No Graph or overlap
  guarantee is made. This is not item-39 full signoff.
