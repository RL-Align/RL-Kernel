# Nemotron Nano expert-parallel transport

The CUDA `ep_all_to_all` row of
[RFC #434](https://github.com/RL-Align/RL-Kernel/issues/434): strict dispatch and return
of routed expert rows across an EP group, forward and backward. Transport moves bytes;
no floating-point merge happens on the wire. The only floating-point work here is the
per-token gradient accumulation on the source rank.

Expert kernels, the weighted combine kernel, expert weight-gradient reduction and
EP2/4/8 == EP1 at model level belong to `moe_experts_combine` and `ep_invariance`.
ROCm is a separate column.

## Interface

```python
from rl_engine.distributed.ep import EpPlacement
from rl_engine.distributed.ep.nemotron import EpInvocation, NemotronEpExchange, ep_backward

placement = EpPlacement.from_owners(expert_owners, ep_size)  # 128 entries: expert -> EP rank
exchange = NemotronEpExchange(
    placement, model_version=model_id, placement_version=layout_id, group=ep_group
)

call = exchange.dispatch(x, ids, EpInvocation(step, microbatch, layer))
y_rows = run_experts(call.x_rows, call.expert_offsets, call.local_experts)
rows = call.return_outputs(y_rows)          # unweighted, (token, slot) order
out = combine(rows, weights)                # caller's fixed-order weighted combine

ep_backward(exchange, loss, [call])         # all ranks, including ranks with no tokens
```

| Tensor | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `x` | `[T,2688]` | BF16 | Source token rows, `0 <= T <= 65536` |
| `ids` | `[T,6]` | int64 | Router output, ascending and unique per token |
| `x_rows`, `y_rows` | `[R,2688]` | BF16 | Expert order on the owning rank |
| `rows` | `[6T,2688]` | BF16 | Unweighted outputs in `(token, slot)` order |
| `x.grad` | `[T,2688]` | BF16 | Accumulated `dX` |

`x_rows` is grouped by local expert: `expert_offsets` holds the exclusive prefix counts and
`local_experts` the global id of each local expert. `call.provenance()` reports the EP size,
rank, backend, payload dtype, placement fingerprint, run identity, strict flag and that no
fallback ran.

## Row orders

| Order | Key | Where |
| --- | --- | --- |
| Send | `(destination, token, slot)` | Source packing |
| Arrival | `(source, token, slot)` | Fixed by the all-to-all displacements, not by timing |
| Expert | `(local expert, source, token, slot)` | What the experts consume |
| Canonical | `(token, slot)` | Returned to the caller |

With EP size 1 the expert order equals the router's `packed` order. `plan.py` and
`exchange.py` come from the DSV4 forward exchange (#643), which is why the orders match that
work. `exchange.py` has two changes: local type annotations, so that CI's `mypy` passes, and
every buffer a collective needs, including the strict echo, is now allocated before the
agreement that lets peers enter it (see Failure behavior). `test_ep_plan.py` is reformatted
by the repository's `black`.

## Gradient contract

`dy` follows the saved route to the owning ranks, and `dX` returns in reverse. Each token's
gradient is the FP32 sum from `+0.0` over slots 0 to 5, cast once to the dtype of `x`, the
same as the router's `_unpack_grad`. Slot order is observable under cancellation, and the
`+0.0` start is observable when all six gradients are `-0.0`. The saved-route draft #633
starts from slot 0 instead, and so gives `-0.0` there.

## Failure behavior

Autograd only runs nodes that reach the loss, and a node that does not run cannot report
anything. So checks run where all ranks are known to be:

| Situation | Behavior |
| --- | --- |
| Bad `x`, `ids` or expert output on one rank | All ranks raise before payload moves; the failing rank keeps its own message |
| Ranks disagree on invocation, placement, `strict`, or training versus `no_grad` | All ranks raise before payload moves |
| Allocation fails on one rank, including the strict echo buffer and route metadata | All ranks raise before the collective that needed it |
| Rank with no tokens, or receiving no rows | Completes normally |
| Frozen `x` | Completes: an internal anchor keeps the dispatch node, so peers still get their `dX` |
| Expert output with no gradient path | Completes, returning zero `dX` |
| A rank's output not in its loss, or no graph at all | `ep_backward` rejects on all ranks before payload moves |
| Two layers entered in different orders | Rejected before payload moves, with or without `ep_backward` |
| Replayed backward, wrong gradient shape or dtype | All ranks raise |
| Exchange during backward, including reentrant checkpoint | Rejected |

**Not detectable:** a rank that dies, or never calls `ep_backward` or the entry points, leaves
its peers waiting until the process-group timeout. A collective cannot tell a late rank from
an absent one. Calling `loss.backward()` directly skips the reachability check and has the
same exposure. Frameworks that must own the backward call can run
`ep_backward_preflight(exchange, loss, calls)` first. `EpExchange` construction is local, so
construct it on every rank before the first exchange.

## Design assumptions

The RFC cites section 4.2.1 for this row, but the section is not published. These are proposals
until the maintainers confirm or replace them:

1. **Transport:** NCCL `all_to_all_single`. The payload is bytes-only and displacements fix
   the order. A DeepEP-style kernel can replace it through the `transport` argument. No DeepEP
   baseline is reported.
2. **Porting:** `plan.py` keeps the `dsv4-rocm` logic unchanged and CUDA code lives in new
   files. `exchange.py` differs by local variable annotations, which the same `mypy` check
   would need on `dsv4-rocm`, and by one fix that applies there too: in the ported version a
   rank whose strict echo buffer or route metadata failed to allocate left its peers waiting
   in the next collective until the process-group timeout.
3. **Backward:** the saved-route design of #633, implemented separately. It is ROCm/Gloo
   only, keeps per-row Python objects (about 390k rows at `T=65536`), and has a different
   accumulation start. Its P3 dependency is optional, and is not a reason.
4. **Boundary:** this row ends at unweighted canonical rows. "Source-rank fixed-order combine"
   is read as the backward accumulation here plus an interface obligation on the forward
   combine, validated with a stand-in; the combine kernel is `moe_experts_combine`.
5. **Wire and checks:** `dy` and `dX` travel as BF16. Training requires `strict=True`, whose
   metadata echo costs one extra collective per return. `strict=False` is `no_grad` only.
6. **Routing input:** this row consumes the router's `ids` directly and depends on the
   router's provisional ascending-id order; `dispatch` rejects anything else.

The architecture fingerprint is still in progress, so geometry constants are taken from the
RFC (revision `bf77c31`).

## Limits

- Single node only; multi-node is not validated.
- Second-order gradients, CUDA Graph and non-reentrant checkpoint are unqualified.
- Bitwise claims assume the same architecture and software stack.
- Results rely on batch-invariant expert kernels, because row grouping changes with the EP size.
- The stand-in expert and combine are test code. The real kernels are not yet part of any
  run here.

## Validation

CPU, runnable without a GPU:

```bash
python -m pytest tests/distributed/ep -q
```

A real process group, comparing every rank with a local EP size 1 run:

```bash
PYTHONPATH=. torchrun --nproc-per-node=8 --master-addr=127.0.0.1 --master-port=29500 \
    tests/distributed/ep/check_ep_nemotron.py --scale
PYTHONPATH=. GLOO_SOCKET_IFNAME=lo torchrun --nproc-per-node=4 --master-addr=127.0.0.1 \
    --master-port=29500 tests/distributed/ep/check_ep_nemotron.py --backend gloo --tokens 32
```

Performance against native NCCL, run with the GPUs to itself. Both sides get the same
inputs, routing and process group, and an identity expert:

- **Forward only** (rollout), under `no_grad`: a plain `torch.distributed` baseline (route,
  pack, dispatch, return, restore `(token, slot)`, no checks) against `dispatch` plus
  `return_outputs`, with and without the strict echo. The exchange side includes the input
  checks, agreements and route building that every call runs.
- **Training:** the baseline above plus `dy` to the owners and `dX` back with the FP32 slot
  sum, against the public path (`dispatch`, `return_outputs`, `ep_backward`). Both sides
  get the same `dy`: the exchange's loss is a benchmark-only injector that hands `dy` to the
  returned rows unchanged, so no full-size multiply or reduction is timed on either side.
  The injector's node and scalar remain on the exchange side.
- **Direct** (a breakdown, not the training path): the same sequence of agreements and
  all-to-alls called on the exchange under `no_grad`. It skips the call layer's output
  checks, agrees on the backward with the route's tag rather than a digest of all calls,
  and has no autograd nodes, anchor, link, graph walk or injector.
- **Checks before timing:** outputs and `dX` of every scope equal the baseline's bit for
  bit on every rank, and the gradient reaching the exchange's rows equals `dy`.
- **Statistics:** warm-up rounds, then the first measured round reported apart and the
  median, minimum, maximum and all samples of the rest. Each round starts behind a barrier
  and each time is the slowest rank's. Phase scopes synchronize at every boundary. Peak
  allocated memory is the largest over ranks.

```bash
PYTHONPATH=. torchrun --nproc-per-node=8 --master-addr=127.0.0.1 --master-port=29500 \
    benchmarks/distributed/benchmark_ep_all_to_all.py --output ep_all_to_all_h20.json
```

### Results

| Check | Environment | Result |
| --- | --- | --- |
| `tests/distributed/ep` | CPU, threaded transport | 167 passed, 3 skipped (1 needs a GPU; 2 need `contracts/operators/moe.py`, absent on this branch) |
| Coordinated failure (mixed `strict`; allocation failure at every allocation point, strict and not, and in route and echo metadata) | CPU, threaded transport with a 2 s collective timeout | Every rank raises the agreed error, no rank succeeds, nothing waits for the timeout. Reverting either fix makes these tests fail |
| B3b full chain: router weights computed from `x`, so `x.grad` sums the payload branch through the exchange and the routing branch | CPU, EP 1/2/4/8, stand-in expert and combine | Per-slot `dy` and routing-weight gradients bitwise equal to the router's own single-device packing; those two and total `x.grad` bitwise equal across EP sizes; total `x.grad` within BF16 rounding (not bitwise) of that reference and of an independent FP64 chain, while dropping the routing branch misses FP64 by over 8x the tolerance |
| B3c leaf routing weights | CPU, EP 2/4/8 | Bitwise equal to EP 1, and equal to an FP64 closed form within `rtol=1e-5`, `atol=1e-2` |
| `check_ep_nemotron.py --backend gloo --tokens 32` | CPU, real processes, EP 1/2/4/8 | All cases pass; router cases skipped. `--scale` also passes at EP 2 |
| pre-commit hooks (`black` 24.4.2, `isort` 5.13.2, `flake8` 7.0.0) and CI `mypy` (Python 3.10, no torch) | CPU | Clean on every changed file; `mypy rl_engine/` reports no issues |
| `check_ep_nemotron.py --scale` (NCCL; the distributed cases split 2048 tokens over the whole group, so 256 per rank at EP 8, and the single-source case puts all 2048 on one rank) | 8 x H20-3e, EP 1/2/4/8 on 1/2/4/8 GPUs, GPUs to itself | All cases pass on every rank (13 at EP 1, 17 at EP 2/4/8), outputs and gradients bitwise equal to that rank's local EP size 1 run: contiguous and shuffled placements, empty source, one owner, one source, batch and position, arbitrary bit patterns, stacked layers, one rank with a wrong dtype, unused output rejected, reversed layers rejected, and F5, B3a, F6 and F6b below |
| Router comparison (F5 packed payload, B3a payload-branch gradient) | H20-3e, SM90 Triton router on every rank, 2048 tokens each, against that rank's EP size 1 loopback | Bitwise equal. This compares the payload branch only, not a multi-GPU router and MoE chain |
| F6 largest legal load: 65536 tokens from every rank, all six slots to experts on rank 0 | 8 x H20-3e, EP 1/2/4/8 | Every arriving row equals its regenerated source, and the returned rows are bitwise equal to the rows sent. 2.11 GB per source, 16.9 GB into the receiver at EP 8; peak allocated memory, largest rank: 8.9 / 13.2 / 21.7 / 38.8 GB at EP 1/2/4/8 |
| F6b one pair above 2 GiB (2,147,507,712 bytes) | 8 x H20-3e, EP 2/4/8 | Arrives equal to the per-row, per-column pattern in a sentinel-filled buffer |
| Performance | 8 x H20-3e, GPUs to itself | See below |

Two earlier NCCL runs used older code. The first failed in the check script itself: its EP
size 1 reference used the device `cuda` rather than `cuda:0`, which the exchange's exact
device check rejects. Every rank raised together and nothing hung. The second passed the
then-current cases while sharing the GPUs with a utilization filler. The results above come
from one later run on the final code, with the source hashes recorded, the filler stopped,
and the GPU process list sampled every 2 s with any process not started by the job stopping
the run; none appeared. Sampling cannot exclude interference shorter than 2 s.

Two kinds of comparison appear above. **Across EP sizes the comparison is bitwise**: every
NCCL row, the router comparison, and B3b, where per-slot `dy`, routing-weight gradients and
total `x.grad` at EP 2/4/8 have the same bits as at EP 1. **Against a different computation
the comparison is within tolerance**: B2's `dX` against an FP64 closed form (`rtol=0.05`,
`atol=0.05`, covering the BF16 combine, expert and accumulation); B3b's total `x.grad`
against the single-device reference and against FP64 (`rtol=2**-7`, `atol=2**-6`, BF16
rounding), because the reference sums the payload branch in `index_select`'s backward,
whose rounding points are not specified; and B3c's routing-weight gradients against an FP64
closed form (`rtol=1e-5`, `atol=1e-2`). B3b's per-slot `dy` and
routing-weight gradients are also bitwise equal to the single-device reference.

All GPU evidence here is operator-level validation on one node with 8 x H20-3e. It makes no
model-level claim and says nothing about multi-node runs.

### Performance

Run `ep-a2a-excl-002` on 8 x H20-3e with the GPUs to itself (torch 2.9.1+cu128, NCCL
2.27.5): median of 50 rounds after 10 warm-up rounds, in milliseconds, slowest rank per
round. Every rank sends the same number of tokens, routed uniformly at random over 128
experts with a contiguous placement. The shapes are synthetic decode (1 to 256 tokens per
rank) and prefill (2048 to 32768) sizes, not a real serving configuration, so these are
operator timings, not end-to-end latency or request throughput. Every pre-timing bit check
passed, and no foreign GPU process or failed query appeared.

The exchange is slower than the native baseline at every size.

**Forward only.** The strict exchange takes 5.96 to 6.65 times the baseline's time at 1 to
256 tokens per rank, 2.96 to 3.33 times at 2048, 1.81 to 2.07 times at 8192 and 1.26 to 1.34
times at 32768. With `strict=False` it takes 5.18 to 5.77 and 1.20 to 1.28 times at the
same ends. The strict echo adds 0.31 to 0.57 ms.

| EP | Tokens per rank | Payload per rank | Transport | Native | Exchange, strict | Exchange, `strict=False` | Strict / native time | `strict=False` / native time | Strict echo | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 0.0308 MiB | 0.04 | 0.43 | 2.74 | 2.35 | 6.44 | 5.53 | 0.39 | < 0.01 GB |
| 2 | 16 | 0.492 MiB | 0.03 | 0.42 | 2.77 | 2.39 | 6.65 | 5.74 | 0.38 | < 0.01 GB |
| 2 | 64 | 1.97 MiB | 0.04 | 0.43 | 2.80 | 2.41 | 6.55 | 5.64 | 0.39 | 0.02 GB |
| 2 | 256 | 7.88 MiB | 0.05 | 0.44 | 2.80 | 2.41 | 6.39 | 5.52 | 0.38 | 0.06 GB |
| 2 | 2048 | 63 MiB | 0.16 | 1.00 | 3.32 | 2.95 | 3.33 | 2.95 | 0.37 | 0.48 GB |
| 2 | 8192 | 252 MiB | 0.49 | 2.28 | 4.73 | 4.31 | 2.07 | 1.89 | 0.42 | 1.91 GB |
| 2 | 32768 | 1008 MiB | 1.78 | 7.36 | 9.87 | 9.43 | 1.34 | 1.28 | 0.44 | 7.63 GB |
| 4 | 1 | 0.0308 MiB | 0.04 | 0.44 | 2.78 | 2.41 | 6.38 | 5.53 | 0.37 | < 0.01 GB |
| 4 | 16 | 0.492 MiB | 0.04 | 0.43 | 2.84 | 2.48 | 6.61 | 5.77 | 0.36 | < 0.01 GB |
| 4 | 64 | 1.97 MiB | 0.04 | 0.45 | 2.82 | 2.43 | 6.23 | 5.38 | 0.38 | 0.02 GB |
| 4 | 256 | 7.88 MiB | 0.06 | 0.46 | 2.82 | 2.46 | 6.19 | 5.40 | 0.36 | 0.06 GB |
| 4 | 2048 | 63 MiB | 0.22 | 1.12 | 3.51 | 3.12 | 3.15 | 2.80 | 0.39 | 0.48 GB |
| 4 | 8192 | 252 MiB | 0.73 | 2.78 | 5.27 | 4.87 | 1.90 | 1.75 | 0.40 | 1.91 GB |
| 4 | 32768 | 1008 MiB | 2.89 | 9.61 | 12.23 | 11.79 | 1.27 | 1.23 | 0.44 | 7.63 GB |
| 8 | 1 | 0.0308 MiB | 0.04 | 0.45 | 2.77 | 2.35 | 6.11 | 5.18 | 0.42 | < 0.01 GB |
| 8 | 16 | 0.492 MiB | 0.04 | 0.43 | 2.75 | 2.44 | 6.38 | 5.66 | 0.31 | < 0.01 GB |
| 8 | 64 | 1.97 MiB | 0.05 | 0.45 | 2.80 | 2.43 | 6.23 | 5.40 | 0.37 | 0.02 GB |
| 8 | 256 | 7.88 MiB | 0.07 | 0.47 | 2.80 | 2.43 | 5.96 | 5.18 | 0.37 | 0.06 GB |
| 8 | 2048 | 63 MiB | 0.25 | 1.18 | 3.50 | 3.12 | 2.96 | 2.63 | 0.38 | 0.48 GB |
| 8 | 8192 | 252 MiB | 0.83 | 3.05 | 5.51 | 5.08 | 1.81 | 1.67 | 0.42 | 1.91 GB |
| 8 | 32768 | 1008 MiB | 3.23 | 10.41 | 13.09 | 12.52 | 1.26 | 1.20 | 0.57 | 7.63 GB |

**Training (forward plus backward).** The public path takes 3.36 to 6.39 times the
baseline's time at 64 and 1024 tokens per rank, 1.63 to 1.72 times at 8192 and 1.26 to 1.33
times at 32768.

| EP | Tokens per rank | Payload per rank | Transport | Native | Exchange, public path | Exchange / native time | Direct | Direct / native time | Public − direct | `build_route` | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 1.97 MiB | 0.04 | 0.66 | 3.83 | 5.78 | 3.40 | 5.13 | 0.43 | 1.68 | 0.02 GB |
| 2 | 1024 | 31.5 MiB | 0.10 | 1.21 | 4.35 | 3.59 | 4.01 | 3.31 | 0.33 | 1.85 | 0.39 GB |
| 2 | 8192 | 252 MiB | 0.49 | 5.21 | 8.98 | 1.72 | 8.83 | 1.69 | 0.15 | 1.95 | 3.09 GB |
| 2 | 32768 | 1008 MiB | 1.78 | 18.64 | 24.79 | 1.33 | 24.55 | 1.32 | 0.25 | 2.02 | 12.36 GB |
| 4 | 64 | 1.97 MiB | 0.04 | 0.68 | 3.88 | 5.72 | 3.43 | 5.05 | 0.46 | 1.66 | 0.02 GB |
| 4 | 1024 | 31.5 MiB | 0.13 | 1.33 | 4.46 | 3.36 | 4.14 | 3.12 | 0.31 | 1.85 | 0.39 GB |
| 4 | 8192 | 252 MiB | 0.73 | 6.20 | 10.10 | 1.63 | 9.86 | 1.59 | 0.24 | 1.97 | 3.09 GB |
| 4 | 32768 | 1008 MiB | 2.89 | 23.15 | 29.47 | 1.27 | 29.14 | 1.26 | 0.33 | 2.04 | 12.37 GB |
| 8 | 64 | 1.97 MiB | 0.05 | 0.69 | 4.38 | 6.39 | 3.45 | 5.04 | 0.93 | 1.66 | 0.02 GB |
| 8 | 1024 | 31.5 MiB | 0.15 | 1.39 | 4.88 | 3.51 | 4.17 | 3.01 | 0.70 | 1.84 | 0.39 GB |
| 8 | 8192 | 252 MiB | 0.83 | 6.74 | 11.09 | 1.64 | 10.38 | 1.54 | 0.70 | 1.97 | 3.09 GB |
| 8 | 32768 | 1008 MiB | 3.23 | 24.75 | 31.29 | 1.26 | 30.95 | 1.25 | 0.34 | 2.19 | 12.36 GB |

"Transport" is one `all_to_all_single` of the payload. "Strict echo" is strict forward minus
forward with `strict=False`. "Direct" compares the direct implementation with the
hand-written baseline, extra checks, metadata and allocation included. "Public − direct" is
a difference of two medians: an approximate increment of the public path (autograd, call
layer checks, graph and signature preparation, the injector) over the direct one, not a
measurement of autograd alone; it came out between 0.15 and 0.93 ms.

Small batches pay a nearly fixed cost: 2.3 to 2.4 ms more than the baseline in the forward,
where `build_route` alone takes 1.7 ms and the agreement before each step synchronizes with
the host. At 32768 tokens the largest per-phase gap in training is the `dy` trip, 6.7 to 8.3
ms against 3.4 to 5.0 ms for the baseline; the backward agreement (preflight) costs 0.27 to
0.33 ms. Per-phase times and every sample are in the JSON the benchmark writes.

Spread: from 2048 tokens up, the first measured round was within 15% of the median and no
exchange sample exceeded 1.91 times its median; the baseline's forward had 5 of 450 samples
above twice its median. Below 2048 tokens, where medians are under 5 ms, up to 5% of a
scope's samples exceeded twice its median (the baseline's slowest sample was 8.0 times its
median, the exchange's 3.2 times), and the first round reached up to 2.3 times the median.

An earlier run (`ep-a2a-excl-001`) used a benchmark whose exchange side also timed an
identity multiply, a loss multiply and reduction, and their backward, and had no
forward-only baseline. Its table is superseded by the one above.

F6 sends the largest legal load from every rank to one receiver, which checks each arriving
row against the regenerated source before returning it. F6b moves one pair just above 2 GiB
with a pattern distinct per row and column into a buffer pre-filled with a sentinel, so a
short or shifted copy fails.
