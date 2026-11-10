# Nemotron Nano expert-parallel transport

The CUDA `ep_all_to_all` row of
[RFC #434](https://github.com/RL-Align/RL-Kernel/issues/434): strict dispatch and return
of routed expert rows across an EP group, forward only. Transport moves bytes; no
floating-point merge happens on the wire. The backward exchange, which adds the
fixed-order gradient accumulation on the source rank, and the performance comparison
follow in a separate change.

Expert kernels, the weighted combine kernel, expert weight-gradient reduction and
EP2/4/8 == EP1 at model level belong to `moe_experts_combine` and `ep_invariance`.
ROCm is a separate column.

## Interface

```python
from rl_engine.distributed.ep import EpPlacement
from rl_engine.distributed.ep.nemotron import EpInvocation, NemotronEpExchange

placement = EpPlacement.from_owners(expert_owners, ep_size)  # 128 entries: expert -> EP rank
exchange = NemotronEpExchange(
    placement, model_version=model_id, placement_version=layout_id, group=ep_group
)

with torch.no_grad():                       # all ranks, including ranks with no tokens
    call = exchange.dispatch(x, ids, EpInvocation(step, microbatch, layer))
    y_rows = run_experts(call.x_rows, call.expert_offsets, call.local_experts)
    rows = call.return_outputs(y_rows)      # unweighted, (token, slot) order
    out = combine(rows, weights)            # caller's fixed-order weighted combine
```

| Tensor | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `x` | `[T,2688]` | BF16 | Source token rows, `0 <= T <= 65536` |
| `ids` | `[T,6]` | int64 | Router output, ascending and unique per token |
| `x_rows`, `y_rows` | `[R,2688]` | BF16 | Expert order on the owning rank |
| `rows` | `[6T,2688]` | BF16 | Unweighted outputs in `(token, slot)` order |

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

## Failure behavior

Every check runs where all ranks are known to be, before the collective it protects:

| Situation | Behavior |
| --- | --- |
| Bad `x`, `ids` or expert output on one rank | All ranks raise before payload moves; the failing rank keeps its own message |
| Ranks disagree on invocation, placement or `strict` | All ranks raise before payload moves |
| Grad mode enabled on any rank | All ranks raise before payload moves; the backward exchange is not part of this change |
| Allocation fails on one rank, including the strict echo buffer and route metadata | All ranks raise before the collective that needed it |
| Rank with no tokens, or receiving no rows | Completes normally |
| Two layers entered in different orders | Rejected before payload moves |
| A route reused after its outputs returned | All ranks raise |

**Not detectable:** a rank that dies, or never calls the entry points, leaves its peers
waiting until the process-group timeout. A collective cannot tell a late rank from an absent
one. `EpExchange` construction is local, so construct it on every rank before the first
exchange.

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
3. **Boundary:** this row ends at unweighted canonical rows. "Source-rank fixed-order combine"
   is read as an interface obligation on the forward combine here, validated with a stand-in,
   plus the backward accumulation in the follow-up; the combine kernel is
   `moe_experts_combine`.
4. **Checks:** `strict=True` is the default and echoes each row's identity back with the
   return, which costs one extra collective per return; `strict=False` skips the echo.
5. **Routing input:** this row consumes the router's `ids` directly and depends on the
   router's provisional ascending-id order; `dispatch` rejects anything else.

The architecture fingerprint is still in progress, so geometry constants are taken from the
RFC (revision `bf77c31`).

## Limits

- Single node only; multi-node is not validated.
- Forward only; grad mode is rejected. CUDA Graph is unqualified.
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

### Results

| Check | Environment | Result |
| --- | --- | --- |
| `tests/distributed/ep` | CPU, threaded transport | 124 passed, 3 skipped (1 needs a GPU; 2 need `contracts/operators/moe.py`, absent on this branch) |
| Coordinated failure (mixed `strict`; allocation failure at every allocation point, strict and not, and in route and echo metadata) | CPU, threaded transport with a 2 s collective timeout | Every rank raises the agreed error, no rank succeeds, nothing waits for the timeout. Reverting either fix makes these tests fail |
| `check_ep_nemotron.py --backend gloo --tokens 32` | CPU, real processes, EP 4 | All cases pass; router case skipped |
| pre-commit hooks (`black` 24.4.2, `isort` 5.13.2, `flake8` 7.0.0) and CI `mypy` (Python 3.10, no torch) | CPU | Clean on every changed file; `mypy rl_engine/` reports no issues |
| `check_ep_nemotron.py --scale` (NCCL; the distributed cases split 2048 tokens over the whole group, so 256 per rank at EP 8, and the single-source case puts all 2048 on one rank) | 8 x H20-3e, EP 1/2/4/8 on 1/2/4/8 GPUs, sharing them with a utilization filler | All cases pass on every rank (10 at EP 1, 12 at EP 2/4/8): output bitwise equal to that rank's local EP size 1 run for contiguous and shuffled placements, empty source, one owner, one source, batch and position, and arbitrary bit patterns; one rank with a wrong dtype rejected on all ranks; and F5, F6 and F6b below |
| Router comparison (F5 packed payload) | H20-3e, SM90 Triton router on every rank, 2048 tokens each, against that rank's EP size 1 loopback | Bitwise equal. This compares the packed payload only, not a multi-GPU router and MoE chain |
| F6 largest legal load: 65536 tokens from every rank, all six slots to experts on rank 0 | 8 x H20-3e, EP 1/2/4/8 | Every arriving row equals its regenerated source, and the returned rows are bitwise equal to the rows sent. 2.11 GB per source, 16.9 GB into the receiver at EP 8; peak allocated memory, largest rank: 8.9 / 13.2 / 21.7 / 38.8 GB at EP 1/2/4/8 |
| F6b one pair above 2 GiB (2,147,507,712 bytes) | 8 x H20-3e, EP 2/4/8 | Arrives equal to the per-row, per-column pattern in a sentinel-filled buffer |

Every comparison in this change is exact: "bitwise" means equal bits against the local EP
size 1 run or against a regenerated source, and no case uses a tolerance.

The NCCL rows come from one run of this change's own code and check script, with the commit
and source hashes recorded. The run checked correctness only, so the GPUs were shared with
a utilization filler's matrix-multiply workers (about 0.6 GB per GPU); no timing is reported
here. The follow-up change ran the same exchange code with the GPUs to itself, with
gradients enabled in most cases, and got the same results, including the same F6 peak
memory.

All GPU evidence here is operator-level validation on one node with 8 x H20-3e. It makes no
model-level claim and says nothing about multi-node runs.

F6 sends the largest legal load from every rank to one receiver, which checks each arriving
row against the regenerated source before returning it. F6b moves one pair just above 2 GiB
with a pattern distinct per row and column into a buffer pre-filled with a sentinel, so a
short or shifted copy fails.
