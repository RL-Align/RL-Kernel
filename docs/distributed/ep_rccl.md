# DSV4 Expert Parallelism on RCCL

Forward routed-expert dispatch and return for DSV4 on ROCm, using
`torch.distributed.all_to_all_single` over RCCL. Each rank routes its own tokens,
sends every Top-K slot to the rank that owns its expert, and gets one output row
per slot back. Transport moves bytes only; no floating-point merge happens in it.

The output in `(token, slot)` order does not depend on the EP size.

Forward only. Backward is task 39 (#633), LoRA execution is task 45, and the output
merge is P6.

## Modules

| Module | Role |
|---|---|
| `rl_engine/distributed/ep/plan.py` | Integer routing plan and inverse mapping (task 35) |
| `rl_engine/distributed/ep/exchange.py` | Handshake, dispatch and return (task 37) |
| `rl_engine/moe/ep_expert.py` | BF16 input, receiver quantization, expert kernels (task 38) |

## Usage

Whole layer, with the fused kernels:

```python
from rl_engine.distributed.ep import EpExchange, EpIdentity, EpPlacement
from rl_engine.moe.ep_expert import routed_expert_forward_ep, select_local_weights
from rl_engine.moe.triton_fused_mlp import TritonFusedMoeMlp

placement = EpPlacement.from_owners(expert_owners, group_size)  # 256 entries: expert -> EP rank
identity = EpIdentity(run_id, model_version, placement_version)
exchange = EpExchange(group, placement)

route = exchange.build_route(topk_experts, identity, valid=None)  # int [T, 6], one per forward
out = routed_expert_forward_ep(
    exchange,
    route,
    TritonFusedMoeMlp(),
    x,    # BF16 [T, H]
    p_s,  # FP32 [T, 6]
    select_local_weights(w1, placement, rank),
    select_local_weights(w2, placement, rank),
)
# out.rows: BF16 [S, H], with out.token_indices, out.slot_indices, out.expert_indices,
# out.mapping_run_id and out.mapping_schema_version for MoEOutputCombineInputs.
```

`rank` and `expert_owners` use EP-group ranks, not global ranks.

Transport only, with your own expert computation:

```python
route = exchange.build_route(topk_experts, identity)
x_rows = exchange.dispatch_tokens(route, x)      # expert order
p_s_rows = exchange.dispatch_slots(route, p_s)   # expert order
with exchange.guard():
    y = compute(x_rows, p_s_rows)                # keep expert order
out = exchange.return_outputs(route, y)          # (token, slot) order
```

Run any local work between exchange calls inside `exchange.guard()`. An exception
outside it leaves peers blocked in their next handshake until the process group
times out.

`EpExchange(..., strict=True)` also echoes metadata on the return trip and runs
`verify_echo` on the source. It costs one extra collective per return.

## Row order

| Order | Key | Where |
|---|---|---|
| Send | `(destination, token, slot)` | Source packing; also the order #633's receipt requires |
| Arrival | `(source, token, slot)` | Fixed by `all_to_all_single` displacements, not by timing |
| Expert | `(local expert, source, token, slot)` | What the kernels consume |
| Canonical | `(token, slot)` | Returned to the caller for P6 |

The return trip carries no metadata: rows sent back in arrival order land on the
source in send order.

## Determinism

- Plans are pure integer functions of their inputs. Sorts are stable with a unique
  final key, and CPU and GPU plans are compared in the tests.
- Payload is sent as raw BF16. The receiver keeps those bytes for the LoRA path and
  derives MXFP8 for the base weights; MX scales are local to one row.
- Results rely on the kernels' batch invariance, since row grouping changes with
  the EP size.

## Failure behavior

- Every operation starts with a fixed-size signature gather, so a failed or
  diverging rank makes all ranks raise instead of blocking in an all-to-all.
- After a failure the exchange is poisoned and refuses further work.
- Each wire row carries a 63-bit `run_tag` derived from the identity, `top_k` and
  the owner table. Rows from another run are rejected.
- The receiver rejects count mismatches, duplicates, out-of-order rows, wrong
  sources and rows for experts it does not own. A row lost together with its count
  is visible only to the source, through `strict=True`.
- A returned route cannot be reused.
- Payloads with `requires_grad` are rejected, because all-to-all would drop the graph.
- `mx_quantize` rejects non-finite input, which fails all ranks.

## Validation

```bash
python -m pytest tests/distributed/ep -q

# Transport with a stand-in kernel; output compared with a local reference
PYTHONPATH=. HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 \
  tests/distributed/ep/check_ep_exchange_rccl.py 8192 4096

# Real kernels; rank 0 compares the output with a reference that uses no exchange
PYTHONPATH=. HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 \
  tests/distributed/ep/check_ep_expert_rccl.py 2048 4096 2048
```

Use `HIP_VISIBLE_DEVICES`. `tests/distributed/ep/test_ep_plan.py` has two P6 tests
that skip unless `rl_engine/contracts/operators/moe.py` is present.

With the real kernels at T=2048, H=4096, F=2048 and rank 1 sourcing no tokens, 1, 2,
4 and 8 ranks give the same output, SHA-256 prefix `558c96fedb4ed976`.

## Benchmarks

Single node, 8 × MI300X (`gfx942`), torch 2.11.0 with ROCm 7.2, idle machine.
Timings include `torch.cuda.synchronize`. Rank 1 sources no tokens when there are
several ranks, so inbound rows per rank are not uniform.

### Full layer, real kernels

T=2048 tokens per rank, H=4096, F=2048, random weights, 5th round, range over ranks.

| Ranks | Inbound rows | `build_route` | Input + quant | Kernels | Return |
|---|---|---|---|---|---|
| 1 | 12288 | 2.0 ms | 2.1 ms | 8.8 ms | 0.5 ms |
| 2 | ~6100 | 1.9–2.1 ms | 2.5 ms | 4.4–4.5 ms | 1.5 ms |
| 4 | ~9200 | 2.5–2.6 ms | 2.4–2.5 ms | 5.8–6.0 ms | 1.1–1.3 ms |
| 8 | ~10700 | 2.0–2.1 ms | 2.2–2.3 ms | 5.9–6.2 ms | 0.8–1.1 ms |

### Transport only

T=8192 tokens per rank, H=4096, BF16 `x` only, 6th round, range over ranks.

| Ranks | Inbound rows | `build_route` | Dispatch | Return |
|---|---|---|---|---|
| 1 | 49152 | 2.0 ms | 0.9 ms | 1.0 ms |
| 2 | ~24600 | 2.0–2.3 ms | 4.7 ms | 4.6–4.8 ms |
| 4 | ~36900 | 2.1–2.4 ms | 2.7–2.8 ms | 2.7–2.9 ms |
| 8 | ~43000 | 2.3–2.5 ms | 1.8–2.0 ms | 1.9–2.1 ms |

The first round is much slower. The first collective on a new process group took
13.8–15.4 s at 8 ranks, so warm up before measuring or serving.

### Large transfers, unstaged

8 ranks, H=4096, one round per size, slowest rank. Input is arbitrary bytes, and the
round trip was compared byte for byte.

| T per rank | Data per rank | Largest pair | Result | Dispatch | Return |
|---|---|---|---|---|---|
| 8192 | 0.4 GiB | 0.05 GiB | Exact | 6 ms | 2 ms |
| 65536 | 3 GiB | 0.38 GiB | Exact | 16 ms | 14 ms |
| 262144 | 12 GiB | 1.51 GiB | Exact | 776 ms | 774 ms |
| 524288 | 24 GiB | 3.01 GiB | Exact | 1118 ms | 492 ms |
| 1048576 | 48 GiB | ~6 GiB | GPU out of memory | | |

The jump at 12 GiB has not been investigated. The memory limit includes extra
allocations made by the benchmark's own comparison.

## Staging (task 36, deferred)

Unstaged transfer stayed byte-exact past 2 GiB per pair, and GPU memory ran out
before transport did. Staging bounds a single collective but not total memory, since
the assembled output is still full size. It is needed only for a fixed-capacity
backend (task 42) or if multi-node testing calls for it.

## Limits

- Tested on one node only. Multi-node is not validated.
- Forward only. Real kernels were run without LoRA or autograd.
- `TorchAllToAll` accepts only ROCm NCCL/RCCL.
- `select_local_weights` slices weights held in full on each rank. The real layout
  comes from task 50.
- Invalid slots are absent from the output, not zero-filled. P6 rejects an incomplete
  Top-K when `num_tokens` is declared; the semantics are not settled yet.
- `saved_route_slots()` returns EP-group ranks, while #633's receipt uses global
  ranks. They differ on a stage-local group.
