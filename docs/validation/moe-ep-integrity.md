# P6 EP result integrity (task 58 foundation)

`rl_engine.validation.moe.ep_integrity.audit_ep_results` checks independent expert
rows before floating-point combination. It does not register or implement a MoE
backend. The expected plan and expected values must be captured independently of
the received data; using received rows to build the oracle invalidates the check.

## Boundary

- Expected routes: int64 `[rows, 5]` containing source EP-group rank, source-local
  token, slot, expert ID, and expert-owner EP-group rank.
- Received batches: one `EPRankResults` per expert rank, including empty ranks;
  routes are int64 `[rows, 4]` without the owner column. The batch rank is the
  actual processing rank, not a destination copied from the plan.
- Expected coverage: all `sum(source_num_tokens) * top_k` valid slots, exactly
  once. Padding is excluded by the caller. Dropped slots require a different
  explicit expected-domain contract and are not silently accepted here.
- Payload: matching fp16/bf16/fp32 `[rows, hidden]`, compared as original bytes.
  This checks transport integrity, including signed zero and non-finite bit
  patterns; it is not a numerical finiteness/accuracy acceptance policy.
- Run identity is required and compared, as is the diagnostic envelope's
  `p6-ep-integrity.v1` schema. This schema is local to this checker; it does not
  assert that a P4 adapter has adopted it.
- Empty source/expert ranks are permitted; an entirely empty oracle is rejected.

The report distinguishes missing/duplicate ranks, missing/duplicate/unexpected
token-slot pairs, expert-ID errors, rank ownership, run/schema mismatch, payload
layout, and payload-byte differences. Error details include rank, source rank,
token, slot, expert, and physical row when available. Details are bounded while
the total error count is retained. Producer backend/build identity is supplied
by the producer and recorded, not independently attested by the auditor.

## Executable transport baseline

The runner performs actual `all_to_all_single` dispatch and return, using a
synthetic BF16 expert sentinel and a separately computed CPU oracle. It reverses
received row order, checks return-to-source, and covers balanced/skewed placement
and empty ranks. Transport never performs floating-point reduction.

Run **inside the container**, from the checkout root:

```bash
python3 -m pytest tests/validation/distributed/test_moe_ep_integrity.py -q
PYTHONPATH="$PWD" timeout 60s torchrun --standalone --nproc-per-node=2 \
  -m rl_engine.validation.moe.ep_integrity_runner \
  --backend gloo --output /data/outputs/task58-gloo-ep2.json
```

For AMD, first set `ROCR_VISIBLE_DEVICES` to the assigned idle GPU IDs and set
`--nproc-per-node` to that assignment's size (1, 2, 4, or 8). Remove conflicting
HIP/CUDA visibility settings if present. Then run **inside the container**:

```bash
PYTHONPATH="$PWD" timeout 60s torchrun --standalone --nproc-per-node=1 \
  -m rl_engine.validation.moe.ep_integrity_runner \
  --backend rccl --output /data/outputs/task58-rccl-ep1.json
```

RCCL mode rejects a missing explicit device mask and non-ROCm builds; it does not
fall back to Gloo. Each run exits and destroys its process group. JSON records
the actual backend, per-rank device assignment, PyTorch/HIP versions, checker and
runner source hash, slot counts, and integrity results. The counts/values are
small correctness fixtures, not performance measurements or a production model.

## Acceptance gap

At implementation time, neither this stack nor upstream `dsv4-rocm` provides the
P4 expert dispatch/return adapter for tasks 35–39. The executable is therefore a
**transport baseline**, and its reports explicitly set
`p4_integration_verified: false`. Gloo results are CPU communication evidence;
EP=1 RCCL is single-device evidence. Neither satisfies real multi-GPU P4 EP
acceptance for task 58.

To complete task 58, connect independently captured P4 route/run identity and
actual per-rank outputs to this auditor, verify all declared target EP sizes on
assigned AMD GPUs, and record simulated, transport-baseline, and P4-integration
results separately. Do not mark task 58 complete from the foundation alone.
