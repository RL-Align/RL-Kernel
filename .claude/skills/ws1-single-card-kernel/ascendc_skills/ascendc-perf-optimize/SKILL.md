---
name: ascendc-perf-optimize
description: WS1 Ascend C operator performance optimization (single card). When profiling shows an op is slow or an optimization strategy is needed - first build the theoretical tiling model, then run the single-core pipeline bound diagnosis (memory/vec/scalar), and speed things up with UB fill-up, double buffering, repeatTimes tuning; optimizations must never change the reduction order (batch invariance is preserved). Trigger on performance optimization, slow op, bound diagnosis, pipeline analysis, tiling correction.
---

# WS1 Ascend C Operator Performance Optimization (single card)

> Sub-skill of `ws1-single-card-kernel`, curated from cannbot-skills
> `ascendc-perf-optimize` / `ascendc-performance-best-practices`
> (https://gitcode.com/cann/cannbot-skills). This repo is single-card kernel
> work, so the inter-card / inter-core pipeline steps are skipped and the
> flow reduces to Step 1 + Step 4.

## Constraint #1 in this repo: determinism outranks performance

No optimization may **change the reduction order or the
instruction-sequence-vs-shape relationship** — the row-granular split
(`row += GetBlockNum()`) is the foundation of batch invariance and must not
be reshaped into multi-row blocks or dynamic work distribution for load
balancing; the invariance assertions in `tests/test_<op>_ascend.py` must keep
passing with `torch.equal`. Correct first, fast second.

## Optimization flow (two steps on a single card)

### Step 1 — theoretical tiling model

- Input: operator class, shape, dtype, kernel pseudocode.
- Output, the ideal tiling data:
  - [ ] multi-core split (row-granular; blockNum = min(N, 128))
  - [ ] UB split (tile-size formula, chunk count)
  - [ ] buffer plan (purpose/size per buffer; reuse + fill)
  - [ ] branch coverage (fp32/bf16/fp16, alignment tails)

### Step 4 — single-core pipeline optimization (bound diagnosis)

Classify the bound from msprof profiling data, then pick the strategy:

| Bound | Signature | Levers |
|-------|-----------|--------|
| memory bound | MTE2/MTE3 dominate, V waits | bigger tiles, double buffering, merged DataCopy, UB fill-up |
| vec bound | V dominates | wider per-op vectors (32B aligned), drop redundant Casts, trade fp32 compute against halved bandwidth |
| scalar bound | S dominates (loop control / scalar math) | fewer loop levels, vectorize scalar math (8-element workaround), unroll inner loops |
| no bound | nothing saturated | are all cores used? blockNum far below physical cores |

Deliverables: simulation-chart reading + profiling report + corrections to
the Step 1 tiling.

## Levers, ordered by value

1. **UB fill-up and reuse**: size tiles against the **usable** UB budget —
   `GetCoreMemSize(UB)` reports raw hardware capacity and some architectures
   reserve part of it for compiler/API use, so validate the allocatable size
   on the target arch (include all live buffers) and leave headroom; steps
   with non-overlapping lifetimes share one `TBuf`; drive utilization
   (allocation / validated budget) as high as that budget allows.
2. **Double buffering**: `BUFFER_NUM=2` overlaps MTE2 with V; mind the ×2
   tiling threshold and the tail-tile logic (no underflow at loop=0).
3. **Merged movement**: merge only rows the SAME block already owns
   end-to-end into one `DataCopyPad` — row ownership and the shape-only
   instruction sequence must stay intact (never re-bucket rows across
   blocks; that breaks the `torch.equal` batch-invariance check, see
   Constraint #1); use the stride parameters for strided copies instead of
   per-row loops.
4. **Saturate repeatTimes**: a single repeat is capped at 255 but should be
   as large as possible; set mask/stride so one instruction covers a whole
   tile.
5. **Fewer Casts**: remove only **provably redundant** casts — a
   promote/demote pair with no arithmetic in between (bit-identical under
   `CAST_RINT`), or promoting an input that is already fp32. Collapsing
   per-step demotions into one final demotion moves the rounding points and
   therefore changes bits. Guardrails:
   - copy/lookup ops (bitwise vs golden) must not have their casts touched
     at all;
   - keep the final output quantization point (demote back to the input
     dtype at the same semantic boundary) and `CAST_RINT` (CUDA
     `static_cast` alignment);
   - for reductions, re-validate against the golden at the
     `tolerance_contract.json` tolerances (op_class × dtype) and disclose
     the value change in the PR;
   - **passing batch-invariance does NOT prove before/after bitwise
     equivalence** — invariance compares one implementation across batch
     shapes, not the old vs the new kernel; assert against
     pre-optimization outputs explicitly when claiming value preservation.
6. **Broadcast instead of copy**: constants/row vectors broadcast via
   `src1RepStride=0` rather than re-loading into UB.
7. **Remove coarse sync**: consecutive `PipeBarrier<PIPE_ALL>` / redundant
   flags → fine-grained (rules SYNC-09/11; see the locally installed skill
   `ascendc-sync-audit`).
8. **Avoid AiCpu ops**: `torch.argsort(int64, stable=True)` lands on AiCpu
   and is slow; switch dtype/implementation where possible (correctness
   first — see main SKILL.md gotchas).

## Verification loop

After every change, rerun the same baseline command and record the numbers:

```bash
python scripts/check_operator.py --op <op> --candidate ascend --device npu \
    --dtype fp16 --batch 2 --seq 16 --vocab 257 --normalized-dim 4096
python -m pytest tests/test_<op>_ascend.py -q   # correctness + invariance must still pass
```

- If an optimization does nothing, return to the bound diagnosis instead of
  blind iterations (same discipline as precision debugging: ≤7 attempts).
- The PR must state before/after timings, the bound-classification evidence,
  and the determinism argument (reduction order unchanged → invariance still
  bitwise).

## Further resources

Locally installed skills: `ascendc-perf-optimize` (Step 2 inter-card / Step 3
inter-core pipelines, MC² fused communication-compute),
`ascendc-performance-best-practices` (design docs and code templates for
broadcast/transpose/conversion patterns), `msopprof-visualization`
(profiling visualization).
