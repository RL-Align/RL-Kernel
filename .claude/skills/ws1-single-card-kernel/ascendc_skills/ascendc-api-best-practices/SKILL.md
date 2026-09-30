---
name: ascendc-api-best-practices
description: WS1 Ascend C API best practices and blacklist. Consult when using a specific API in a kernel - DataCopy/DataCopyPad alignment rules, Cast RoundMode, repeatTimes<=255, SetFlag/WaitFlag directions, the GlobalTensor.GetValue/SetValue ban, and the scalar-unit exp/log workaround. Trigger on API parameter errors, "VEC supports illegal configurations", wrong data after Cast, repeatTimes overflow, or when an API quick-reference is needed.
---

# WS1 Ascend C API Best Practices

> Sub-skill of `ws1-single-card-kernel`, curated from cannbot-skills
> `ascendc-api-best-practices` (https://gitcode.com/cann/cannbot-skills).
> Conclusions battle-tested in this repo (main SKILL.md gotchas) take
> precedence over the generic guidance; for full parameter tables invoke the
> locally installed skill `ascendc-api-best-practices` (23 reference docs
> under references/).

## API blacklist

| API | Why banned | Replacement |
|-----|-----------|-------------|
| `GlobalTensor::SetValue()` | extremely slow, and GM scalar writes proved unreliable here | UB staging + `S_MTE3` flag + `DataCopyPad` out |
| `GlobalTensor::GetValue()` | same | 32B `DataCopyPad` window into UB (4 int64 / 8 fp32 per window), wait `MTE2_S` then `GetValue` on the UB tensor |

For single-point debugging, stage the value through the same 32B `DataCopyPad`
window (wait `MTE2_S`) and read the UB tensor — do not call `GlobalTensor`
`GetValue`/`SetValue` even when only debugging.

## Data movement (DataCopy / DataCopyPad)

- `DataCopy` (GM↔UB) only when data is **strictly 32-byte aligned**; otherwise
  `DataCopyPad`.
- Measured here: UB→UB `DataCopy`, `Cast` and other vector ops on small counts
  fail with "VEC supports illegal configurations" → round the count up to a
  multiple of `32/sizeof(T)`. **Round BEFORE allocating**: every source and
  destination `LocalTensor` must be sized for the rounded count — the vector
  instruction touches all rounded elements, so limiting the GM writeback to
  the real byte count does NOT make the over-copy safe. Allocate
  `alignedCount * sizeof(T)` per buffer (× `BUFFER_NUM` for queues), issue
  the op with `alignedCount`, write only the real length back to GM:
  ```cpp
  constexpr uint32_t quantum = 32 / sizeof(T);              // fp16=16, fp32=8
  const uint32_t alignedCount = (len + quantum - 1) / quantum * quantum;
  pipe.InitBuffer(inQueueX, BUFFER_NUM, alignedCount * sizeof(T));
  // cast/promote buffers follow the same rule, incl. the fp32 cast buffer:
  // roundUp(innerDim) * sizeof(float) per buffer, not raw innerDim
  ```
- `DataCopyPad` handles unaligned GM sides plus zero-fill — the default for
  loads/stores in this repo.
- For multi-dimensional strided copies, verify every shape/stride field's unit
  (elements vs bytes) one by one.

## Precision conversion (Cast)

| Direction | Recommended RoundMode | Notes |
|-----------|----------------------|-------|
| half/bf16 → float | `CAST_NONE` | lossless widening |
| float → half/bf16 | generic advice `CAST_ROUND`; **this repo uses `CAST_RINT`** (IEEE round-to-nearest, bitwise-identical to CUDA `static_cast`) | the final `Cast` argument is the element count to convert: pass the real count rounded UP to a multiple of `32/sizeof(T)` — the quantum itself is NOT the count (passing literally `32/sizeof(T)` converts only that many elements) |

## Vector compute restrictions

- **repeatTimes ≤ 255**: per-repeat-instruction cap; loop in batches above it.
- `Compare` requires 256-byte alignment — plan padding accordingly.
- **The scalar unit (S) has no exp/log**: scalar math goes through a padded
  8-element vector workaround: `SetValue → S_V flag → vector Exp/Log → V_S
  wait → GetValue`.
- Reductions (`ReduceSum/ReduceMax`): prefer fp32 accumulators; fp16
  intermediates amplify error.
- Broadcast: `BinaryRepeatParams.src1RepStride = 0` broadcasts a row vector —
  saves UB and bandwidth.

## Synchronization primitives

- `SetFlag/WaitFlag` template parameter must be a `HardEvent::direction`
  matching the dataflow:

| HardEvent | Meaning |
|-----------|---------|
| `MTE2_S` | load done → scalar may read |
| `S_V` | scalar write done → vector may compute |
| `V_S` | vector compute done → scalar may read |
| `S_MTE3` | writes done → may store out |
| `MTE3_S` | store done → scalar may proceed |

- `PipeBarrier<PIPE_*>` is a coarse full stall: keep it for temporary
  verification (output becomes correct when inserted = a sync bug) and last
  resorts; production code uses queues or flags.
- Kernel-launch `GM_ADDR` parameters take `uint8_t*` (non-const).
- Full rules: the locally installed skill `ascendc-sync-audit` (14 SYNC rules).

## Buffer management (TBuf / TQue)

- `TPipe::InitBuffer` partitions UB once; `TQue(VECIN/VECOUT)` carries
  implicit synchronization.
- Double buffering (`BUFFER_NUM=2`) overlaps MTE2 with V; **tiling threshold
  math must account ×2**.
- Outputs must go through a `VECOUT` queue; using `VECIN` by mistake
  classically yields "output equals input".
- `AllocTensor` only allocates memory — it does not wait for the copy;
  `DeQue` is the synchronization point.

## Host side (this repo)

- Device/context is owned by torch_npu; the host wrapper only does shape
  checks, tiling computation, and the launch.
- Kernel-lookup / tiling errors (561xxx): see the locally installed skill
  `ascendc-runtime-debug`.
- When pip swallows the real bisheng compile error, compile the `.asc`
  manually with `bisheng` to see it.

## Quick routing

| Scenario | Use |
|----------|-----|
| Unaligned row load/store | `DataCopyPad` |
| Small-count vector op reports illegal configuration | round count to a multiple of `32/sizeof(T)` |
| Scalar needs exp/log | 8-element vector workaround |
| Read/write individual GM scalars | 32B DataCopyPad window + flags |
| Row-by-row output to GM | UB staging + `S_MTE3` + `DataCopyPad`, drain `MTE3_S` between rows |
| Bitwise match with CUDA | `Cast(out, in, CAST_RINT, roundUp(count, 32/sizeof(T)))` |
